"""
Day 1-2: 电路板缺陷MLLM微调脚本
使用Qwen3-VL-32B-Instruct + LoRA进行微调
支持 HuggingFace 和 ModelScope 两种加载方式
"""
import torch
import os
import numpy as np
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.data_loader import load_pcb_dataset
import json

# 尝试导入 ModelScope
try:
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    print("⚠️  ModelScope未安装，将使用HuggingFace。如需使用ModelScope，请运行: pip install modelscope")


class PCBDataCollator:
    """使用官方 chat template 构造输入与标签"""
    
    def __init__(self, processor, max_length=2048):
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, batch):
        images = [item["image"] for item in batch]

        messages = []
        for item in batch:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": item["question"] + "\n请输出JSON数组。"},
                ],
            }
            assistant_msg = {"role": "assistant", "content": item["answer"]}
            messages.append([user_msg, assistant_msg])

        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False
            )
            for m in messages
        ]

        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        # 确保输入在正确的设备上，并且保持梯度连接
        # 对于视觉-语言模型，图像不需要梯度（视觉编码器被冻结），
        # 但input_ids需要通过模型产生梯度
        labels = inputs["input_ids"].clone().detach()  # labels不需要梯度
        # input_ids保持原样，让模型的前向传播产生梯度
        
        # 获取 assistant token ID（Qwen3-VL 使用特殊 token）
        assistant_token_id = None
        try:
            # 直接尝试获取 assistant token ID
            if hasattr(self.processor.tokenizer, "convert_tokens_to_ids"):
                token_id = self.processor.tokenizer.convert_tokens_to_ids("<|assistant|>")
                if token_id is not None and token_id != self.processor.tokenizer.unk_token_id:
                    assistant_token_id = token_id
        except Exception:
            pass

        # 掩码标签：只保留 assistant 响应部分
        input_ids = inputs["input_ids"]  # 确保是张量
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        
        if assistant_token_id is not None and isinstance(assistant_token_id, (int, torch.Tensor)):
            # 确保 assistant_token_id 是标量
            if isinstance(assistant_token_id, torch.Tensor):
                assistant_token_id = assistant_token_id.item() if assistant_token_id.numel() == 1 else int(assistant_token_id)
            
            # 找到所有 assistant token 的位置
            for i in range(input_ids.shape[0]):
                # 确保比较操作返回张量（使用 torch.eq 更安全）
                matches = torch.eq(input_ids[i], assistant_token_id)
                assistant_positions = torch.nonzero(matches, as_tuple=False)
                if len(assistant_positions) > 0:
                    start_pos = assistant_positions[0].item() + 1  # assistant token 之后开始
                    labels[i, :start_pos] = -100
        else:
            # 如果找不到 assistant token，通过文本分析来确定位置
            for i, text in enumerate(texts):
                if "<|assistant|>" in text:
                    # 对完整文本进行 tokenize
                    full_encoded = self.processor.tokenizer.encode(text, add_special_tokens=False)
                    # 找到 "<|assistant|>" 在文本中的位置
                    assistant_part = text.split("<|assistant|>")[-1]
                    assistant_encoded = self.processor.tokenizer.encode(assistant_part, add_special_tokens=False)
                    # 在完整序列中查找 assistant 部分的起始位置
                    if len(assistant_encoded) > 0:
                        for j in range(len(full_encoded) - len(assistant_encoded) + 1):
                            if full_encoded[j:j+len(assistant_encoded)] == assistant_encoded:
                                # 找到匹配位置，掩码之前的部分
                                labels[i, :j] = -100
                                break
        
        # 掩码 padding token
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        inputs["labels"] = labels
        return inputs


def setup_model(model_name: str = "Qwen/Qwen3-VL-32B-Instruct", 
                use_4bit: bool = True,
                device_map: str = "auto",
                model_revision: str = "main",
                use_modelscope: bool = False):
    """
    加载并配置模型
    
    Args:
        model_name: 模型名称（HuggingFace格式或ModelScope格式）
        use_4bit: 是否使用4-bit量化加载
        device_map: 设备映射策略
        model_revision: 模型版本（commit/tag），用于锁定版本
    """
    print(f"加载模型: {model_name}")
    
    # ModelScope 模型映射
    modelscope_model_map = {
        "Qwen/Qwen3-VL-32B-Instruct": "qwen/Qwen3-VL-32B-Instruct",
        "qwen/Qwen3-VL-32B-Instruct": "qwen/Qwen3-VL-32B-Instruct",
    }
    
    # 如果使用ModelScope，先下载模型到本地
    if use_modelscope and MODELSCOPE_AVAILABLE:
        if model_name in modelscope_model_map:
            modelscope_name = modelscope_model_map[model_name]
        elif model_name.startswith("Qwen/") or model_name.startswith("qwen/"):
            modelscope_name = model_name.replace("Qwen/", "qwen/")
        else:
            modelscope_name = model_name
        
        print(f"使用ModelScope下载模型: {modelscope_name}")
        try:
            # 下载模型到本地缓存
            local_model_path = snapshot_download(
                modelscope_name,
                cache_dir=os.getenv("MODELSCOPE_CACHE", "./modelscope_cache")
            )
            print(f"模型已下载到: {local_model_path}")
            model_name = local_model_path  # 使用本地路径
        except Exception as e:
            print(f"⚠️  ModelScope下载失败: {e}")
            print("   将尝试使用HuggingFace加载...")
            use_modelscope = False
    
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "torch_dtype": torch.float16,
        # 不强制使用 flash_attention_2，避免环境未安装 FlashAttention2 报错
        "local_files_only": False,  # 允许从本地加载
        "revision": model_revision,
    }
    
    # 如果是本地路径，强制使用本地文件
    if os.path.exists(model_name) or model_name.startswith("./") or model_name.startswith("/"):
        model_kwargs["local_files_only"] = True
        print(f"使用本地模型路径: {model_name}")
    
    if use_4bit:
        from transformers import BitsAndBytesConfig
        # 注意：4-bit量化主要用于推理，LoRA训练时可能不兼容
        # 如果遇到梯度问题，建议使用 --no_4bit 禁用量化
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quantization_config
        print("⚠️  使用4-bit量化加载模型。")
        print("   如果训练时出现梯度错误，请使用 --no_4bit 禁用量化。")
    
    # 临时禁用 HuggingFace 的在线检查（如果使用本地路径）
    if model_kwargs.get("local_files_only", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
    
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    finally:
        # 恢复环境变量
        if "HF_HUB_OFFLINE" in os.environ:
            del os.environ["HF_HUB_OFFLINE"]
    
    # 注意：视觉塔的冻结应该在LoRA应用之后进行
    # 因为LoRA会创建新的可训练参数，我们需要确保视觉塔部分被冻结
    # 但这里先不冻结，等LoRA应用后再处理
    
    return model


def setup_lora(model, r=16, alpha=32, dropout=0.05):
    """
    配置LoRA
    
    Args:
        model: 基础模型
        r: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
    """
    # 禁用 AWQ 检测（避免导入错误）
    import os
    os.environ["PEFT_DISABLE_AWQ"] = "1"
    
    # 找到所有线性层名称
    target_modules = []
    for name, module in model.named_modules():
        if any(layer in name for layer in ["q_proj", "k_proj", "v_proj", "o_proj",
                                            "gate_proj", "up_proj", "down_proj"]):
            target_modules.append(name.split('.')[-1])  # 只取最后一层名称
    
    # 去重
    target_modules = list(set(target_modules))
    
    print(f"LoRA目标模块: {target_modules}")
    
    lora_config = LoraConfig(
        r=r,  # 缺陷模式比通用视觉简单，16足够
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",  # Vision2Seq使用因果语言模型任务类型
        modules_to_save=None,  # 不保存额外模块
    )
    
    # 临时禁用 AWQ 相关的导入
    import sys
    awq_modules = [k for k in sys.modules.keys() if 'awq' in k.lower()]
    for mod in awq_modules:
        if mod in sys.modules:
            del sys.modules[mod]
    
    try:
        # 确保模型处于训练模式
        model.train()
    model = get_peft_model(model, lora_config)
        # 再次确保训练模式
        model.train()
    except ImportError as e:
        if 'awq' in str(e).lower() or 'PytorchGELUTanh' in str(e):
            print("\n" + "="*60)
            print("❌ 错误：检测到 AWQ 库兼容性问题")
            print("   解决方案：卸载过时的 awq 库")
            print("   命令：pip uninstall -y autoawq awq")
            print("="*60 + "\n")
            raise RuntimeError(
                "AWQ 库与当前 transformers 版本不兼容。\n"
                "请运行: pip uninstall -y autoawq awq\n"
                "然后重新运行训练命令。"
            ) from e
        raise
    
    # 打印LoRA适配器信息
    print("\n📊 LoRA适配器信息:")
    if hasattr(model, "peft_config"):
        for adapter_name, adapter_config in model.peft_config.items():
            print(f"  适配器: {adapter_name}")
            print(f"  - rank: {adapter_config.r}")
            print(f"  - alpha: {adapter_config.lora_alpha}")
            print(f"  - target_modules: {adapter_config.target_modules}")
    
    # 确保LoRA适配器是可训练的
    # 对于量化模型，需要显式启用LoRA参数的梯度
    lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            lora_params.append(name)
    
    print(f"\n🔧 找到 {len(lora_params)} 个LoRA参数组")
    if len(lora_params) > 0:
        print(f"  示例LoRA参数: {lora_params[:3]}...")
    
    # 冻结视觉塔，避免在LoRA时破坏视觉特征
    # 注意：这应该在LoRA应用之后进行，确保LoRA参数不受影响
    vision_frozen = 0
    for name, param in model.named_parameters():
        if ("vision" in name.lower() or "visual" in name.lower()) and "lora" not in name.lower():
            param.requires_grad = False
            vision_frozen += 1
    
    if vision_frozen > 0:
        print(f"🔒 冻结了 {vision_frozen} 个视觉塔参数")
    
    # 打印可训练参数信息
    print("\n📈 可训练参数统计:")
    model.print_trainable_parameters()
    
    # 验证至少有一些参数是可训练的
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("❌ 错误：没有可训练的参数！LoRA适配器可能未正确创建。")
    
    # 检查可训练参数是否有梯度函数
    trainable_with_grad_fn = [p for p in trainable_params if p.requires_grad and p.grad_fn is not None]
    print(f"✅ 找到 {len(trainable_params)} 个可训练参数，{len(trainable_with_grad_fn)} 个有梯度函数")
    
    if len(trainable_with_grad_fn) == 0 and len(trainable_params) > 0:
        print("⚠️  警告：可训练参数存在但没有梯度函数。这可能在训练时导致问题。")
    
    return model


def train_pcb_model(
    data_dir: str,
    output_dir: str = "./checkpoints/pcb_checkpoints",
    model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
    max_steps: int = 2000,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    use_4bit: bool = True,
    save_steps: int = 500,
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_modelscope: bool = False,
    model_revision: str = "main",
):
    """
    训练PCB缺陷检测模型
    
    Args:
        data_dir: 数据集目录
        output_dir: 输出目录
        model_name: 基础模型名称
        max_steps: 最大训练步数
        batch_size: 批次大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        use_4bit: 是否使用4-bit量化
        save_steps: 保存步数间隔
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        model_revision: 模型版本（commit/tag），用于锁定版本
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集
    print("加载数据集...")
    train_dataset = load_pcb_dataset(data_dir, augment=True)
    print(f"数据集大小: {len(train_dataset)}")
    
    # 加载模型和处理器
    print("加载模型...")
    actual_model_path = model_name
    skip_download = False
    
    # 首先检查是否是本地路径且已存在
    if os.path.exists(model_name) and os.path.exists(os.path.join(model_name, "config.json")):
        print(f"✅ 使用本地已有模型: {model_name}")
        actual_model_path = model_name
        skip_download = True
    # 如果不是本地路径，且需要使用ModelScope
    elif use_modelscope and MODELSCOPE_AVAILABLE:
        modelscope_model_map = {
            "Qwen/Qwen3-VL-32B-Instruct": "qwen/Qwen3-VL-32B-Instruct",
            "qwen/Qwen3-VL-32B-Instruct": "qwen/Qwen3-VL-32B-Instruct",
        }
        if model_name in modelscope_model_map:
            modelscope_name = modelscope_model_map[model_name]
        elif model_name.startswith("Qwen/") or model_name.startswith("qwen/"):
            modelscope_name = model_name.replace("Qwen/", "qwen/")
        else:
            modelscope_name = model_name
        
        # 检查ModelScope缓存中是否已有模型（检查多个可能的缓存位置）
        cache_dir = os.getenv("MODELSCOPE_CACHE", "./modelscope_cache")
        possible_cache_paths = [
            os.path.join(cache_dir, modelscope_name.replace("/", "--")),
            os.path.join(cache_dir, modelscope_name),
            os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", modelscope_name),
        ]
        
        cached_path = None
        for cache_path in possible_cache_paths:
            if os.path.exists(cache_path) and os.path.exists(os.path.join(cache_path, "config.json")):
                cached_path = cache_path
                break
        
        if cached_path:
            print(f"✅ 使用ModelScope缓存中的模型: {cached_path}")
            actual_model_path = cached_path
            skip_download = True
        else:
            # 只有在缓存中找不到时才下载
            print(f"🔄 从ModelScope下载模型: {modelscope_name}")
        print("   这将避免网络连接问题...")
        try:
            actual_model_path = snapshot_download(
                modelscope_name,
                    cache_dir=cache_dir
            )
            print(f"✅ 模型已下载到本地: {actual_model_path}")
        except Exception as e:
            print(f"❌ ModelScope下载失败: {e}")
            print("   将尝试使用HuggingFace加载...")
            use_modelscope = False
    elif use_modelscope and not MODELSCOPE_AVAILABLE:
        print("⚠️  未安装 ModelScope，请运行: pip install modelscope")
        print("   将尝试使用 HuggingFace 加载（可能需要网络连接）")
    
    # 禁用 HuggingFace 的在线检查（如果使用本地路径）
    if os.path.exists(actual_model_path) or actual_model_path.startswith("./") or actual_model_path.startswith("/"):
        print(f"📁 检测到本地模型路径，禁用 HuggingFace 在线检查")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    try:
        model = setup_model(
            actual_model_path,
            use_4bit=use_4bit,
            device_map="auto",
            model_revision=model_revision,
            use_modelscope=use_modelscope,
        )  # 已下载，直接使用本地路径
        
        # Processor 也使用本地路径，并锁定版本
        processor_kwargs = {"trust_remote_code": True, "revision": model_revision}
        if os.path.exists(actual_model_path) or actual_model_path.startswith("./") or actual_model_path.startswith("/"):
            processor_kwargs["local_files_only"] = True
        
        # 关键文件完整性检查，防止静默回退
        # 检查 tokenizer.json（必需）
        if not os.path.exists(os.path.join(actual_model_path, "tokenizer.json")):
            print(f"❌ 本地模型缺少关键文件: tokenizer.json")
            print("   请重新下载或指定完整的模型目录")
            raise FileNotFoundError(f"模型文件不完整: tokenizer.json")
        
        # 检查 processor 配置文件（preprocessor_config.json 或 processor_config.json 之一即可）
        has_preprocessor = os.path.exists(os.path.join(actual_model_path, "preprocessor_config.json"))
        has_processor = os.path.exists(os.path.join(actual_model_path, "processor_config.json"))
        if not (has_preprocessor or has_processor):
            print(f"⚠️  本地模型缺少 processor 配置文件（preprocessor_config.json 或 processor_config.json）")
            print("   将尝试继续加载，如果失败请重新下载模型")
        else:
            config_type = "preprocessor_config.json" if has_preprocessor else "processor_config.json"
            print(f"✅ 找到 processor 配置文件: {config_type}")
        
        processor = AutoProcessor.from_pretrained(actual_model_path, **processor_kwargs)
    finally:
        # 恢复环境变量
        if "HF_HUB_OFFLINE" in os.environ:
            del os.environ["HF_HUB_OFFLINE"]
        if "TRANSFORMERS_OFFLINE" in os.environ:
            del os.environ["TRANSFORMERS_OFFLINE"]
    
    # 配置LoRA
    print("配置LoRA...")
    model = setup_lora(model, r=lora_r, alpha=lora_alpha)
    
    # 确保模型处于训练模式
    model.train()
    
    # 最终验证：确保模型有可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "❌ 致命错误：模型没有可训练参数！\n"
            "可能的原因：\n"
            "1. LoRA适配器未正确创建\n"
            "2. 4-bit量化与LoRA不兼容\n"
            "3. 所有参数被意外冻结\n"
            "建议：尝试禁用4-bit量化（使用 --no_4bit 参数）"
        )
    
    # 检查可训练参数的梯度状态
    trainable_with_grad = sum(1 for p in trainable_params if p.requires_grad)
    print(f"\n✅ 训练前验证：{trainable_with_grad}/{len(trainable_params)} 个可训练参数已启用梯度")
    
    # 确保模型的所有可训练参数都正确设置
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 确保参数不是detached的
            if not param.is_leaf or param.grad_fn is not None:
                # 如果参数有grad_fn，说明它依赖于其他计算，这是正常的
                pass
            # 对于LoRA参数，它们应该是leaf节点，但requires_grad=True
            if "lora" in name.lower() and not param.requires_grad:
                print(f"⚠️  警告：LoRA参数 {name} 的 requires_grad=False，强制设置为True")
                param.requires_grad = True
    
    # 如果使用4-bit量化，给出警告
    if use_4bit:
        print("\n" + "="*60)
        print("⚠️  重要提示：使用4-bit量化进行LoRA训练")
        print("   如果训练时出现 'does not require grad' 错误，")
        print("   请使用 --no_4bit 参数禁用4-bit量化。")
        print("="*60 + "\n")
    
    # 数据整理器
    data_collator = PCBDataCollator(processor, max_length=2048)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        save_steps=save_steps,
        fp16=True,
        save_only_model=False,  # 保存完整模型，便于直接推理
        logging_steps=10,  # 更频繁的日志，便于早期发现问题
        warmup_ratio=0.1,
        weight_decay=0.01,
        report_to="none",
        remove_unused_columns=False,  # 保留所有列，由data_collator处理
        max_grad_norm=0.3,  # 更严格的梯度裁剪
        dataloader_pin_memory=True,
        gradient_checkpointing=False,  # 暂时禁用，避免梯度连接问题
        fp16_full_eval=False,
        logging_nan_inf_filter=True,
    )
    
    # 早停回调（防止过拟合）
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if len(train_dataset) < 1000 else []
    
    # 8bit AdamW 优化器（与4bit加载配合更稳）
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(model.parameters(), lr=learning_rate)
        optimizers = (optimizer, None)
    except Exception as e:
        print(f"⚠️  未安装bitsandbytes，回退到默认AdamW: {e}")
        optimizers = (None, None)
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        optimizers=optimizers,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 合并LoRA权重并保存完整模型
    print("正在合并LoRA权重并保存最终模型...")
    model.eval()
    if hasattr(model, "merge_and_unload"):
        merged_model = model.merge_and_unload()
        print("✅ LoRA权重已合并")
    else:
        merged_model = model
        print("⚠️ 未检测到 merge_and_unload，直接保存当前模型")

    final_model_path = os.path.join(output_dir, "final")
    merged_model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    
    print(f"🎉 训练完成！完整模型保存在: {final_model_path}")
    
    return merged_model, processor


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练PCB缺陷检测模型")
    parser.add_argument("--data_dir", type=str, required=True, help="数据集目录")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/pcb_checkpoints", help="输出目录")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-32B-Instruct", help="基础模型名称")
    parser.add_argument("--max_steps", type=int, default=2000, help="最大训练步数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--save_steps", type=int, default=500, help="保存checkpoint的步数间隔")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--no_4bit", action="store_true", help="不使用4-bit量化")
    parser.add_argument("--use_modelscope", action="store_true", help="使用ModelScope加载模型（解决网络问题）")
    parser.add_argument("--model_revision", type=str, default="main", help="模型版本（commit hash或tag）")
    
    args = parser.parse_args()
    
    train_pcb_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        use_4bit=not args.no_4bit,
        save_steps=args.save_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_modelscope=args.use_modelscope,
        model_revision=args.model_revision,
    )

