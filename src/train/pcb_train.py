"""
Day 1-2: 电路板缺陷MLLM微调脚本
使用Qwen3-VL-32B-Instruct + LoRA进行微调
"""
import torch
import os
from transformers import (
    AutoModelForVision2Seq,
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


class PCBDataCollator:
    """PCB数据整理器：强制输出JSON格式"""
    
    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, batch):
        # 在prompt中强制约束输出格式
        texts = []
        for item in batch:
            # 关键：在prompt末尾加格式约束
            constrained_prompt = item["question"] + "\n\n请严格返回JSON数组，不要额外解释。"
            texts.append(constrained_prompt)
        
        # 准备图像
        images = [item["image"] for item in batch]
        
        # 处理输入
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
        )
        
        # 准备标签（answer）
        answers = [item["answer"] for item in batch]
        labels = self.processor.tokenizer(
            answers,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )["input_ids"]
        
        # 将labels中padding token的位置设为-100（忽略loss计算）
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        inputs["labels"] = labels
        return inputs


def setup_model(model_name: str = "Qwen/Qwen3-VL-32B-Instruct", 
                use_4bit: bool = True,
                device_map: str = "auto"):
    """
    加载并配置模型
    
    Args:
        model_name: 模型名称
        use_4bit: 是否使用4-bit量化加载
        device_map: 设备映射策略
    """
    print(f"加载模型: {model_name}")
    
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "torch_dtype": torch.float16,
    }
    
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_kwargs["quantization_config"] = quantization_config
    
    model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs)
    
    # 冻结视觉塔（电路板缺陷检测不需要调视觉特征）
    frozen_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "vision_tower" in name or "vision_model" in name:
            param.requires_grad = False
            frozen_params += param.numel()
    
    print(f"冻结参数: {frozen_params / 1e9:.2f}B / {total_params / 1e9:.2f}B")
    
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
        task_type="CAUSAL_LM"  # Vision2Seq使用因果语言模型任务类型
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def train_pcb_model(
    data_dir: str,
    output_dir: str = "./checkpoints/pcb_checkpoints",
    model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
    max_steps: int = 2000,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 5e-4,
    use_4bit: bool = True,
    save_steps: int = 500,
    lora_r: int = 16,
    lora_alpha: int = 32,
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
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集
    print("加载数据集...")
    train_dataset = load_pcb_dataset(data_dir, augment=True)
    print(f"数据集大小: {len(train_dataset)}")
    
    # 加载模型和处理器
    print("加载模型...")
    model = setup_model(model_name, use_4bit=use_4bit)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # 配置LoRA
    print("配置LoRA...")
    model = setup_lora(model, r=lora_r, alpha=lora_alpha)
    
    # 数据整理器
    data_collator = PCBDataCollator(processor, max_length=512)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        save_steps=save_steps,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=not torch.cuda.is_bf16_supported(),
        save_only_model=True,
        logging_steps=50,
        eval_strategy="no",
        warmup_steps=100,
        weight_decay=0.01,
        report_to="none",
    )
    
    # 早停回调（防止过拟合）
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if len(train_dataset) < 1000 else []
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, "final")
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    
    print(f"训练完成！模型保存在: {final_model_path}")
    
    return model, processor


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练PCB缺陷检测模型")
    parser.add_argument("--data_dir", type=str, required=True, help="数据集目录")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/pcb_checkpoints", help="输出目录")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-32B-Instruct", help="基础模型名称")
    parser.add_argument("--max_steps", type=int, default=2000, help="最大训练步数")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="学习率")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--no_4bit", action="store_true", help="不使用4-bit量化")
    
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
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

