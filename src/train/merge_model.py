"""
Day 3: 模型合并脚本
合并LoRA权重到基础模型，并固化JSON格式约束
"""
import torch
import json
import os
import glob
from peft import PeftModel, LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

# 尝试导入 safetensors
try:
    from safetensors.torch import load_file as safe_load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("⚠️  safetensors未安装，将使用标准torch.load")

# 尝试导入 ModelScope
try:
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    print("⚠️  ModelScope未安装，将使用HuggingFace。如需使用ModelScope，请运行: pip install modelscope")


def merge_lora_weights(
    base_model_name: str,
    lora_checkpoint: str,
    output_dir: str,
    trust_remote_code: bool = True,
    use_modelscope: bool = False
):
    """
    合并LoRA权重到基础模型
    
    Args:
        base_model_name: 基础模型路径
        lora_checkpoint: LoRA检查点路径
        output_dir: 输出目录
        trust_remote_code: 是否信任远程代码
        use_modelscope: 是否使用ModelScope加载（解决网络问题）
    """
    actual_model_path = base_model_name
    
    # ModelScope 模型映射
    modelscope_model_map = {
        "Qwen/Qwen3-VL-32B-Instruct": "qwen/Qwen3-VL-32B-Instruct",
        "qwen/Qwen3-VL-32B-Instruct": "qwen/Qwen3-VL-32B-Instruct",
    }
    
    # 如果使用ModelScope，先下载模型到本地
    if use_modelscope and MODELSCOPE_AVAILABLE:
        if base_model_name in modelscope_model_map:
            modelscope_name = modelscope_model_map[base_model_name]
        elif base_model_name.startswith("Qwen/") or base_model_name.startswith("qwen/"):
            modelscope_name = base_model_name.replace("Qwen/", "qwen/")
        else:
            modelscope_name = base_model_name
        
        print(f"🔄 使用ModelScope下载模型: {modelscope_name}")
        try:
            local_model_path = snapshot_download(
                modelscope_name,
                cache_dir=os.getenv("MODELSCOPE_CACHE", "./modelscope_cache")
            )
            print(f"✅ 模型已下载到本地: {local_model_path}")
            actual_model_path = local_model_path
        except Exception as e:
            print(f"❌ ModelScope下载失败: {e}")
            print("   将尝试使用HuggingFace加载...")
            use_modelscope = False
    elif use_modelscope and not MODELSCOPE_AVAILABLE:
        print("⚠️  未安装 ModelScope，请运行: pip install modelscope")
        print("   将尝试使用 HuggingFace 加载（可能需要网络连接）")
    
    # 如果是本地路径，禁用 HuggingFace 的在线检查
    if os.path.exists(actual_model_path) or actual_model_path.startswith("./") or actual_model_path.startswith("/"):
        print(f"📁 检测到本地模型路径，禁用 HuggingFace 在线检查")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    print(f"加载基础模型: {actual_model_path}")
    
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "attn_implementation": "flash_attention_2",
    }
    
    # 如果是本地路径，强制使用本地文件
    if os.path.exists(actual_model_path) or actual_model_path.startswith("./") or actual_model_path.startswith("/"):
        model_kwargs["local_files_only"] = True
    
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            actual_model_path,
            **model_kwargs
        )
    finally:
        # 恢复环境变量
        if "HF_HUB_OFFLINE" in os.environ:
            del os.environ["HF_HUB_OFFLINE"]
        if "TRANSFORMERS_OFFLINE" in os.environ:
            del os.environ["TRANSFORMERS_OFFLINE"]
    
    print(f"加载LoRA权重: {lora_checkpoint}")
    
    # 检查 LoRA 配置
    import json
    lora_config_path = os.path.join(lora_checkpoint, "adapter_config.json")
    if os.path.exists(lora_config_path):
        with open(lora_config_path, 'r', encoding='utf-8') as f:
            lora_config = json.load(f)
        print(f"   LoRA配置: r={lora_config.get('r')}, alpha={lora_config.get('lora_alpha')}")
    
    # 尝试加载 LoRA，如果遇到 AWQ 兼容性问题，使用手动加载
    try:
        model = PeftModel.from_pretrained(model, lora_checkpoint)
    except Exception as e:
        error_msg = str(e)
        if "awq" in error_msg.lower() or "PytorchGELUTanh" in error_msg or "cannot import name" in error_msg:
            print(f"⚠️  检测到 AWQ 兼容性问题，使用手动加载方式...")
            print(f"   错误: {error_msg[:150]}")
            
            # 手动加载 LoRA 权重（绕过 AWQ 检查）
            # 查找权重文件
            lora_weight_files = (
                glob.glob(os.path.join(lora_checkpoint, "adapter_model*.safetensors")) +
                glob.glob(os.path.join(lora_checkpoint, "adapter_model*.bin"))
            )
            
            if not lora_weight_files:
                raise FileNotFoundError(f"未找到 LoRA 权重文件: {lora_checkpoint}")
            
            print(f"   找到权重文件: {os.path.basename(lora_weight_files[0])}")
            
            # 读取 LoRA 配置
            lora_config_obj = LoraConfig.from_pretrained(lora_checkpoint)
            
            # 创建 PEFT 模型
            model = get_peft_model(model, lora_config_obj)
            
            # 加载权重
            weight_file = lora_weight_files[0]
            if weight_file.endswith('.safetensors') and SAFETENSORS_AVAILABLE:
                state_dict = safe_load_file(weight_file)
            else:
                state_dict = torch.load(weight_file, map_location="cpu")
            
            # 设置权重
            set_peft_model_state_dict(model, state_dict)
            print(f"   ✅ 手动加载 LoRA 权重成功")
        else:
            # 其他错误，直接抛出
            raise
    
    print("合并LoRA权重...")
    merged_model = model.merge_and_unload()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"保存合并后的模型到: {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    
    # 保存处理器（使用实际模型路径）
    processor_kwargs = {"trust_remote_code": trust_remote_code}
    if os.path.exists(actual_model_path) or actual_model_path.startswith("./") or actual_model_path.startswith("/"):
        processor_kwargs["local_files_only"] = True
    
    # 临时禁用在线检查
    if processor_kwargs.get("local_files_only", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    try:
        processor = AutoProcessor.from_pretrained(actual_model_path, **processor_kwargs)
        processor.save_pretrained(output_dir)
    finally:
        if "HF_HUB_OFFLINE" in os.environ:
            del os.environ["HF_HUB_OFFLINE"]
        if "TRANSFORMERS_OFFLINE" in os.environ:
            del os.environ["TRANSFORMERS_OFFLINE"]
    
    # 保存配置（添加JSON格式约束）
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {}
    
    # 添加JSON格式约束配置
    config["forced_json"] = True
    config["json_schema"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "defect": {
                    "type": "string",
                    "enum": ["short", "open", "missing", "normal"]
                },
                "bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4
                },
                "repair": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["defect", "bbox", "repair"]
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # 简单验证合并后模型的前向传播，确保权重未损坏
    try:
        print("验证合并后的模型...")
        merged_model.eval()
        device = next(merged_model.parameters()).device
        dummy_image = Image.new("RGB", (224, 224), color="white")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Test"}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=dummy_image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = merged_model(**inputs)
        if torch.isnan(outputs.logits).any():
            raise RuntimeError("合并后模型输出NaN，权重已损坏！")
        if outputs.logits.var() < 1e-6:
            raise RuntimeError("合并后模型输出方差过小，权重可能未正确加载！")
        print("✅ 合并模型验证通过")
    except Exception as e:
        print(f"❌ 合并后验证失败: {e}")
        raise
    
    print("✅ 模型合并完成！")
    print(f"   输出目录: {output_dir}")
    print(f"   JSON格式约束已固化到config.json")
    
    # 计算模型大小
    total_size = 0
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    
    print(f"   模型大小: {total_size / 1e9:.2f} GB")
    
    return merged_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="合并LoRA权重到基础模型")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-32B-Instruct", 
                       help="基础模型路径")
    parser.add_argument("--lora_checkpoint", type=str, required=True,
                       help="LoRA检查点路径")
    parser.add_argument("--output_dir", type=str, default="./models/qwen3-vl-pcb",
                       help="输出目录")
    parser.add_argument("--use_modelscope", action="store_true",
                       help="使用ModelScope加载模型（解决网络问题）")
    
    args = parser.parse_args()
    
    merge_lora_weights(
        base_model_name=args.base_model,
        lora_checkpoint=args.lora_checkpoint,
        output_dir=args.output_dir,
        use_modelscope=args.use_modelscope
    )

