"""
Day 3: 模型合并脚本
合并LoRA权重到基础模型，并固化JSON格式约束
"""
import torch
import json
import os
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor


def merge_lora_weights(
    base_model_name: str,
    lora_checkpoint: str,
    output_dir: str,
    trust_remote_code: bool = True
):
    """
    合并LoRA权重到基础模型
    
    Args:
        base_model_name: 基础模型路径
        lora_checkpoint: LoRA检查点路径
        output_dir: 输出目录
        trust_remote_code: 是否信任远程代码
    """
    print(f"加载基础模型: {base_model_name}")
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"加载LoRA权重: {lora_checkpoint}")
    model = PeftModel.from_pretrained(model, lora_checkpoint)
    
    print("合并LoRA权重...")
    merged_model = model.merge_and_unload()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"保存合并后的模型到: {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    
    # 保存处理器
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=trust_remote_code)
    processor.save_pretrained(output_dir)
    
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
    
    args = parser.parse_args()
    
    merge_lora_weights(
        base_model_name=args.base_model,
        lora_checkpoint=args.lora_checkpoint,
        output_dir=args.output_dir
    )

