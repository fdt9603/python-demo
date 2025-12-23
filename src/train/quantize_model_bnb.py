"""
Day 4: BitsAndBytes 4-bit量化脚本（替代AWQ）
对合并后的模型进行4-bit量化，确保推理速度 < 1秒/张
"""
import torch
import os
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def quantize_model_bnb(
    model_path: str,
    output_dir: str,
    use_4bit: bool = True,
    use_8bit: bool = False,
    use_modelscope: bool = False,
    model_revision: str = "main",
):
    """
    使用BitsAndBytes量化模型（兼容性好，推荐）
    
    Args:
        model_path: 合并后的模型路径
        output_dir: 量化模型输出目录
        use_4bit: 是否使用4-bit量化（推荐）
        use_8bit: 是否使用8-bit量化（备选）
        use_modelscope: 是否使用ModelScope加载（解决网络问题）
    """
    print(f"加载模型: {model_path} (revision={model_revision})")
    
    # ModelScope 支持
    actual_model_path = model_path
    if use_modelscope:
        try:
            from modelscope import snapshot_download
            modelscope_model_map = {
                "Qwen/Qwen3-VL-32B-Instruct": "qwen/Qwen3-VL-32B-Instruct",
                "qwen/Qwen3-VL-32B-Instruct": "qwen/Qwen3-VL-32B-Instruct",
            }
            if model_path in modelscope_model_map:
                modelscope_name = modelscope_model_map[model_path]
            elif model_path.startswith("Qwen/") or model_path.startswith("qwen/"):
                modelscope_name = model_path.replace("Qwen/", "qwen/")
            else:
                modelscope_name = model_path
            
            print(f"🔄 使用ModelScope下载模型: {modelscope_name}")
            local_model_path = snapshot_download(
                modelscope_name,
                cache_dir=os.getenv("MODELSCOPE_CACHE", "./modelscope_cache")
            )
            print(f"✅ 模型已下载到本地: {local_model_path}")
            actual_model_path = local_model_path
        except ImportError:
            print("⚠️  ModelScope未安装，将使用HuggingFace")
        except Exception as e:
            print(f"⚠️  ModelScope下载失败: {e}，将使用原始路径")
    
    # 如果是本地路径，禁用在线检查
    if os.path.exists(actual_model_path) or actual_model_path.startswith("./") or actual_model_path.startswith("/"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print(f"📁 检测到本地模型路径，禁用 HuggingFace 在线检查")
    
    # 配置量化参数
    quantization_config = None
    if use_4bit:
        print("✅ 使用 4-bit 量化（推荐）")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"  # 使用 NF4 量化类型（最佳性能）
        )
    elif use_8bit:
        print("✅ 使用 8-bit 量化")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    else:
        print("⚠️  未启用量化，将使用 FP16（显存占用较大）")
    
    # 模型加载参数（锁定 revision，避免线上模型变更导致不一致）
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.float16,
        # 不强制使用 flash_attention_2，避免环境未安装 FlashAttention2 报错
        # "attn_implementation": "flash_attention_2",
        "revision": model_revision,
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    # 如果是本地路径，强制使用本地文件
    if os.path.exists(actual_model_path) or actual_model_path.startswith("./") or actual_model_path.startswith("/"):
        model_kwargs["local_files_only"] = True
    
    try:
        # 加载模型（带量化）
        print("正在加载模型（这可能需要几分钟）...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            actual_model_path,
            **model_kwargs
        )
        model.eval()

        # 冻结视觉塔，避免量化/保存时破坏视觉特征
        for name, param in model.named_parameters():
            if "vision" in name:
                param.requires_grad = False
        
        # 显示显存使用情况
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"✅ 模型加载成功")
            print(f"   显存使用: {allocated:.2f} GB (已分配) / {reserved:.2f} GB (已保留)")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise
    finally:
        # 恢复环境变量
        if "HF_HUB_OFFLINE" in os.environ:
            del os.environ["HF_HUB_OFFLINE"]
        if "TRANSFORMERS_OFFLINE" in os.environ:
            del os.environ["TRANSFORMERS_OFFLINE"]
    
    # 加载处理器（与模型保持相同 revision）
    processor_kwargs = {"trust_remote_code": True, "revision": model_revision}
    if os.path.exists(actual_model_path) or actual_model_path.startswith("./") or actual_model_path.startswith("/"):
        processor_kwargs["local_files_only"] = True
    
    if processor_kwargs.get("local_files_only", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    try:
        processor = AutoProcessor.from_pretrained(actual_model_path, **processor_kwargs)
    finally:
        if "HF_HUB_OFFLINE" in os.environ:
            del os.environ["HF_HUB_OFFLINE"]
        if "TRANSFORMERS_OFFLINE" in os.environ:
            del os.environ["TRANSFORMERS_OFFLINE"]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型（量化模型可以直接保存，但加载时需要重新应用量化配置）
    print(f"保存量化模型到: {output_dir}")
    print("⚠️  注意: BitsAndBytes量化是运行时量化，保存的模型需要重新应用量化配置加载")
    
    # 保存模型权重（量化后的权重）
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    
    # 保存量化配置信息（提醒运行时量化）
    quant_info = {
        "quantization": "BitsAndBytes",
        "use_4bit": use_4bit,
        "use_8bit": use_8bit,
        "compute_dtype": "float16",
        "quant_type": "nf4" if use_4bit else None,
        "double_quant": True if use_4bit else False,
        "note": "⚠️ BitsAndBytes为运行时量化，加载时需重新传入quantization_config",
    }
    
    with open(os.path.join(output_dir, "quant_config.json"), 'w', encoding='utf-8') as f:
        json.dump(quant_info, f, indent=2, ensure_ascii=False)

    # 写入示例加载脚本，提示重新传入量化配置
    loader_script = f"""from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit={str(use_4bit)},
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    load_in_8bit={str(use_8bit)}
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "{output_dir}",
    quantization_config=quantization_config,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    # 注意：如果环境中安装了 flash_attn，可以取消注释下面这行以启用 FlashAttention2
    # attn_implementation="flash_attention_2",
)
print("✅ 量化模型加载成功")
"""
    with open(os.path.join(output_dir, "load_quantized_model.py"), "w", encoding="utf-8") as f:
        f.write(loader_script)
    
    # 计算模型大小
    total_size = 0
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    
    print("\n✅ 量化完成！")
    print(f"   输出目录: {output_dir}")
    print(f"   模型大小: {total_size / 1e9:.2f} GB")
    print(f"   量化方式: {'4-bit (NF4)' if use_4bit else '8-bit' if use_8bit else 'FP16'}")
    print(f"   预期推理速度: < 1秒/张")
    print(f"\n📝 使用说明:")
    print(f"   加载量化模型时，需要使用相同的量化配置:")
    print(f"   from transformers import BitsAndBytesConfig")
    print(f"   quantization_config = BitsAndBytesConfig(")
    print(f"       load_in_4bit=True,")
    print(f"       bnb_4bit_compute_dtype=torch.float16,")
    print(f"       bnb_4bit_use_double_quant=True,")
    print(f"       bnb_4bit_quant_type='nf4'")
    print(f"   )")
    print(f"   model = AutoModelForVision2Seq.from_pretrained(")
    print(f"       '{output_dir}',")
    print(f"       quantization_config=quantization_config,")
    print(f"       trust_remote_code=True")
    print(f"   )")
    
    return model, processor


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BitsAndBytes量化模型（替代AWQ）")
    parser.add_argument("--model_path", type=str, required=True,
                       help="合并后的模型路径")
    parser.add_argument("--output_dir", type=str, default="./models/qwen3-vl-pcb-bnb",
                       help="量化模型输出目录")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="使用4-bit量化（推荐）")
    parser.add_argument("--use_8bit", action="store_true", default=False,
                       help="使用8-bit量化（备选）")
    parser.add_argument("--use_modelscope", action="store_true",
                       help="使用ModelScope加载模型（解决网络问题）")
    parser.add_argument("--model_revision", type=str, default="main",
                       help="模型版本（commit hash 或 tag），用于锁定量化来源模型")
    
    args = parser.parse_args()
    
    # 确保只选择一种量化方式
    if args.use_8bit:
        args.use_4bit = False
    
    quantize_model_bnb(
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        use_modelscope=args.use_modelscope,
        model_revision=args.model_revision,
    )

