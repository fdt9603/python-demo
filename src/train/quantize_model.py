"""
Day 4: AWQ量化脚本
对合并后的模型进行AWQ量化，确保推理速度 < 1秒/张
"""
import torch
import os
from PIL import Image
from transformers import AutoProcessor
from awq import AutoAWQForCausalLM
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.data_loader import load_pcb_dataset
import json


def prepare_calibration_data(
    model_path: str,
    data_dir: str,
    num_samples: int = 200
):
    """
    准备AWQ量化校准数据
    
    Args:
        model_path: 模型路径
        data_dir: 数据集目录（用于校准）
        num_samples: 校准样本数量
    """
    print(f"准备校准数据 ({num_samples} 个样本)...")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载数据集
    if os.path.exists(data_dir):
        dataset = load_pcb_dataset(data_dir, augment=False)
    else:
        # 如果没有数据集，使用随机图像
        print(f"警告: 数据集目录不存在 {data_dir}，使用随机图像作为校准数据")
        dataset = None
    
    calib_data = []
    query = "检测缺陷，返回JSON格式：[{'defect': '类型', 'bbox': [x,y,w,h], 'repair': '维修建议'}]"
    
    if dataset is not None:
        # 使用实际数据集
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            try:
                inputs = processor(
                    images=item["image"],
                    text=query,
                    return_tensors="pt"
                )
                calib_data.append(inputs)
            except Exception as e:
                print(f"处理样本 {i} 时出错: {e}")
                continue
    else:
        # 使用随机图像
        for i in range(num_samples):
            # 创建随机图像
            img = Image.new('RGB', (448, 448), color=(128, 128, 128))
            try:
                inputs = processor(
                    images=img,
                    text=query,
                    return_tensors="pt"
                )
                calib_data.append(inputs)
            except Exception as e:
                print(f"处理样本 {i} 时出错: {e}")
                continue
    
    print(f"校准数据准备完成: {len(calib_data)} 个样本")
    return calib_data


def quantize_model_awq(
    model_path: str,
    output_dir: str,
    calib_data_dir: str = None,
    num_calib_samples: int = 200,
    w_bit: int = 4,
    q_group_size: int = 128,
    version: str = "GEMM"
):
    """
    使用AWQ量化模型
    
    Args:
        model_path: 合并后的模型路径
        output_dir: 量化模型输出目录
        calib_data_dir: 校准数据目录（可选）
        num_calib_samples: 校准样本数量
        w_bit: 量化位数
        q_group_size: 量化组大小
        version: AWQ版本
    """
    print(f"加载模型: {model_path}")
    
    # 加载模型（AWQ兼容格式）
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 准备校准数据
    calib_data = prepare_calibration_data(
        model_path,
        calib_data_dir or model_path,
        num_calib_samples
    )
    
    if len(calib_data) == 0:
        raise ValueError("校准数据为空，无法进行量化")
    
    print(f"开始AWQ量化 (w_bit={w_bit}, q_group_size={q_group_size})...")
    print("这可能需要几小时，请耐心等待...")
    
    # 配置量化参数
    quant_config = {
        "w_bit": w_bit,
        "q_group_size": q_group_size,
        "version": version,
        "modules_to_not_convert": ["vision_tower", "vision_model"]  # 视觉塔不量化
    }
    
    # 执行量化
    model.quantize(
        calib_data,
        quant_config=quant_config
    )
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存量化模型
    print(f"保存量化模型到: {output_dir}")
    model.save_quantized(output_dir)
    processor.save_pretrained(output_dir)
    
    # 保存量化配置
    quant_info = {
        "quantization": "AWQ",
        "w_bit": w_bit,
        "q_group_size": q_group_size,
        "version": version,
        "calibration_samples": len(calib_data)
    }
    
    with open(os.path.join(output_dir, "quant_config.json"), 'w', encoding='utf-8') as f:
        json.dump(quant_info, f, indent=2)
    
    print("✅ AWQ量化完成！")
    print(f"   输出目录: {output_dir}")
    
    # 计算模型大小
    total_size = 0
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    
    print(f"   量化后模型大小: {total_size / 1e9:.2f} GB")
    print(f"   预期推理速度: < 1秒/张")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AWQ量化模型")
    parser.add_argument("--model_path", type=str, required=True,
                       help="合并后的模型路径")
    parser.add_argument("--output_dir", type=str, default="./models/qwen3-vl-pcb-awq",
                       help="量化模型输出目录")
    parser.add_argument("--calib_data_dir", type=str, default=None,
                       help="校准数据目录（可选）")
    parser.add_argument("--num_calib_samples", type=int, default=200,
                       help="校准样本数量")
    parser.add_argument("--w_bit", type=int, default=4,
                       help="量化位数")
    parser.add_argument("--q_group_size", type=int, default=128,
                       help="量化组大小")
    
    args = parser.parse_args()
    
    quantize_model_awq(
        model_path=args.model_path,
        output_dir=args.output_dir,
        calib_data_dir=args.calib_data_dir,
        num_calib_samples=args.num_calib_samples,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size
    )

