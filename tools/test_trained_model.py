#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练好的模型权重测试脚本
用于测试训练完成后的模型权重文件
"""
import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.pcb_agent import SimplePCBAgent
from PIL import Image
import json


def test_single_image(model_path: str, image_path: str, inspection_type: str = "full"):
    """
    测试单张图像
    
    Args:
        model_path: 训练好的模型路径（可以是 LoRA checkpoint、合并模型或量化模型）
        image_path: 测试图像路径
        inspection_type: 检测类型（full/short/open/missing）
    """
    print("=" * 60)
    print("测试训练好的模型权重")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"测试图像: {image_path}")
    print(f"检测类型: {inspection_type}")
    print("=" * 60)
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型路径不存在: {model_path}")
        print("\n提示:")
        print("  - LoRA权重路径: checkpoints/pcb_checkpoints/final/")
        print("  - 合并模型路径: models/qwen3-vl-pcb/")
        print("  - 量化模型路径: models/qwen3-vl-pcb-awq/ (推荐用于推理)")
        return
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误: 图像文件不存在: {image_path}")
        return
    
    # 创建智能体（自动加载模型）
    print("\n加载模型...")
    try:
        agent = SimplePCBAgent(model_path=model_path)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("\n可能的原因:")
        print("  1. 模型路径不正确")
        print("  2. 模型文件不完整（缺少配置文件或权重文件）")
        print("  3. 如果使用LoRA权重，需要先合并到基础模型")
        return
    
    # 执行检测
    print("\n执行缺陷检测...")
    try:
        defects = agent.inspect(
            image_path=image_path,
            inspection_type=inspection_type
        )
        
        print(f"\n✅ 检测完成！")
        print(f"检测到 {len(defects)} 个缺陷:")
        print("-" * 60)
        
        if defects:
            for i, defect in enumerate(defects, 1):
                defect_type = defect.get("defect", defect.get("type", "unknown"))
                bbox = defect.get("bbox", [])
                confidence = defect.get("confidence", defect.get("score", 0))
                repair = defect.get("repair", "")
                
                print(f"\n缺陷 {i}:")
                print(f"  类型: {defect_type}")
                print(f"  边界框: {bbox}")
                print(f"  置信度: {confidence:.2%}" if confidence else "  置信度: N/A")
                print(f"  维修建议: {repair}")
        else:
            print("  未检测到缺陷（图像正常）")
        
        print("-" * 60)
        
        # 保存结果到JSON文件
        output_file = f"test_result_{Path(image_path).stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(defects, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 检测过程出错: {e}")
        import traceback
        traceback.print_exc()


def test_batch_images(model_path: str, image_dir: str, inspection_type: str = "full"):
    """
    批量测试多张图像
    
    Args:
        model_path: 训练好的模型路径
        image_dir: 测试图像目录
        inspection_type: 检测类型
    """
    print("=" * 60)
    print("批量测试训练好的模型权重")
    print("=" * 60)
    
    if not os.path.exists(image_dir):
        print(f"❌ 错误: 图像目录不存在: {image_dir}")
        return
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        f for f in os.listdir(image_dir)
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]
    
    if not image_files:
        print(f"❌ 错误: 在目录 {image_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图像")
    print(f"模型路径: {model_path}")
    print("=" * 60)
    
    # 加载模型（只加载一次）
    print("\n加载模型...")
    try:
        agent = SimplePCBAgent(model_path=model_path)
        print("✅ 模型加载成功\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 批量处理
    results = []
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)
        print(f"[{i}/{len(image_files)}] 处理: {image_file}")
        
        try:
            defects = agent.inspect(
                image_path=image_path,
                inspection_type=inspection_type
            )
            defect_count = len(defects)
            print(f"  ✅ 检测到 {defect_count} 个缺陷")
            
            results.append({
                "image": image_file,
                "defects_count": defect_count,
                "defects": defects
            })
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            results.append({
                "image": image_file,
                "error": str(e)
            })
    
    # 保存批量结果
    output_file = "batch_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("批量测试完成")
    print("=" * 60)
    
    total_images = len(results)
    success_count = sum(1 for r in results if "error" not in r)
    total_defects = sum(r.get("defects_count", 0) for r in results if "defects_count" in r)
    
    print(f"总图像数: {total_images}")
    print(f"成功处理: {success_count}")
    print(f"总缺陷数: {total_defects}")
    print(f"结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="测试训练好的模型权重文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 测试单张图像
  python tools/test_trained_model.py --model_path ./models/qwen3-vl-pcb-awq --image_path test.jpg
  
  # 测试单张图像（指定检测类型）
  python tools/test_trained_model.py --model_path ./models/qwen3-vl-pcb-awq --image_path test.jpg --type short
  
  # 批量测试
  python tools/test_trained_model.py --model_path ./models/qwen3-vl-pcb-awq --image_dir ./data/test_images/

支持的模型路径:
  - LoRA权重: checkpoints/pcb_checkpoints/final/
  - 合并模型: models/qwen3-vl-pcb/
  - 量化模型: models/qwen3-vl-pcb-awq/ (推荐用于推理)
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练好的模型路径（LoRA checkpoint、合并模型或量化模型）"
    )
    
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="单张测试图像路径"
    )
    
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="测试图像目录（批量测试）"
    )
    
    parser.add_argument(
        "--type",
        type=str,
        default="full",
        choices=["full", "short", "open", "missing"],
        help="检测类型（默认: full）"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not args.image_path and not args.image_dir:
        parser.error("必须提供 --image_path 或 --image_dir 参数")
    
    if args.image_path and args.image_dir:
        parser.error("不能同时使用 --image_path 和 --image_dir")
    
    # 执行测试
    if args.image_path:
        test_single_image(args.model_path, args.image_path, args.type)
    else:
        test_batch_images(args.model_path, args.image_dir, args.type)


if __name__ == "__main__":
    main()

