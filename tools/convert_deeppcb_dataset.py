#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepPCB数据集格式转换工具
将DeepPCB数据集格式转换为项目所需的格式

DeepPCB数据集格式：
- 图像对：xxx_test.jpg（测试图像）和xxx_temp.jpg（模板图像）
- 标注文件：xxx.txt
- 标注格式：x1,y1,x2,y2,type （x1,y1为左上角，x2,y2为右下角）
- type: 0-背景, 1-open, 2-short, 3-mousebite, 4-spur, 5-copper, 6-pin-hole

项目所需格式：
- labels.json
- 格式：[{"image": "xxx.jpg", "defects": [{"type": "short", "bbox": [x,y,w,h], "repair": "..."}]}]
- bbox: [x, y, width, height] （左上角坐标+宽高）

使用方法：
    python convert_deeppcb_dataset.py --deeppcb_dir /path/to/DeepPCB-master
"""
import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import random


# DeepPCB缺陷类型映射
DEEPPCB_TYPE_MAP = {
    0: None,           # 背景，不使用
    1: "open",         # 断路
    2: "short",        # 短路
    3: "mousebite",    # 鼠咬（映射为open）
    4: "spur",         # 毛刺（映射为short）
    5: "copper",       # 多余铜（映射为missing）
    6: "pin-hole",     # 针孔（映射为missing）
}

# DeepPCB类型名称映射（用于生成维修建议）
DEEPPCB_TYPE_NAMES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin-hole",
}

# 维修建议
REPAIR_SUGGESTIONS = {
    "open": "补焊连接，检查线路完整性",
    "short": "清理焊锡桥接，检查相邻焊盘",
    "missing": "检查元件缺失，补装缺失元件",
    "mousebite": "修复线路断口，补焊连接",
    "spur": "清理多余焊锡，去除毛刺",
    "copper": "去除多余铜箔",
    "pin-hole": "检查并修复针孔缺陷",
}


def convert_bbox_xyxy_to_xywh(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
    """
    将bbox从(x1,y1,x2,y2)格式转换为(x,y,w,h)格式
    
    Args:
        x1, y1: 左上角坐标
        x2, y2: 右下角坐标
    
    Returns:
        (x, y, width, height)
    """
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return (x, y, w, h)


def parse_deeppcb_annotation(annotation_path: str, image_width: int = 640, image_height: int = 640) -> List[Dict[str, Any]]:
    """
    解析DeepPCB标注文件
    
    Args:
        annotation_path: 标注文件路径（.txt文件）
        image_width: 图像宽度（用于验证bbox范围，默认640）
        image_height: 图像高度（用于验证bbox范围，默认640）
    
    Returns:
        缺陷列表，每个缺陷包含type和bbox
    """
    defects = []
    
    if not os.path.exists(annotation_path):
        return defects
    
    # 尝试多种编码
    encodings = ['utf-8', 'gbk', 'latin-1']
    content = None
    
    for encoding in encodings:
        try:
            with open(annotation_path, 'r', encoding=encoding) as f:
                content = f.read()
                break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        print(f"⚠️  警告: 无法读取标注文件 {annotation_path}，跳过")
        return defects
    
    for line_num, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        
        try:
            # 格式：x1 y1 x2 y2 type 或 x1,y1,x2,y2,type（支持两种格式）
            # 先尝试按逗号分割，如果没有逗号则按空格分割
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
            else:
                parts = line.split()  # 按空格分割
            
            if len(parts) < 5:
                continue
            
            x1 = int(parts[0])
            y1 = int(parts[1])
            x2 = int(parts[2])
            y2 = int(parts[3])
            type_id = int(parts[4])
            
            # 验证type_id范围
            if type_id not in DEEPPCB_TYPE_MAP:
                continue
            
            # 转换为项目格式的type
            defect_type = DEEPPCB_TYPE_MAP.get(type_id)
            if defect_type is None:
                continue  # 跳过背景（type_id=0）
            
            # 转换bbox格式
            x, y, w, h = convert_bbox_xyxy_to_xywh(x1, y1, x2, y2)
            
            # 验证bbox是否在图像范围内
            if x < 0 or y < 0 or x + w > image_width or y + h > image_height:
                # 裁剪到图像范围内
                x = max(0, min(x, image_width - 1))
                y = max(0, min(y, image_height - 1))
                w = min(w, image_width - x)
                h = min(h, image_height - y)
                if w <= 0 or h <= 0:
                    continue  # 无效的bbox，跳过
            
            # 获取维修建议
            original_type_name = DEEPPCB_TYPE_NAMES.get(type_id, defect_type)
            repair = REPAIR_SUGGESTIONS.get(original_type_name, REPAIR_SUGGESTIONS.get(defect_type, "检查并修复缺陷"))
            
            defects.append({
                "type": defect_type,
                "bbox": [x, y, w, h],
                "repair": repair
            })
        except (ValueError, IndexError) as e:
            print(f"⚠️  警告: 解析标注文件 {annotation_path} 第 {line_num} 行失败: {line} ({e})")
            continue
    
    return defects


def find_pcbdata_dir(deeppcb_dir: str) -> Optional[str]:
    """
    查找PCBData目录
    
    Args:
        deeppcb_dir: DeepPCB数据集根目录
    
    Returns:
        PCBData目录路径，如果找不到返回None
    """
    deeppcb_path = Path(deeppcb_dir)
    
    if not deeppcb_path.exists():
        return None
    
    # 优先检查: PCBData子目录（递归查找 *_test.jpg 文件）
    pcbdata_path = deeppcb_path / "PCBData"
    if pcbdata_path.exists() and pcbdata_path.is_dir():
        if any(pcbdata_path.rglob("*_test.jpg")):
            return str(pcbdata_path)
    
    # 尝试1: 直接是PCBData目录（递归查找 *_test.jpg 文件）
    if any(deeppcb_path.rglob("*_test.jpg")):
        return str(deeppcb_path)
    
    # 尝试2: 递归查找所有子目录（DeepPCB数据集可能分组存储）
    for subdir in deeppcb_path.iterdir():
        if subdir.is_dir():
            # 递归检查子目录（支持任意深度的嵌套）
            if any(subdir.rglob("*_test.jpg")):
                return str(subdir)
    
    return None


def convert_deeppcb_dataset(
    deeppcb_dir: str,
    output_dir: str = "./data/pcb_defects",
    split_ratio: float = 0.8,
    shuffle: bool = True,
    seed: int = 42
):
    """
    转换DeepPCB数据集为项目格式
    
    Args:
        deeppcb_dir: DeepPCB数据集目录（可以是DeepPCB-master根目录或PCBData目录）
        output_dir: 输出目录
        split_ratio: 训练集比例（默认0.8，即80%训练，20%测试）
        shuffle: 是否随机打乱数据集（默认True）
        seed: 随机种子（默认42）
    """
    print("=" * 60)
    print("🔄 DeepPCB数据集格式转换工具")
    print("=" * 60)
    
    # 查找PCBData目录
    pcbdata_dir = find_pcbdata_dir(deeppcb_dir)
    
    if pcbdata_dir is None:
        raise ValueError(
            f"未找到DeepPCB数据集目录。请检查路径: {deeppcb_dir}\n"
            f"提示: 确保目录中包含 *_test.jpg 文件，或者包含 PCBData/ 子目录"
        )
    
    print(f"📁 找到数据集目录: {pcbdata_dir}")
    
    # 创建输出目录
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 输出目录: {output_dir}")
    
    # 查找所有测试图像文件（递归查找，支持分组目录结构）
    pcbdata_path = Path(pcbdata_dir)
    test_images_full_paths = list(pcbdata_path.rglob("*_test.jpg"))
    
    if len(test_images_full_paths) == 0:
        raise ValueError(f"未找到任何 *_test.jpg 图像文件，请检查目录: {pcbdata_dir}")
    
    # 转换为相对于pcbdata_dir的路径，用于后续处理
    test_images_info = []
    for img_path in test_images_full_paths:
        # 获取相对于pcbdata_dir的路径
        rel_path = img_path.relative_to(pcbdata_path)
        test_images_info.append((str(rel_path), img_path))
    
    test_images_info.sort(key=lambda x: x[0])  # 按路径排序
    print(f"📊 找到 {len(test_images_info)} 张测试图像")
    
    # 转换数据
    converted_data = []
    image_mapping = {}  # 记录原始文件名到新文件名的映射
    skipped_count = 0
    error_count = 0
    
    for idx, (test_img_rel_path, test_img_full_path) in enumerate(test_images_info):
        # 获取标注文件路径（DeepPCB数据集结构中，标注文件在 *_not 子目录中）
        # 例如：group12000/12000/12000001_test.jpg -> group12000/12000_not/12000001.txt
        base_name = Path(test_img_rel_path).stem.replace("_test", "")
        
        # 首先尝试在 *_not 目录中查找（DeepPCB标准结构）
        parent_dir_name = test_img_full_path.parent.name
        not_dir = test_img_full_path.parent.parent / f"{parent_dir_name}_not"
        annotation_path = not_dir / f"{base_name}.txt"
        
        # 如果 *_not 目录不存在，尝试在同一目录查找（兼容其他结构）
        if not annotation_path.exists():
            annotation_path = test_img_full_path.parent / f"{base_name}.txt"
        
        # 检查图像文件是否存在
        if not test_img_full_path.exists():
            print(f"⚠️  跳过: 图像文件不存在 {test_img_full_path}")
            skipped_count += 1
            continue
        
        # 检查标注文件是否存在
        if not annotation_path.exists():
            print(f"⚠️  跳过: 标注文件不存在 {annotation_path}")
            skipped_count += 1
            continue
        
        # 读取图像以获取尺寸（用于验证bbox）
        try:
            img = Image.open(test_img_full_path)
            img_width, img_height = img.size
            img = img.convert('RGB')  # 确保RGB格式
        except Exception as e:
            print(f"⚠️  读取图像失败 {test_img_full_path}: {e}")
            error_count += 1
            continue
        
        # 解析标注
        defects = parse_deeppcb_annotation(str(annotation_path), img_width, img_height)
        
        # 生成新文件名（使用索引避免文件名冲突）
        new_img_name = f"deeppcb_{idx:06d}.jpg"
        new_img_path = Path(output_images_dir) / new_img_name
        
        # 保存图像文件
        try:
            img.save(str(new_img_path), 'JPEG', quality=95)
        except Exception as e:
            print(f"⚠️  保存图像失败 {new_img_path}: {e}")
            error_count += 1
            continue
        
        # 添加到转换后的数据
        converted_data.append({
            "image": new_img_name,
            "defects": defects
        })
        
        image_mapping[test_img_rel_path] = new_img_name
        
        if (idx + 1) % 100 == 0:
            print(f"✅ 已转换 {idx + 1}/{len(test_images_info)} 张图像...")
    
    print(f"✅ 转换完成！共转换 {len(converted_data)} 张图像")
    if skipped_count > 0:
        print(f"⚠️  跳过了 {skipped_count} 个文件（文件不存在）")
    if error_count > 0:
        print(f"❌ 处理失败 {error_count} 个文件")
    
    print(f"✅ 转换完成！共转换 {len(converted_data)} 张图像")
    
    # 统计缺陷类型
    defect_stats = {}
    for item in converted_data:
        for defect in item["defects"]:
            defect_type = defect["type"]
            defect_stats[defect_type] = defect_stats.get(defect_type, 0) + 1
    
    print("\n📊 缺陷统计:")
    for defect_type, count in sorted(defect_stats.items()):
        print(f"  {defect_type}: {count} 个")
    
    # 分割训练集和测试集
    if shuffle:
        random.seed(seed)
        random.shuffle(converted_data)
        print(f"🔀 数据集已随机打乱（seed={seed}）")
    
    split_idx = int(len(converted_data) * split_ratio)
    train_data = converted_data[:split_idx]
    test_data = converted_data[split_idx:]
    
    print(f"\n📊 数据集分割:")
    print(f"  训练集: {len(train_data)} 张")
    print(f"  测试集: {len(test_data)} 张")
    
    # 保存labels.json（训练集）
    train_labels_path = os.path.join(output_dir, "labels.json")
    with open(train_labels_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 训练集标签已保存: {train_labels_path}")
    
    # 保存测试集标签（可选）
    test_labels_path = os.path.join(output_dir, "labels_test.json")
    with open(test_labels_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 测试集标签已保存: {test_labels_path}")
    
    # 保存图像映射（可选，用于参考）
    mapping_path = os.path.join(output_dir, "image_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(image_mapping, f, ensure_ascii=False, indent=2)
    print(f"✅ 图像映射已保存: {mapping_path}")
    
    print("\n" + "=" * 60)
    print("✨ 转换完成！")
    print("=" * 60)
    print(f"\n📁 输出目录结构:")
    print(f"  {output_dir}/")
    print(f"    images/          # {len(converted_data)} 张图像")
    print(f"    labels.json      # 训练集标签（{len(train_data)} 个样本）")
    print(f"    labels_test.json # 测试集标签（{len(test_data)} 个样本）")
    print(f"    image_mapping.json # 原始文件名映射")
    print("\n💡 接下来可以:")
    print(f"  1. 检查数据集: python quick_start.py")
    print(f"  2. 开始训练: python pcb_train.py --data_dir {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepPCB数据集格式转换工具")
    parser.add_argument(
        "--deeppcb_dir",
        type=str,
        required=True,
        help="DeepPCB数据集目录路径（包含PCBData文件夹）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/pcb_defects",
        help="输出目录（默认: ./data/pcb_defects）"
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="训练集比例（默认: 0.8，即80%%训练，20%%测试）"
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="不打乱数据集（默认会随机打乱）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）"
    )
    
    args = parser.parse_args()
    
    try:
        convert_deeppcb_dataset(
            deeppcb_dir=args.deeppcb_dir,
            output_dir=args.output_dir,
            split_ratio=args.split_ratio,
            shuffle=not args.no_shuffle,
            seed=args.seed
        )
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

