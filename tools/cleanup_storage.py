#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""自动清理存储空间，删除不需要的文件"""

import os
import shutil
import argparse
from pathlib import Path

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def get_dir_size(path):
    """计算目录大小"""
    total = 0
    if not os.path.exists(path):
        return 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total

def cleanup_base_model(models_dir: str = "./models", dry_run: bool = False):
    """清理基础模型（训练完成后可以删除）"""
    print("\n[1] 清理基础模型")
    print("-" * 60)
    
    base_model_patterns = [
        "Qwen3-VL-32B-Instruct",  # 基础模型目录
        "qwen3-vl-32b-instruct",  # 小写版本
    ]
    
    total_freed = 0
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f"   模型目录不存在: {models_dir}")
        return total_freed
    
    for pattern in base_model_patterns:
        model_path = models_path / pattern
        if model_path.exists() and model_path.is_dir():
            size = get_dir_size(model_path)
            total_freed += size
            print(f"   找到基础模型: {model_path}")
            print(f"   大小: {format_size(size)}")
            
            if not dry_run:
                try:
                    shutil.rmtree(model_path)
                    print(f"   ✓ 已删除: {model_path}")
                except Exception as e:
                    print(f"   ✗ 删除失败: {e}")
            else:
                print(f"   [模拟] 将删除: {model_path}")
        else:
            # 检查是否在子目录中
            for subdir in models_path.iterdir():
                if subdir.is_dir() and pattern.lower() in subdir.name.lower():
                    size = get_dir_size(subdir)
                    total_freed += size
                    print(f"   找到基础模型: {subdir}")
                    print(f"   大小: {format_size(size)}")
                    
                    if not dry_run:
                        try:
                            shutil.rmtree(subdir)
                            print(f"   ✓ 已删除: {subdir}")
                        except Exception as e:
                            print(f"   ✗ 删除失败: {e}")
                    else:
                        print(f"   [模拟] 将删除: {subdir}")
    
    if total_freed == 0:
        print("   未找到基础模型文件")
    
    return total_freed

def cleanup_merged_model(models_dir: str = "./models", dry_run: bool = False):
    """清理合并后的模型（如果已有量化模型，可以删除）"""
    print("\n[2] 清理合并后的模型")
    print("-" * 60)
    
    merged_patterns = [
        "qwen3-vl-pcb-merged",
        "qwen3-vl-pcb-*merged*",
    ]
    
    total_freed = 0
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f"   模型目录不存在: {models_dir}")
        return total_freed
    
    for item in models_path.iterdir():
        if item.is_dir() and "merged" in item.name.lower():
            # 检查是否有对应的量化模型
            quantized_name = item.name.replace("merged", "awq")
            quantized_path = models_path / quantized_name
            
            if quantized_path.exists():
                size = get_dir_size(item)
                total_freed += size
                print(f"   找到合并模型: {item}")
                print(f"   大小: {format_size(size)}")
                print(f"   检测到量化模型: {quantized_path} (保留量化模型)")
                
                if not dry_run:
                    try:
                        shutil.rmtree(item)
                        print(f"   ✓ 已删除: {item}")
                    except Exception as e:
                        print(f"   ✗ 删除失败: {e}")
                else:
                    print(f"   [模拟] 将删除: {item}")
            else:
                print(f"   找到合并模型: {item}")
                print(f"   警告: 未找到对应的量化模型，跳过删除（安全起见）")
    
    if total_freed == 0:
        print("   未找到可删除的合并模型（或未找到对应的量化模型）")
    
    return total_freed

def cleanup_old_checkpoints(checkpoints_dir: str = "./checkpoints", keep: int = 2, dry_run: bool = False):
    """清理旧的检查点，只保留最新的N个"""
    print(f"\n[3] 清理旧检查点（保留最新 {keep} 个）")
    print("-" * 60)
    
    total_freed = 0
    checkpoints_path = Path(checkpoints_dir)
    
    if not checkpoints_path.exists():
        print(f"   检查点目录不存在: {checkpoints_dir}")
        return total_freed
    
    # 获取所有检查点目录
    checkpoint_dirs = []
    for item in checkpoints_path.iterdir():
        if item.is_dir():
            # 按修改时间排序
            mtime = item.stat().st_mtime
            checkpoint_dirs.append((mtime, item))
    
    if len(checkpoint_dirs) <= keep:
        print(f"   检查点数量 ({len(checkpoint_dirs)}) <= 保留数量 ({keep})，无需清理")
        return total_freed
    
    # 按时间排序，最新的在前
    checkpoint_dirs.sort(reverse=True)
    
    # 保留最新的N个，删除其他的
    to_delete = checkpoint_dirs[keep:]
    to_keep = checkpoint_dirs[:keep]
    
    print(f"   总检查点数: {len(checkpoint_dirs)}")
    print(f"   保留: {len(to_keep)} 个")
    print(f"   删除: {len(to_delete)} 个")
    
    for mtime, checkpoint_dir in to_keep:
        print(f"   保留: {checkpoint_dir.name} ({format_size(get_dir_size(checkpoint_dir))})")
    
    for mtime, checkpoint_dir in to_delete:
        size = get_dir_size(checkpoint_dir)
        total_freed += size
        print(f"   删除: {checkpoint_dir.name} ({format_size(size)})")
        
        if not dry_run:
            try:
                shutil.rmtree(checkpoint_dir)
                print(f"   ✓ 已删除: {checkpoint_dir}")
            except Exception as e:
                print(f"   ✗ 删除失败: {e}")
        else:
            print(f"   [模拟] 将删除: {checkpoint_dir}")
    
    return total_freed

def cleanup_original_dataset(deeppcb_dir: str = "./DeepPCB-master", dry_run: bool = False):
    """清理原始DeepPCB数据集（转换完成后可以删除）"""
    print("\n[4] 清理原始DeepPCB数据集")
    print("-" * 60)
    
    deeppcb_path = Path(deeppcb_dir)
    
    if not deeppcb_path.exists():
        print(f"   原始数据集目录不存在: {deeppcb_dir}")
        return 0
    
    # 检查是否已有转换后的数据集
    converted_data = Path("./tools/data/pcb_defects/labels.json")
    if not converted_data.exists():
        print(f"   警告: 未找到转换后的数据集 ({converted_data})")
        print(f"   建议: 先确认转换后的数据集存在，再删除原始数据")
        response = input("   是否继续删除原始数据集? (yes/no): ")
        if response.lower() != 'yes':
            print("   已取消删除")
            return 0
    
    size = get_dir_size(deeppcb_path)
    print(f"   找到原始数据集: {deeppcb_path}")
    print(f"   大小: {format_size(size)}")
    print(f"   转换后的数据集已存在: {converted_data}")
    
    if not dry_run:
        try:
            shutil.rmtree(deeppcb_path)
            print(f"   ✓ 已删除: {deeppcb_path}")
            return size
        except Exception as e:
            print(f"   ✗ 删除失败: {e}")
            return 0
    else:
        print(f"   [模拟] 将删除: {deeppcb_path}")
        return size

def cleanup_cache(cache_dirs: list = None, dry_run: bool = False):
    """清理缓存文件"""
    print("\n[5] 清理缓存文件")
    print("-" * 60)
    
    if cache_dirs is None:
        cache_dirs = [
            "__pycache__",
            ".cache",
            ".pytest_cache",
            "*.pyc",
        ]
    
    total_freed = 0
    project_root = Path(".")
    
    # 清理 __pycache__ 目录
    for pycache in project_root.rglob("__pycache__"):
        if pycache.is_dir():
            size = get_dir_size(pycache)
            total_freed += size
            print(f"   找到缓存目录: {pycache}")
            print(f"   大小: {format_size(size)}")
            
            if not dry_run:
                try:
                    shutil.rmtree(pycache)
                    print(f"   ✓ 已删除: {pycache}")
                except Exception as e:
                    print(f"   ✗ 删除失败: {e}")
            else:
                print(f"   [模拟] 将删除: {pycache}")
    
    # 清理 .pyc 文件
    for pyc_file in project_root.rglob("*.pyc"):
        if pyc_file.is_file():
            size = pyc_file.stat().st_size
            total_freed += size
            print(f"   找到缓存文件: {pyc_file}")
            
            if not dry_run:
                try:
                    pyc_file.unlink()
                    print(f"   ✓ 已删除: {pyc_file}")
                except Exception as e:
                    print(f"   ✗ 删除失败: {e}")
            else:
                print(f"   [模拟] 将删除: {pyc_file}")
    
    if total_freed == 0:
        print("   未找到缓存文件")
    
    return total_freed

def main():
    parser = argparse.ArgumentParser(description="清理项目存储空间")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行，不实际删除文件")
    parser.add_argument("--base-model", action="store_true", help="删除基础模型")
    parser.add_argument("--merged-model", action="store_true", help="删除合并后的模型")
    parser.add_argument("--checkpoints", type=int, default=0, help="清理旧检查点，保留N个（0=不清理）")
    parser.add_argument("--original-dataset", action="store_true", help="删除原始DeepPCB数据集")
    parser.add_argument("--cache", action="store_true", help="清理缓存文件")
    parser.add_argument("--all", action="store_true", help="执行所有清理操作（训练完成后推荐）")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("项目存储空间清理工具")
    print("=" * 70)
    
    if args.dry_run:
        print("\n[模式] 模拟运行（不会实际删除文件）")
    
    total_freed = 0
    
    # 如果使用 --all，执行所有清理
    if args.all:
        print("\n[执行所有清理操作]")
        total_freed += cleanup_base_model(dry_run=args.dry_run)
        total_freed += cleanup_merged_model(dry_run=args.dry_run)
        total_freed += cleanup_old_checkpoints(keep=2, dry_run=args.dry_run)
        total_freed += cleanup_original_dataset(dry_run=args.dry_run)
        total_freed += cleanup_cache(dry_run=args.dry_run)
    else:
        # 单独执行各项清理
        if args.base_model:
            total_freed += cleanup_base_model(dry_run=args.dry_run)
        
        if args.merged_model:
            total_freed += cleanup_merged_model(dry_run=args.dry_run)
        
        if args.checkpoints > 0:
            total_freed += cleanup_old_checkpoints(keep=args.checkpoints, dry_run=args.dry_run)
        
        if args.original_dataset:
            total_freed += cleanup_original_dataset(dry_run=args.dry_run)
        
        if args.cache:
            total_freed += cleanup_cache(dry_run=args.dry_run)
        
        if not any([args.base_model, args.merged_model, args.checkpoints > 0, 
                   args.original_dataset, args.cache]):
            print("\n未指定清理选项，使用 --help 查看帮助")
            print("\n推荐用法:")
            print("  # 模拟运行，查看将删除什么")
            print("  python tools/cleanup_storage.py --all --dry-run")
            print("\n  # 训练完成后，清理所有不需要的文件")
            print("  python tools/cleanup_storage.py --all")
            print("\n  # 只清理基础模型")
            print("  python tools/cleanup_storage.py --base-model")
            return
    
    # 总结
    print("\n" + "=" * 70)
    print("清理总结")
    print("=" * 70)
    print(f"释放空间: {format_size(total_freed)}")
    
    if args.dry_run:
        print("\n这是模拟运行，未实际删除文件")
        print("要实际执行清理，请去掉 --dry-run 参数")

if __name__ == "__main__":
    main()

