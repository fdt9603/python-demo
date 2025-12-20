#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速启动脚本 - 帮助用户快速配置和运行项目
"""
import os
import sys
import json
from pathlib import Path


def check_environment():
    """检查环境配置"""
    print("=" * 60)
    print("🔍 检查环境配置...")
    print("=" * 60)
    
    checks = []
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        checks.append(True)
    else:
        print(f"❌ Python版本过低: {python_version.major}.{python_version.minor} (需要3.8+)")
        checks.append(False)
    
    # 检查核心依赖
    try:
        import torch
        print(f"✅ PyTorch已安装: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   GPU可用: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️  GPU不可用（训练需要GPU）")
        checks.append(True)
    except ImportError:
        print("❌ PyTorch未安装，请运行: pip install torch")
        checks.append(False)
    
    try:
        import transformers
        print(f"✅ Transformers已安装: {transformers.__version__}")
        checks.append(True)
    except ImportError:
        print("❌ Transformers未安装，请运行: pip install transformers")
        checks.append(False)
    
    # 检查可选依赖
    optional_checks = []
    try:
        import chromadb
        print("✅ ChromaDB已安装（向量数据库支持）")
        optional_checks.append(True)
    except ImportError:
        print("⚠️  ChromaDB未安装（可选，用于向量数据库）")
        optional_checks.append(False)
    
    try:
        import langgraph
        print("✅ LangGraph已安装（工作流支持）")
        optional_checks.append(True)
    except ImportError:
        print("⚠️  LangGraph未安装（可选，用于工作流）")
        optional_checks.append(False)
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ 核心环境配置正常")
    else:
        print("❌ 部分依赖缺失，请先安装: pip install -r requirements.txt")
    print("=" * 60 + "\n")
    
    return all(checks)


def check_dataset():
    """检查数据集配置"""
    print("=" * 60)
    print("📊 检查数据集配置...")
    print("=" * 60)
    
    data_dir = Path("data/pcb_defects")
    images_dir = data_dir / "images"
    labels_file = data_dir / "labels.json"
    
    # 检查目录结构
    if not data_dir.exists():
        print(f"❌ 数据集目录不存在: {data_dir}")
        print("   请创建目录: mkdir -p data/pcb_defects/images")
        return False
    
    if not images_dir.exists():
        print(f"❌ 图像目录不存在: {images_dir}")
        print("   请创建目录: mkdir -p data/pcb_defects/images")
        return False
    
    # 检查图像文件
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if len(image_files) == 0:
        print(f"⚠️  图像目录为空: {images_dir}")
        print("   请添加电路板图像文件（.jpg或.png）")
        return False
    else:
        print(f"✅ 找到 {len(image_files)} 张图像")
    
    # 检查标签文件
    if not labels_file.exists():
        print(f"⚠️  标签文件不存在: {labels_file}")
        print("   正在创建示例标签文件...")
        create_sample_labels(labels_file, images_dir, len(image_files))
        print(f"✅ 已创建示例标签文件: {labels_file}")
        print("   ⚠️  请根据实际情况修改标签文件中的缺陷标注")
    else:
        print(f"✅ 标签文件存在: {labels_file}")
        # 验证JSON格式
        try:
            with open(labels_file, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            print(f"   包含 {len(labels)} 个样本")
        except Exception as e:
            print(f"❌ 标签文件格式错误: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ 数据集配置检查完成")
    print("=" * 60 + "\n")
    return True


def create_sample_labels(labels_file, images_dir, num_samples):
    """创建示例标签文件"""
    from data_loader import create_sample_labels_json
    
    create_sample_labels_json(
        str(labels_file),
        str(images_dir),
        num_samples=min(num_samples, 10)  # 最多10个示例
    )


def check_model():
    """检查模型文件"""
    print("=" * 60)
    print("🤖 检查模型文件...")
    print("=" * 60)
    
    model_paths = [
        "models/qwen3-vl-pcb-awq",
        "models/qwen3-vl-pcb",
        "checkpoints/pcb_checkpoints"
    ]
    
    found_models = []
    for path in model_paths:
        if Path(path).exists():
            print(f"✅ 找到模型: {path}")
            found_models.append(path)
        else:
            print(f"⚠️  模型不存在: {path}")
    
    if not found_models:
        print("\n⚠️  未找到训练好的模型")
        print("   你需要先训练模型:")
        print("   python pcb_train.py --data_dir ./data/pcb_defects")
        print("\n   或者使用基础模型（效果较差）")
    else:
        print(f"\n✅ 找到 {len(found_models)} 个模型/检查点")
    
    print("=" * 60 + "\n")
    return len(found_models) > 0


def show_next_steps():
    """显示下一步操作"""
    print("=" * 60)
    print("📝 下一步操作建议")
    print("=" * 60)
    
    print("\n1️⃣  如果你有数据集但还没训练模型:")
    print("   python pcb_train.py --data_dir ./data/pcb_defects")
    
    print("\n2️⃣  如果你已有训练好的模型:")
    print("   python pcb_agent.py --image_path your_image.jpg --model_path ./models/qwen3-vl-pcb-awq")
    
    print("\n3️⃣  如果你想启动API服务:")
    print("   python mllm_api.py --port 8000")
    
    print("\n4️⃣  如果你想测试数据加载:")
    print("   python -c \"from data_loader import load_pcb_dataset; d=load_pcb_dataset('data/pcb_defects'); print(f'数据集大小: {len(d)}')\"")
    
    print("\n5️⃣  查看详细文档:")
    print("   - README.md - 项目总览")
    print("   - QUICKSTART.md - 快速开始")
    print("   - RUN_GUIDE.md - 运行指南")
    print("   - VECTOR_STORE_GUIDE.md - 向量数据库指南")
    
    print("\n" + "=" * 60)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🚀 PCB缺陷检测系统 - 快速启动检查")
    print("=" * 60 + "\n")
    
    # 检查环境
    env_ok = check_environment()
    
    # 检查数据集
    dataset_ok = check_dataset()
    
    # 检查模型
    model_ok = check_model()
    
    # 总结
    print("=" * 60)
    print("📋 检查总结")
    print("=" * 60)
    print(f"环境配置: {'✅' if env_ok else '❌'}")
    print(f"数据集配置: {'✅' if dataset_ok else '❌'}")
    print(f"模型文件: {'✅' if model_ok else '⚠️  (需要训练)'}")
    print("=" * 60 + "\n")
    
    # 显示下一步
    show_next_steps()
    
    # 给出建议
    if not env_ok:
        print("\n❌ 请先安装依赖: pip install -r requirements.txt")
    elif not dataset_ok:
        print("\n❌ 请先准备数据集（见上方说明）")
    elif not model_ok:
        print("\n⚠️  数据集已准备好，可以开始训练模型了！")
    else:
        print("\n✅ 一切就绪，可以开始使用了！")


if __name__ == "__main__":
    main()

