#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Autodl A800兼容性检查脚本
检查项目是否可以在Autodl A800 80GB服务器上运行
"""
import os
import sys
import subprocess
from pathlib import Path


def check_gpu():
    """检查GPU信息"""
    print("=" * 60)
    print("🔍 检查GPU信息...")
    print("=" * 60)
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA不可用")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        print(f"✅ GPU型号: {gpu_name}")
        print(f"✅ 显存大小: {gpu_memory:.1f} GB")
        
        # 检查是否满足要求（>=80GB）
        if gpu_memory >= 80:
            print(f"✅ GPU显存满足要求（>=80GB）")
            return True
        elif gpu_memory >= 40:
            print(f"⚠️  GPU显存 {gpu_memory:.1f}GB，可能足够（推荐80GB+）")
            return True
        else:
            print(f"❌ GPU显存不足: {gpu_memory:.1f}GB（需要>=80GB）")
            return False
            
    except ImportError:
        print("❌ PyTorch未安装")
        return False


def check_cuda_version():
    """检查CUDA版本"""
    print("\n" + "=" * 60)
    print("🔍 检查CUDA版本...")
    print("=" * 60)
    
    try:
        import torch
        cuda_version = torch.version.cuda
        if cuda_version:
            print(f"✅ CUDA版本: {cuda_version}")
            # 检查版本是否>=11.8
            major, minor = map(int, cuda_version.split('.')[:2])
            if major > 11 or (major == 11 and minor >= 8):
                print("✅ CUDA版本满足要求（>=11.8）")
                return True
            else:
                print(f"⚠️  CUDA版本 {cuda_version} 可能较旧（推荐11.8+）")
                return True  # 仍然可以运行
        else:
            print("⚠️  无法获取CUDA版本（可能使用CPU版本）")
            return False
    except Exception as e:
        print(f"⚠️  检查CUDA版本时出错: {e}")
        return False


def check_disk_space():
    """检查磁盘空间"""
    print("\n" + "=" * 60)
    print("🔍 检查磁盘空间...")
    print("=" * 60)
    
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)
        
        print(f"总空间: {total_gb:.1f} GB")
        print(f"已使用: {used_gb:.1f} GB")
        print(f"可用空间: {free_gb:.1f} GB")
        
        # 估算需要的空间
        required_space = 200  # GB
        model_space = 25  # 量化模型
        checkpoint_space = 50  # 训练检查点（多个）
        dataset_space = 5  # 数据集
        cache_space = 80  # HuggingFace缓存（基础模型）
        total_required = model_space + checkpoint_space + dataset_space + cache_space
        
        print(f"\n📊 空间需求估算:")
        print(f"  - 量化模型: ~{model_space}GB")
        print(f"  - 训练检查点: ~{checkpoint_space}GB")
        print(f"  - 数据集: ~{dataset_space}GB")
        print(f"  - HuggingFace缓存: ~{cache_space}GB")
        print(f"  - 总计需求: ~{total_required}GB")
        
        if free_gb >= total_required:
            print(f"✅ 可用空间充足（需要{total_required}GB，可用{free_gb:.1f}GB）")
            return True
        elif free_gb >= required_space:
            print(f"⚠️  可用空间可能不足（需要{total_required}GB，可用{free_gb:.1f}GB）")
            print("   建议清理空间或使用外部存储")
            return True
        else:
            print(f"❌ 可用空间不足（需要{total_required}GB，可用{free_gb:.1f}GB）")
            return False
            
    except Exception as e:
        print(f"⚠️  检查磁盘空间时出错: {e}")
        return True  # 假设可以


def check_python_version():
    """检查Python版本"""
    print("\n" + "=" * 60)
    print("🔍 检查Python版本...")
    print("=" * 60)
    
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 8):
        print("✅ Python版本满足要求（>=3.8）")
        return True
    else:
        print("❌ Python版本过低（需要>=3.8）")
        return False


def check_dependencies():
    """检查关键依赖"""
    print("\n" + "=" * 60)
    print("🔍 检查关键依赖...")
    print("=" * 60)
    
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT (LoRA)',
        'autoawq': 'AutoAWQ (量化)',
        'accelerate': 'Accelerate',
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: 未安装")
            all_ok = False
    
    return all_ok


def check_model_config():
    """检查模型配置兼容性"""
    print("\n" + "=" * 60)
    print("🔍 检查模型配置...")
    print("=" * 60)
    
    try:
        try:
            import yaml
        except ImportError:
            try:
                import ruamel.yaml as yaml
            except ImportError:
                print("⚠️  无法导入yaml库，跳过配置检查")
                return True
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查4-bit配置（对A800很重要）
        use_4bit = config.get('model', {}).get('use_4bit', True)
        if use_4bit:
            print("✅ 已启用4-bit量化（节省显存，适合A800）")
        else:
            print("⚠️  未启用4-bit量化（可能需要更多显存）")
        
        # 检查batch size
        batch_size = config.get('training', {}).get('batch_size', 1)
        grad_accum = config.get('training', {}).get('gradient_accumulation_steps', 16)
        effective_batch = batch_size * grad_accum
        print(f"✅ 批次大小: {batch_size} × {grad_accum} = {effective_batch}（有效批次）")
        
        # 检查device_map
        device_map = config.get('model', {}).get('device_map', 'auto')
        print(f"✅ 设备映射: {device_map}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  检查配置时出错: {e}")
        return True


def estimate_memory_usage():
    """估算内存使用"""
    print("\n" + "=" * 60)
    print("📊 内存使用估算...")
    print("=" * 60)
    
    print("训练阶段（使用4-bit量化）:")
    print("  - 模型（4-bit）: ~20-25GB 显存")
    print("  - 优化器状态: ~5-10GB 显存")
    print("  - 激活值: ~5-10GB 显存")
    print("  - 总计: ~30-45GB 显存（A800 80GB充足）")
    
    print("\n推理阶段（量化模型）:")
    print("  - 量化模型: ~20-25GB 显存")
    print("  - 激活值: ~2-5GB 显存")
    print("  - 总计: ~25-30GB 显存（A800 80GB充足）")
    
    print("\n系统内存（RAM）:")
    print("  - 数据加载: ~5-10GB")
    print("  - Python进程: ~2-5GB")
    print("  - 总计: ~10-20GB（100GB RAM充足）")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🚀 Autodl A800 兼容性检查")
    print("=" * 60)
    print("\n服务器配置:")
    print("  - GPU: A800 80GB")
    print("  - 内存: 100GB")
    print("  - 存储: 200GB")
    print("=" * 60 + "\n")
    
    checks = []
    
    # 执行各项检查
    checks.append(("GPU检查", check_gpu()))
    checks.append(("CUDA版本", check_cuda_version()))
    checks.append(("Python版本", check_python_version()))
    checks.append(("磁盘空间", check_disk_space()))
    checks.append(("依赖检查", check_dependencies()))
    checks.append(("配置检查", check_model_config()))
    
    # 内存估算
    estimate_memory_usage()
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 检查总结")
    print("=" * 60)
    
    all_passed = True
    for name, result in checks:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 兼容性检查通过！项目可以在Autodl A800上运行")
        print("\n💡 建议:")
        print("  1. 确保使用4-bit量化（config.yaml中已配置）")
        print("  2. 训练时监控显存使用: watch -n 1 nvidia-smi")
        print("  3. 如果显存不足，可以减小batch_size或gradient_accumulation_steps")
        print("  4. 注意清理HuggingFace缓存以节省空间")
    else:
        print("❌ 存在兼容性问题，请根据上述检查结果解决")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

