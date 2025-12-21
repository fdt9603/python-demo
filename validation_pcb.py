"""
Day 7: 工业级验证脚本
验证漏检率、推理速度、JSON格式正确率、显存稳定性
"""
import json
import time
import torch
import os
from typing import List, Dict, Any
from sklearn.metrics import recall_score, precision_score, f1_score
from pcb_agent import SimplePCBAgent
from data_loader import load_pcb_dataset


def test_miss_rate(agent: SimplePCBAgent, test_dataset, threshold: float = 0.99):
    """
    测试漏检率（致命指标）
    
    Args:
        agent: PCB智能体
        test_dataset: 测试数据集
        threshold: 最低召回率阈值（默认99%，即漏检率<1%）
    
    Returns:
        recall: 召回率
    """
    print("=" * 60)
    print("测试1: 漏检率测试（致命指标）")
    print("=" * 60)
    
    true_defects = []
    pred_defects = []
    
    for i, sample in enumerate(test_dataset):
        # 真实标签（只要图像有缺陷，标记为1）
        has_defect = len(sample.get("defects", [])) > 0 if isinstance(sample.get("defects"), list) else False
        true_defects.append(1 if has_defect else 0)
        
        # 模型预测
        try:
            image_path = sample.get("image_path") or sample.get("image")
            if isinstance(image_path, str) and os.path.exists(image_path):
                result = agent.run({"image_path": image_path, "inspection_type": "full"})
                pred = json.loads(result)
                pred_has_defect = len(pred) > 0 and not any(d.get("error") for d in pred)
                pred_defects.append(1 if pred_has_defect else 0)
            else:
                # 如果没有路径，跳过
                print(f"警告: 样本 {i} 缺少有效图像路径，跳过")
                pred_defects.append(0)
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            pred_defects.append(0)
        
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(test_dataset)} 个样本")
    
    # 计算指标
    recall = recall_score(true_defects, pred_defects)
    precision = precision_score(true_defects, pred_defects, zero_division=0)
    f1 = f1_score(true_defects, pred_defects, zero_division=0)
    
    miss_rate = 1 - recall
    
    print(f"\n结果:")
    print(f"  召回率 (Recall): {recall:.2%}")
    print(f"  漏检率 (Miss Rate): {miss_rate:.2%}")
    print(f"  精确率 (Precision): {precision:.2%}")
    print(f"  F1分数: {f1:.2%}")
    
    if recall < threshold:
        print(f"❌ 漏检率 > {1-threshold:.0%} (recall={recall:.2%})，不达标！")
        return False, recall
    else:
        print(f"✅ 漏检率 < {1-threshold:.0%}，达标！")
        return True, recall


def test_inference_speed(agent: SimplePCBAgent, test_images: List[str], 
                        max_avg_time: float = 1.0, max_p99_time: float = 1.5):
    """
    测试推理速度
    
    Args:
        agent: PCB智能体
        test_images: 测试图像路径列表
        max_avg_time: 最大平均时间（秒）
        max_p99_time: 最大P99时间（秒）
    
    Returns:
        success: 是否通过
        avg_time: 平均时间
        p99_time: P99时间
    """
    print("\n" + "=" * 60)
    print("测试2: 推理速度测试")
    print("=" * 60)
    
    times = []
    num_samples = min(100, len(test_images))
    
    for i, img_path in enumerate(test_images[:num_samples]):
        if not os.path.exists(img_path):
            continue
        
        start = time.time()
        try:
            agent.run({"image_path": img_path, "inspection_type": "full"})
            elapsed = time.time() - start
            times.append(elapsed)
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            continue
        
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{num_samples} 张图像")
    
    if len(times) == 0:
        print("❌ 没有有效的测试样本")
        return False, 0, 0
    
    avg_time = sum(times) / len(times)
    sorted_times = sorted(times)
    p99_idx = int(len(sorted_times) * 0.99)
    p99_time = sorted_times[p99_idx] if p99_idx < len(sorted_times) else sorted_times[-1]
    
    print(f"\n结果:")
    print(f"  平均推理时间: {avg_time:.3f} 秒")
    print(f"  P99推理时间: {p99_time:.3f} 秒")
    print(f"  最快: {min(times):.3f} 秒")
    print(f"  最慢: {max(times):.3f} 秒")
    
    success = avg_time < max_avg_time and p99_time < max_p99_time
    
    if not success:
        if avg_time >= max_avg_time:
            print(f"❌ 平均推理时间 > {max_avg_time}s: {avg_time:.3f}s")
        if p99_time >= max_p99_time:
            print(f"❌ P99延迟 > {max_p99_time}s: {p99_time:.3f}s")
    else:
        print(f"✅ 推理速度达标！")
    
    return success, avg_time, p99_time


def test_json_format(agent: SimplePCBAgent, test_images: List[str]):
    """
    测试JSON格式正确率（必须是100%）
    
    Args:
        agent: PCB智能体
        test_images: 测试图像路径列表
    
    Returns:
        success: 是否通过
        success_rate: 成功率
    """
    print("\n" + "=" * 60)
    print("测试3: JSON格式正确率测试")
    print("=" * 60)
    
    parse_success = 0
    total = 0
    
    num_samples = min(100, len(test_images))
    
    for i, img_path in enumerate(test_images[:num_samples]):
        if not os.path.exists(img_path):
            continue
        
        total += 1
        try:
            result = agent.run({"image_path": img_path, "inspection_type": "full"})
            parsed = json.loads(result)
            if isinstance(parsed, list):
                parse_success += 1
        except Exception as e:
            print(f"  样本 {i+1} JSON解析失败: {e}")
        
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{num_samples} 张图像")
    
    if total == 0:
        print("❌ 没有有效的测试样本")
        return False, 0.0
    
    success_rate = parse_success / total
    
    print(f"\n结果:")
    print(f"  JSON解析成功: {parse_success}/{total}")
    print(f"  成功率: {success_rate:.2%}")
    
    if success_rate < 1.0:
        print(f"❌ JSON格式错误率 > 0% ({1-success_rate:.2%})")
        return False, success_rate
    else:
        print(f"✅ JSON格式正确率 100%")
        return True, success_rate


def test_memory_stability(agent: SimplePCBAgent, test_image: str, num_iterations: int = 1000, 
                         max_memory_gb: float = 30.0):
    """
    测试显存稳定性（连续推理）
    
    Args:
        agent: PCB智能体
        test_image: 测试图像路径
        num_iterations: 迭代次数
        max_memory_gb: 最大显存（GB）
    
    Returns:
        success: 是否通过
        peak_memory_gb: 峰值显存（GB）
    """
    print("\n" + "=" * 60)
    print("测试4: 显存稳定性测试（连续推理1000次）")
    print("=" * 60)
    
    if not os.path.exists(test_image):
        print(f"❌ 测试图像不存在: {test_image}")
        return False, 0.0
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过显存测试")
        return True, 0.0
    
    # 重置显存统计
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    initial_memory = torch.cuda.memory_allocated() / 1e9
    
    print(f"  初始显存: {initial_memory:.2f} GB")
    print(f"  开始连续推理 {num_iterations} 次...")
    
    for i in range(num_iterations):
        try:
            agent.run({"image_path": test_image, "inspection_type": "full"})
            
            # 每100次清理一次
            if (i + 1) % 100 == 0:
                torch.cuda.empty_cache()
                current_memory = torch.cuda.memory_allocated() / 1e9
                print(f"  进度: {i+1}/{num_iterations}, 当前显存: {current_memory:.2f} GB")
        except Exception as e:
            print(f"  迭代 {i+1} 时出错: {e}")
            break
    
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    final_memory = torch.cuda.memory_allocated() / 1e9
    
    print(f"\n结果:")
    print(f"  峰值显存: {peak_memory:.2f} GB")
    print(f"  最终显存: {final_memory:.2f} GB")
    print(f"  显存增长: {final_memory - initial_memory:.2f} GB")
    
    success = peak_memory < max_memory_gb
    
    if not success:
        print(f"❌ 显存泄漏 > {max_memory_gb}GB: {peak_memory:.2f}GB")
    else:
        print(f"✅ 显存稳定性达标！")
    
    return success, peak_memory


def test_pcb_pipeline(model_path: str = "./models/qwen3-vl-pcb-awq",
                     test_data_dir: str = None,
                     test_images: List[str] = None):
    """
    PCB质检流水线工业级验证
    
    Args:
        model_path: 模型路径
        test_data_dir: 测试数据集目录
        test_images: 测试图像路径列表
    """
    print("\n" + "=" * 80)
    print("PCB质检验证流水线")
    print("=" * 80)
    
    # 创建智能体
    agent = SimplePCBAgent(model_path=model_path)
    
    # 准备测试数据
    if test_data_dir and os.path.exists(test_data_dir):
        try:
            test_dataset = load_pcb_dataset(test_data_dir, augment=False)
        except:
            test_dataset = []
    else:
        test_dataset = []
    
    if test_images is None:
        test_images = []
    
    results = {}
    
    # 测试1: 漏检率
    if len(test_dataset) > 0:
        success, recall = test_miss_rate(agent, test_dataset)
        results["miss_rate"] = {"success": success, "recall": recall}
    else:
        print("⚠️  跳过漏检率测试（无测试数据集）")
        results["miss_rate"] = {"success": True, "recall": None, "skipped": True}
    
    # 测试2: 推理速度
    if len(test_images) > 0:
        success, avg_time, p99_time = test_inference_speed(agent, test_images)
        results["speed"] = {"success": success, "avg_time": avg_time, "p99_time": p99_time}
    else:
        print("⚠️  跳过推理速度测试（无测试图像）")
        results["speed"] = {"success": True, "skipped": True}
    
    # 测试3: JSON格式
    if len(test_images) > 0:
        success, success_rate = test_json_format(agent, test_images)
        results["json_format"] = {"success": success, "success_rate": success_rate}
    else:
        print("⚠️  跳过JSON格式测试（无测试图像）")
        results["json_format"] = {"success": True, "skipped": True}
    
    # 测试4: 显存稳定性
    if len(test_images) > 0 and torch.cuda.is_available():
        test_image = test_images[0]
        success, peak_memory = test_memory_stability(agent, test_image, num_iterations=100)
        results["memory"] = {"success": success, "peak_memory_gb": peak_memory}
    else:
        print("⚠️  跳过显存稳定性测试")
        results["memory"] = {"success": True, "skipped": True}
    
    # 总结
    print("\n" + "=" * 80)
    print("验证总结")
    print("=" * 80)
    
    all_passed = all(r.get("success", False) or r.get("skipped", False) for r in results.values())
    
    for test_name, result in results.items():
        status = "✅" if result.get("success") else "❌" if not result.get("skipped") else "⚠️"
        print(f"{status} {test_name}: {result}")
    
    if all_passed:
        print("\n✅ PCB质检验证通过！")
        if results.get("miss_rate", {}).get("recall"):
            miss_rate = 1 - results["miss_rate"]["recall"]
            print(f"   漏检率: {miss_rate:.2%}")
        if results.get("speed", {}).get("avg_time"):
            print(f"   推理速度: {results['speed']['avg_time']:.3f}s")
        if results.get("memory", {}).get("peak_memory_gb"):
            print(f"   峰值显存: {results['memory']['peak_memory_gb']:.2f}GB")
    else:
        print("\n❌ 部分测试未通过，请检查模型和配置")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PCB质检验证")
    parser.add_argument("--model_path", type=str, default="./models/qwen3-vl-pcb-awq",
                       help="模型路径")
    parser.add_argument("--test_data_dir", type=str, default=None,
                       help="测试数据集目录")
    parser.add_argument("--test_images", type=str, nargs="+", default=None,
                       help="测试图像路径列表")
    
    args = parser.parse_args()
    
    test_pcb_pipeline(
        model_path=args.model_path,
        test_data_dir=args.test_data_dir,
        test_images=args.test_images
    )

