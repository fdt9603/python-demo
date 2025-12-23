#!/usr/bin/env python3
"""
训练监控脚本
实时监控训练损失、GPU使用情况等
"""
import json
import os
import time
import sys
from pathlib import Path
import subprocess

def get_gpu_info():
    """获取GPU使用情况"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpu_info.append({
                        'index': parts[0],
                        'name': parts[1],
                        'util': parts[2],
                        'mem_used': parts[3],
                        'mem_total': parts[4],
                        'temp': parts[5]
                    })
            return gpu_info
    except:
        pass
    return None

def read_trainer_state(state_file):
    """读取训练状态文件"""
    if not os.path.exists(state_file):
        return None
    
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        return state
    except:
        return None

def format_loss(loss):
    """格式化损失值"""
    if loss is None:
        return "N/A"
    if loss == 0.0:
        return "0.0 ⚠️"
    if loss > 1000:
        return f"{loss:.2e} ⚠️⚠️"
    return f"{loss:.4f}"

def check_training_process():
    """检查是否有训练进程在运行"""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # 检查是否有 pcb_train.py 进程
            lines = result.stdout.split('\n')
            for line in lines:
                if 'pcb_train.py' in line and 'python' in line:
                    return True
    except:
        pass
    return False

def monitor_training(output_dir, refresh_interval=5):
    """
    监控训练过程
    
    Args:
        output_dir: 训练输出目录（例如 ./checkpoints/pcb_checkpoints-test）
        refresh_interval: 刷新间隔（秒）
    """
    print("=" * 80)
    print("训练监控工具")
    print("=" * 80)
    print(f"监控目录: {output_dir}")
    print(f"刷新间隔: {refresh_interval}秒")
    print("按 Ctrl+C 退出监控")
    print("=" * 80)
    print()
    
    last_step = 0
    last_loss = None
    
    try:
        while True:
            # 清屏（可选，如果不想清屏可以注释掉）
            # os.system('clear' if os.name != 'nt' else 'cls')
            
            # 检查训练进程是否在运行
            training_running = check_training_process()
            
            # 获取GPU信息（用于判断训练是否在进行）
            gpu_info = get_gpu_info()
            gpu_high_usage = False
            if gpu_info:
                for gpu in gpu_info:
                    try:
                        util = int(gpu['util'])
                        if util > 50:  # GPU使用率超过50%可能表示训练在进行
                            gpu_high_usage = True
                            break
                    except:
                        pass
            
            # 查找最新的checkpoint或final目录
            state_file = None
            checkpoint_dirs = []
            
            # 检查final目录
            final_state = os.path.join(output_dir, "final", "trainer_state.json")
            if os.path.exists(final_state):
                state_file = final_state
                checkpoint_dirs.append("final")
            
            # 检查checkpoint目录
            if os.path.exists(output_dir):
                for item in os.listdir(output_dir):
                    checkpoint_path = os.path.join(output_dir, item)
                    if os.path.isdir(checkpoint_path) and item.startswith("checkpoint-"):
                        checkpoint_dirs.append(item)
                        checkpoint_state = os.path.join(checkpoint_path, "trainer_state.json")
                        if os.path.exists(checkpoint_state):
                            state_file = checkpoint_state
            
            # 使用最新的checkpoint
            if checkpoint_dirs:
                checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else 0)
                latest_checkpoint = checkpoint_dirs[-1]
                latest_state = os.path.join(output_dir, latest_checkpoint, "trainer_state.json")
                if os.path.exists(latest_state):
                    state_file = latest_state
            
            # 读取训练状态
            state = read_trainer_state(state_file) if state_file else None
            
            # 显示时间
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{current_time}]")
            
            # 如果训练进程在运行或GPU使用率高，但还没有checkpoint，显示训练中
            if (training_running or gpu_high_usage) and not state_file:
                print("🔄 训练正在进行中...")
                print("   (等待第一个checkpoint保存，save_steps=50)")
                if training_running:
                    print("   ✅ 检测到训练进程")
                if gpu_high_usage:
                    print("   ✅ GPU使用率较高，训练可能在进行")
            
            if state and 'log_history' in state:
                logs = state['log_history']
                if logs:
                    # 获取最新的损失
                    recent_logs = [log for log in logs if 'loss' in log]
                    if recent_logs:
                        latest_log = recent_logs[-1]
                        current_step = latest_log.get('step', 0)
                        current_loss = latest_log.get('loss')
                        
                        # 显示训练进度
                        if 'max_steps' in state:
                            max_steps = state.get('max_steps', 'N/A')
                            progress = f"{current_step}/{max_steps}" if isinstance(max_steps, int) else f"{current_step}/?"
                            print(f"📊 训练进度: {progress} 步")
                        else:
                            print(f"📊 当前步数: {current_step}")
                        
                        print(f"📉 当前损失: {format_loss(current_loss)}")
                        
                        # 显示损失趋势
                        if len(recent_logs) >= 2:
                            prev_loss = recent_logs[-2].get('loss')
                            if prev_loss and current_loss:
                                diff = current_loss - prev_loss
                                trend = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                                print(f"📈 损失变化: {trend} {diff:+.6f}")
                        
                        # 显示最近5步的损失
                        if len(recent_logs) >= 5:
                            print("\n最近5步损失:")
                            for log in recent_logs[-5:]:
                                step = log.get('step', 'N/A')
                                loss = log.get('loss')
                                print(f"  步 {step}: {format_loss(loss)}")
                        
                        # 检查异常
                        if current_loss == 0.0:
                            print("\n⚠️  警告: 损失为0，可能存在问题！")
                        elif current_loss and current_loss > 1000:
                            print("\n⚠️  警告: 损失异常大，可能存在数值溢出！")
                        
                        last_step = current_step
                        last_loss = current_loss
                    else:
                        print("⏳ 等待训练开始...")
                else:
                    print("⏳ 等待训练开始...")
            else:
                if not training_running and not gpu_high_usage:
                    print("⏳ 等待训练开始...")
                    if state_file:
                        print(f"   状态文件: {state_file}")
                    else:
                        print(f"   检查目录: {output_dir}")
                # 如果训练在进行但还没有checkpoint，已经在上面显示了
            
            # 显示GPU信息
            if gpu_info:
                print("\n🖥️  GPU使用情况:")
                for gpu in gpu_info:
                    print(f"   GPU {gpu['index']}: {gpu['util']}% | "
                          f"显存: {gpu['mem_used']}/{gpu['mem_total']}MB | "
                          f"温度: {gpu['temp']}°C")
            
            print("\n" + "-" * 80)
            print(f"下次刷新: {refresh_interval}秒后 (按 Ctrl+C 退出)")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 监控出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="监控训练过程")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="训练输出目录（例如 ./checkpoints/pcb_checkpoints-test）"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="刷新间隔（秒），默认5秒"
    )
    
    args = parser.parse_args()
    
    monitor_training(args.output_dir, args.interval)

