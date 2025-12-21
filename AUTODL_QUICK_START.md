# Autodl A800 快速启动指南

专为Autodl A800 80GB服务器优化的快速启动指南。

## ✅ 服务器配置确认

你的服务器配置：
- ✅ GPU: A800 80GB（满足要求）
- ✅ 内存: 100GB（充足）
- ✅ 存储: 200GB（充足）

## 🚀 快速开始（5步）

### 步骤1: 环境检查（1分钟）

```bash
# 运行兼容性检查
python check_autodl_compatibility.py

# 或手动检查GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'显存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"
```

预期输出：
```
GPU: NVIDIA A800-SXM4-80GB (或类似)
显存: 80.0GB
```

### 步骤2: 安装依赖（5-10分钟）

```bash
# 安装所有依赖
pip install -r requirements.txt

# 如果网络慢，使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证关键依赖
python -c "import torch; import transformers; import peft; print('✅ 核心依赖安装成功')"
```

### 步骤3: 准备数据集（10-30分钟）

**方式A: 使用DeepPCB数据集（推荐）**

```bash
# 如果已有DeepPCB数据集
python convert_deeppcb_dataset.py --deeppcb_dir /path/to/DeepPCB-master

# 数据集会自动保存在 ./data/pcb_defects/
```

**方式B: 使用示例数据（快速测试）**

```bash
# 创建目录
mkdir -p data/pcb_defects/images

# 生成示例标签（需要先有一些测试图像）
python -c "from data_loader import create_sample_labels_json; create_sample_labels_json('data/pcb_defects/labels.json', 'data/pcb_defects/images', num_samples=10)"
```

### 步骤4: 配置检查（1分钟）

确认 `config.yaml` 配置适合A800：

```yaml
model:
  use_4bit: true  # ✅ 已启用，节省显存
  device_map: "auto"  # ✅ 自动分配

training:
  batch_size: 1  # ✅ 保守配置，适合A800
  gradient_accumulation_steps: 16  # ✅ 有效batch size = 16
```

**无需修改**，默认配置已优化！

### 步骤5: 开始训练

```bash
# 启动训练（后台运行，避免SSH断开）
nohup python pcb_train.py \
    --data_dir ./data/pcb_defects \
    --output_dir ./checkpoints/pcb_checkpoints \
    > train.log 2>&1 &

# 查看训练日志
tail -f train.log

# 或者使用screen（推荐）
screen -S pcb_train
python pcb_train.py --data_dir ./data/pcb_defects --output_dir ./checkpoints/pcb_checkpoints
# 按 Ctrl+A 然后 D 分离会话
# 重新连接: screen -r pcb_train
```

## 📊 资源监控

### 实时监控GPU

```bash
# 方法1: 使用watch
watch -n 1 nvidia-smi

# 方法2: 使用gpustat（如果已安装）
pip install gpustat
gpustat -i 1
```

### 监控磁盘空间

```bash
# 查看磁盘使用
df -h .

# 查看各目录大小
du -sh models/* checkpoints/* data/* 2>/dev/null
```

### 监控内存

```bash
# 查看内存使用
free -h

# 查看进程内存
top -p $(pgrep -f pcb_train)
```

## ⚡ 性能优化建议（可选）

如果你的A800显存充足（使用<60GB），可以尝试：

### 1. 增加batch size（可选）

编辑 `config.yaml`:

```yaml
training:
  batch_size: 2  # 从1增加到2
  gradient_accumulation_steps: 8  # 从16减少到8（保持有效batch size=16）
```

### 2. 使用混合精度（已自动启用）

代码会自动检测bf16支持：
- A800支持bf16，会自动启用
- 无需手动配置

### 3. 多GPU训练（如果有多块A800）

```bash
# 使用accelerate配置多GPU
accelerate config

# 然后使用accelerate启动
accelerate launch pcb_train.py --data_dir ./data/pcb_defects
```

## 🐛 常见问题

### Q1: 显存不足怎么办？

**现象**: 训练时报错 `CUDA out of memory`

**解决**:
```yaml
# config.yaml 中调整
training:
  batch_size: 1  # 保持为1
  gradient_accumulation_steps: 32  # 增加到32（如果原来是16）
```

### Q2: 磁盘空间不足怎么办？

**解决**:
```bash
# 1. 清理HuggingFace缓存（节省~80GB）
rm -rf ~/.cache/huggingface/hub/models--Qwen*

# 2. 删除不需要的checkpoint
# 只保留最新的几个checkpoint

# 3. 训练完成后删除基础模型（只保留量化模型）
rm -rf models/qwen3-vl-pcb
```

### Q3: 训练中断怎么办？

**解决**:
```bash
# checkpoint会自动保存，可以从checkpoint恢复
python pcb_train.py \
    --data_dir ./data/pcb_defects \
    --output_dir ./checkpoints/pcb_checkpoints \
    --resume_from_checkpoint ./checkpoints/pcb_checkpoints/checkpoint-500
```

### Q4: 如何查看训练进度？

```bash
# 查看日志
tail -f train.log

# 或者查看最新checkpoint
ls -lh checkpoints/pcb_checkpoints/
```

## 📈 预期时间线

| 阶段 | 预计时间 | 显存使用 | 磁盘使用 |
|------|----------|----------|----------|
| 数据准备 | 10-30分钟 | <1GB | ~5GB |
| 模型下载 | 10-30分钟 | <1GB | ~80GB |
| 训练（2000步） | 2-3天 | ~35-50GB | +50GB |
| 模型合并 | 30-45分钟 | ~60GB | +60GB |
| AWQ量化 | 3-5小时 | ~50GB | +25GB |

**总计**: 约3-4天完成完整流程

## ✅ 验证清单

运行前确认：

- [ ] GPU可用（`nvidia-smi`显示A800）
- [ ] CUDA可用（`python -c "import torch; print(torch.cuda.is_available())"`）
- [ ] 依赖已安装（`python check_autodl_compatibility.py`）
- [ ] 数据集已准备（`ls data/pcb_defects/labels.json`）
- [ ] 磁盘空间充足（`df -h .`显示至少150GB可用）
- [ ] config.yaml配置正确（`use_4bit: true`）

## 🎯 下一步

训练完成后：

1. **合并模型**: `python merge_model.py ...`
2. **量化模型**: `python quantize_model.py ...`
3. **验证模型**: `python validation_pcb.py ...`
4. **部署服务**: `python mllm_api.py ...`

详细步骤请参考 [QUICKSTART.md](QUICKSTART.md)

---

**提示**: Autodl A800完全兼容，无需任何代码修改即可运行！

