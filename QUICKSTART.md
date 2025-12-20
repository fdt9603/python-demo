# 快速开始指南

## 📋 前置准备

### 1. 环境要求
- Python 3.8+
- CUDA 11.8+（GPU必需）
- A100 80GB GPU（推荐）或类似规格
- 200GB存储空间

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 快速开始（完整流程）

### Step 1: 准备数据集

创建数据集目录结构：
```bash
mkdir -p data/pcb_defects/images
```

准备你的电路板图像和标签文件：

**方式A：手动创建 labels.json**
```json
[
  {
    "image": "board_001.jpg",
    "defects": [
      {
        "type": "short",
        "bbox": [120, 350, 45, 12],
        "repair": "清理焊锡桥接"
      }
    ]
  },
  {
    "image": "board_002.jpg",
    "defects": []
  }
]
```

**方式B：使用示例生成器**
```bash
python -c "from data_loader import create_sample_labels_json; create_sample_labels_json('data/pcb_defects/labels.json', 'data/pcb_defects/images', num_samples=10)"
```

### Step 2: 训练模型（Day 1-2）

```bash
python pcb_train.py \
    --data_dir ./data/pcb_defects \
    --output_dir ./checkpoints/pcb_checkpoints \
    --max_steps 2000 \
    --batch_size 1 \
    --gradient_accumulation_steps 16
```

**提示**：如果显存不足，可以使用 `--no_4bit` 禁用4-bit量化（需要更多显存）

**监控存储**（可选，后台运行）：
```bash
chmod +x storage_monitor.sh
./storage_monitor.sh &
```

### Step 3: 合并模型（Day 3）

```bash
python merge_model.py \
    --base_model Qwen/Qwen3-VL-32B-Instruct \
    --lora_checkpoint ./checkpoints/pcb_checkpoints/final \
    --output_dir ./models/qwen3-vl-pcb
```

### Step 4: 量化模型（Day 4）

```bash
python quantize_model.py \
    --model_path ./models/qwen3-vl-pcb \
    --output_dir ./models/qwen3-vl-pcb-awq \
    --num_calib_samples 200
```

**注意**：量化过程可能需要几小时，请耐心等待。

### Step 5: 测试模型（Day 7）

```bash
# 单张图像测试
python pcb_agent.py \
    --image_path ./data/test_image.jpg \
    --inspection_type full \
    --model_path ./models/qwen3-vl-pcb-awq

# 完整验证
python validation_pcb.py \
    --model_path ./models/qwen3-vl-pcb-awq \
    --test_data_dir ./data/pcb_test \
    --test_images ./data/test_images/*.jpg
```

### Step 6: 部署服务（Day 8）

#### 方式A：API服务

```bash
# 启动API服务
python mllm_api.py \
    --host 0.0.0.0 \
    --port 8000 \
    --model_path ./models/qwen3-vl-pcb-awq
```

访问文档：http://localhost:8000/docs

#### 方式B：批量处理

```bash
# 准备输入目录
mkdir -p data/pcb_input
# 将图像放入 data/pcb_input/

# 使用部署脚本
chmod +x deploy_pcb.sh
./deploy_pcb.sh
```

## 🔧 常见问题

### Q1: 显存不足怎么办？

**A**: 尝试以下方法：
1. 使用 `--no_4bit` 参数（但这需要更多显存）
2. 减少 `batch_size` 和 `gradient_accumulation_steps`
3. 使用更小的模型（如果有）

### Q2: 训练很慢怎么办？

**A**: 
1. 确保使用了GPU加速
2. 检查 `bf16` 是否启用（需要A100/H100）
3. 增加 `gradient_accumulation_steps` 而不是 `batch_size`

### Q3: JSON格式错误怎么办？

**A**:
1. 检查prompt中的格式约束
2. 增加 `max_new_tokens` 参数
3. 降低 `temperature` 到 0.1

### Q4: 漏检率高怎么办？

**A**:
1. 增加数据增强倍数（修改 `data_loader.py` 中的增强次数）
2. 降低置信度阈值（`config.yaml` 中的 `confidence_threshold`）
3. 增加训练步数 `max_steps`

## 📊 预期时间线

| 阶段 | 预计时间 | 说明 |
|------|----------|------|
| 数据准备 | 1天 | 根据数据量调整 |
| 模型训练 | 2天 | 2000步，A100约需2天 |
| 模型合并 | 0.5小时 | 快速完成 |
| 模型量化 | 3-4小时 | AWQ量化 |
| 智能体开发 | 1-2天 | 开发和测试 |
| 验证测试 | 1天 | 工业级验证 |
| 部署 | 0.5天 | 部署和文档 |

**总计：约8天**

## 🎯 关键指标检查清单

训练完成后，检查以下指标：

- [ ] 漏检率 < 1%
- [ ] 推理速度 < 1秒/张
- [ ] JSON格式正确率 = 100%
- [ ] 显存占用 < 25GB（推理时）
- [ ] 模型大小 ~25GB（4-bit AWQ）

## 📝 下一步

1. 根据实际数据调整 `config.yaml` 中的参数
2. 优化prompt以提高检测精度
3. 增加难例挖掘以提高模型鲁棒性
4. 部署到生产环境并监控性能

## 💡 提示

- 训练过程中可以随时使用 `Ctrl+C` 停止，模型会自动保存
- 使用 `tensorboard` 监控训练过程（如果配置了）
- 建议使用 `screen` 或 `tmux` 在后台运行长时间任务
- 定期备份checkpoint，防止意外中断

