# 🧪 训练好的模型权重测试指南

## 📋 重要说明：训练输出的文件格式

**本项目不使用 `.pkl` 文件格式**，而是使用 HuggingFace Transformers 的标准格式：

- **LoRA 权重**：保存在 `checkpoints/pcb_checkpoints/final/`（包含 `adapter_config.json`, `adapter_model.safetensors` 等）
- **合并后的模型**：保存在 `models/qwen3-vl-pcb/`（标准 HuggingFace 格式）
- **量化后的模型**：保存在 `models/qwen3-vl-pcb-awq/`（用于推理，文件最小）

这些格式可以直接被 `transformers` 库的 `from_pretrained()` 方法加载，无需 `.pkl` 文件。

---

## 🚀 快速测试训练好的模型

### 方式1：使用测试脚本（推荐）

```bash
# 测试单张图像
python tools/test_trained_model.py \
    --model_path ./models/qwen3-vl-pcb-awq \
    --image_path ./data/test_images/board_001.jpg

# 批量测试多张图像
python tools/test_trained_model.py \
    --model_path ./models/qwen3-vl-pcb-awq \
    --image_dir ./data/test_images/

# 指定检测类型（只检测短路缺陷）
python tools/test_trained_model.py \
    --model_path ./models/qwen3-vl-pcb-awq \
    --image_path test.jpg \
    --type short
```

### 方式2：使用 Python 代码

```python
from src.inference.pcb_agent import SimplePCBAgent

# 加载训练好的模型（推荐使用量化模型）
agent = SimplePCBAgent(model_path="./models/qwen3-vl-pcb-awq")

# 执行检测
defects = agent.inspect(
    image_path="test_image.jpg",
    inspection_type="full"  # 或 "short", "open", "missing"
)

# 查看结果
for defect in defects:
    print(f"类型: {defect['defect']}")
    print(f"边界框: {defect['bbox']}")
    print(f"维修建议: {defect['repair']}")
```

### 方式3：使用完整验证流程（工业级测试）

```bash
# 完整验证（包括漏检率、推理速度、JSON格式等）
python src/inference/validation_pcb.py \
    --model_path ./models/qwen3-vl-pcb-awq \
    --test_data_dir ./data/pcb_defects \
    --test_images ./data/test_images/*.jpg
```

---

## 📁 训练输出的文件结构

### LoRA 权重（训练完成后）

```
checkpoints/pcb_checkpoints/final/
├── adapter_config.json       # LoRA 配置
├── adapter_model.safetensors # LoRA 权重（注意：不是 .pkl）
└── tokenizer_config.json     # Tokenizer 配置
```

**使用方式**：需要先合并到基础模型才能使用（见下方"模型合并"步骤）。

### 合并后的模型（Day 3 输出）

```
models/qwen3-vl-pcb/
├── config.json
├── generation_config.json
├── model.safetensors         # 完整模型权重
├── tokenizer.json
└── ...（其他配置文件）
```

**使用方式**：可以直接使用，但文件较大（约 60GB）。

### 量化后的模型（Day 4 输出，推荐用于推理）

```
models/qwen3-vl-pcb-awq/
├── config.json
├── generation_config.json
├── model.safetensors         # 4-bit 量化权重（约 18-20GB）
├── tokenizer.json
└── ...（其他配置文件）
```

**使用方式**：推荐用于推理，文件最小，速度最快。

---

## 🔄 完整的训练和测试流程

### 步骤1：训练模型（Day 1-2）

```bash
python src/train/pcb_train.py \
    --data_dir ./data/pcb_defects \
    --output_dir ./checkpoints/pcb_checkpoints
```

**输出**：`checkpoints/pcb_checkpoints/final/`（LoRA 权重）

### 步骤2：合并模型（Day 3）

```bash
python src/train/merge_model.py \
    --base_model Qwen/Qwen3-VL-32B-Instruct \
    --lora_checkpoint ./checkpoints/pcb_checkpoints/final \
    --output_dir ./models/qwen3-vl-pcb
```

**输出**：`models/qwen3-vl-pcb/`（完整模型）

### 步骤3：量化模型（Day 4，可选但推荐）

```bash
python src/train/quantize_model.py \
    --model_path ./models/qwen3-vl-pcb \
    --output_dir ./models/qwen3-vl-pcb-awq
```

**输出**：`models/qwen3-vl-pcb-awq/`（量化模型，推荐用于推理）

### 步骤4：测试模型（Day 5）

```bash
# 简单测试
python tools/test_trained_model.py \
    --model_path ./models/qwen3-vl-pcb-awq \
    --image_path test.jpg

# 完整验证
python src/inference/validation_pcb.py \
    --model_path ./models/qwen3-vl-pcb-awq \
    --test_data_dir ./data/pcb_defects
```

---

## 🎯 不同模型路径的使用场景

| 模型路径 | 文件大小 | 使用场景 | 加载方式 |
|---------|---------|---------|---------|
| `checkpoints/.../final/` | 小（几MB） | 需要基础模型+LoRA | 先合并，不能直接使用 |
| `models/qwen3-vl-pcb/` | 大（约60GB） | 完整模型，用于进一步量化 | `SimplePCBAgent(model_path=...)` |
| `models/qwen3-vl-pcb-awq/` | 中（约18-20GB） | **推理推荐** | `SimplePCBAgent(model_path=...)` |

---

## ⚠️ 常见问题

### Q1: 训练完成后只有 LoRA 权重，如何测试？

**A**: LoRA 权重不能直接用于推理，需要先合并到基础模型：

```bash
# 1. 合并 LoRA 权重
python src/train/merge_model.py \
    --base_model Qwen/Qwen3-VL-32B-Instruct \
    --lora_checkpoint ./checkpoints/pcb_checkpoints/final \
    --output_dir ./models/qwen3-vl-pcb

# 2. 然后测试合并后的模型
python tools/test_trained_model.py \
    --model_path ./models/qwen3-vl-pcb \
    --image_path test.jpg
```

### Q2: 为什么没有 `.pkl` 文件？

**A**: 本项目使用 HuggingFace Transformers 标准格式（`.safetensors`、`.bin` 等），这些格式：
- 更安全（safetensors 格式）
- 更标准化（HuggingFace 生态）
- 更容易管理和分享
- 不需要额外的序列化/反序列化代码

### Q3: 如何检查模型文件是否完整？

**A**: 检查模型目录中是否包含必要文件：

```bash
# 检查量化模型（推荐）
ls -lh models/qwen3-vl-pcb-awq/
# 应该看到：config.json, model.safetensors, tokenizer.json 等

# 尝试加载模型
python -c "from src.inference.pcb_agent import SimplePCBAgent; agent = SimplePCBAgent('./models/qwen3-vl-pcb-awq'); print('模型加载成功')"
```

### Q4: 测试时提示模型路径不存在怎么办？

**A**: 
1. 检查路径是否正确（注意使用绝对路径或相对路径）
2. 确认模型已完成训练和合并
3. 如果使用 LoRA 权重，需要先合并

### Q5: 如何只测试 LoRA 权重（不合并）？

**A**: LoRA 权重不能直接测试，必须合并。但可以使用合并后的模型进行测试：

```bash
# LoRA 权重必须先合并
python src/train/merge_model.py \
    --lora_checkpoint ./checkpoints/pcb_checkpoints/final \
    --output_dir ./models/qwen3-vl-pcb-test

# 然后测试
python tools/test_trained_model.py \
    --model_path ./models/qwen3-vl-pcb-test \
    --image_path test.jpg
```

---

## 📊 测试结果解读

### 单张图像测试输出

```
检测到 2 个缺陷:
------------------------------------------------------------

缺陷 1:
  类型: short
  边界框: [120, 350, 45, 12]
  置信度: 0.98
  维修建议: 清理焊锡桥接，检查相邻焊盘

缺陷 2:
  类型: open
  边界框: [200, 150, 30, 8]
  置信度: 0.95
  维修建议: 补焊连接，检查线路完整性
```

### 批量测试输出

```
批量测试完成
============================================================
总图像数: 100
成功处理: 98
总缺陷数: 156
结果已保存到: batch_test_results.json
```

### 完整验证输出

```
PCB质检验证流水线
============================================================
✅ miss_rate: {'success': True, 'recall': 0.99}
✅ speed: {'success': True, 'avg_time': 0.85, 'p99_time': 1.2}
✅ json_format: {'success': True, 'success_rate': 1.0}
✅ memory: {'success': True, 'peak_memory_gb': 24.5}

✅ PCB质检验证通过！
   漏检率: 1.00%
   推理速度: 0.850s
   峰值显存: 24.50GB
```

---

## 🔗 相关文档

- [RUN_GUIDE.md](RUN_GUIDE.md) - 完整运行指南
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [README.md](../README.md) - 项目总览

