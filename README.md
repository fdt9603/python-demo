# PCB电路板缺陷检测系统

基于Qwen3-VL-32B-Instruct的工业级电路板缺陷检测MLLM系统，支持缺陷识别、定位和维修报告生成。

## 🎯 项目特性

- ✅ **高精度检测**：漏检率 < 1%（工业红线）
- ✅ **快速推理**：AWQ量化后 < 1秒/张
- ✅ **结构化输出**：强制JSON格式，100%解析成功率
- ✅ **低显存占用**：推理时 < 25GB（A100-80GB优化）
- ✅ **完整流程**：从数据准备到部署的8天完整方案
- ✅ **向量数据库**：历史案例存储和相似缺陷检索
- ✅ **LangGraph工作流**：多步骤智能体自动化流程

## 📋 系统要求

- **GPU**: A100 80GB（推荐）或类似规格
- **存储**: 200GB可用空间
- **Python**: 3.8+
- **CUDA**: 11.8+

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

#### 方式A：使用自定义数据集（推荐）

数据集目录结构：
```
data/
  pcb_defects/
    images/
      board_001.jpg
      board_002.jpg
      ...
    labels.json
```

`labels.json` 格式：
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
  ...
]
```

#### 方式B：生成示例数据

```bash
python data_loader.py  # 查看示例
```

### 3. 训练模型（Day 1-2）

```bash
python pcb_train.py \
    --data_dir ./data/pcb_defects \
    --output_dir ./checkpoints/pcb_checkpoints \
    --max_steps 2000 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-4
```

### 4. 合并模型（Day 3）

```bash
python merge_model.py \
    --base_model Qwen/Qwen3-VL-32B-Instruct \
    --lora_checkpoint ./checkpoints/pcb_checkpoints/final \
    --output_dir ./models/qwen3-vl-pcb
```

### 5. 量化模型（Day 4）

```bash
python quantize_model.py \
    --model_path ./models/qwen3-vl-pcb \
    --output_dir ./models/qwen3-vl-pcb-awq \
    --num_calib_samples 200
```

### 6. 验证模型（Day 7）

```bash
python validation_pcb.py \
    --model_path ./models/qwen3-vl-pcb-awq \
    --test_data_dir ./data/pcb_test \
    --test_images ./data/test_images/*.jpg
```

### 7. 部署服务（Day 8）

#### 方式A：使用部署脚本

```bash
chmod +x deploy_pcb.sh
./deploy_pcb.sh
```

#### 方式B：启动API服务

```bash
python mllm_api.py \
    --host 0.0.0.0 \
    --port 8000 \
    --model_path ./models/qwen3-vl-pcb-awq
```

API文档：http://localhost:8000/docs

#### 方式C：命令行使用

```bash
python pcb_agent.py \
    --image_path ./data/test_image.jpg \
    --inspection_type full \
    --model_path ./models/qwen3-vl-pcb-awq
```

## 📁 项目结构

```
.
├── data_loader.py          # Day 0: 数据集加载和增强
├── pcb_train.py            # Day 1-2: 模型微调
├── merge_model.py          # Day 3: 模型合并
├── quantize_model.py       # Day 4: AWQ量化
├── pcb_agent.py            # Day 5-6: LangChain智能体
├── vector_store.py         # 向量数据库模块（历史案例存储）
├── pcb_graph.py            # LangGraph工作流模块
├── validation_pcb.py       # Day 7: 工业级验证
├── mllm_api.py             # Day 8: FastAPI服务
├── deploy_pcb.sh           # Day 8: 部署脚本
├── example_usage.py        # 使用示例
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖列表
├── README.md               # 本文档
└── VECTOR_STORE_GUIDE.md   # 向量数据库和LangGraph使用指南
```

## 🎖️ 性能指标

| 指标 | 目标 | 实际 |
|------|------|------|
| 漏检率 | < 1% | - |
| 误报率 | < 5% | - |
| 推理速度 | < 1秒/张 | - |
| JSON格式正确率 | 100% | - |
| 显存占用 | < 25GB | - |
| 模型大小 | ~25GB (4-bit AWQ) | - |

## ⚙️ 配置说明

主要配置在 `config.yaml` 中：

- **LoRA配置**：rank=16（缺陷模式简单）
- **图像分辨率**：448x448（Qwen3-VL最优）
- **冻结视觉塔**：是（电路板图像通用）
- **数据增强**：缺陷样本×10（解决类别不平衡）
- **推理温度**：0.1（客观任务，低随机性）

## 📊 8天作战计划

| 天数 | 任务 | 输出 |
|------|------|------|
| Day 0 | 数据集准备 | 预处理后的数据集 |
| Day 1-2 | 模型微调 | LoRA检查点 |
| Day 3 | 模型合并 | 合并后的模型 |
| Day 4 | AWQ量化 | 量化模型（25GB） |
| Day 5-6 | 智能体开发 | pcb_agent.py |
| Day 7 | 工业验证 | 验证报告 |
| Day 8 | 部署交付 | API服务 |

## 🔧 故障排查

### 问题1：显存不足

**解决方案**：
- 使用 `--no_4bit` 禁用4-bit加载（但需要更多显存）
- 减少 `batch_size` 和 `gradient_accumulation_steps`
- 使用CPU卸载：`device_map="auto"`

### 问题2：JSON格式错误

**解决方案**：
- 检查prompt中的格式约束
- 增加 `max_new_tokens`
- 降低 `temperature` 到 0.1

### 问题3：漏检率高

**解决方案**：
- 增加数据增强倍数
- 降低置信度阈值（`confidence_threshold`）
- 使用滑动窗口检测大图

### 问题4：推理速度慢

**解决方案**：
- 确保使用AWQ量化模型
- 检查是否使用了4-bit量化
- 减少 `max_new_tokens`

## 📝 数据格式说明

### 输入格式

- **图像**：JPG/PNG，推荐448x448，支持自动resize
- **标签JSON**：包含缺陷类型、边界框、维修建议

### 输出格式

```json
[
  {
    "defect": "short",
    "bbox": [120, 350, 45, 12],
    "confidence": 0.98,
    "repair": "清理焊锡桥接"
  },
  ...
]
```

## 🔄 向量数据库和LangGraph

本项目支持向量数据库（ChromaDB）和LangGraph工作流：

- **向量数据库**: 存储历史检测结果，支持相似缺陷案例检索
- **LangGraph**: 构建多步骤智能体工作流（检测→检索→报告→评估）

详细使用指南请参考 [VECTOR_STORE_GUIDE.md](VECTOR_STORE_GUIDE.md)

### 快速开始

```python
from pcb_graph import PCBLangGraphAgent

# 创建LangGraph智能体（自动使用向量数据库）
agent = PCBLangGraphAgent(model_path="./models/qwen3-vl-pcb-awq")

# 执行完整工作流
result = agent.inspect("board.jpg", inspection_type="full")

# 查看结果
print(f"缺陷: {len(result['defects'])}")
print(f"质量分数: {result['quality_score']:.2f}")
print(f"维修报告:\n{result['repair_report']}")
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目遵循MIT许可证。

## 🙏 致谢

- [Qwen3-VL](https://github.com/QwenLM/Qwen-VL)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [PEFT](https://github.com/huggingface/peft)

## 📧 联系方式

如有问题，请提交Issue。

---

**注意**：本项目为工业质检场景优化，需要根据实际数据调整参数。

