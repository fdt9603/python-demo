# PCB电路板缺陷检测系统

基于Qwen3-VL-32B-Instruct的工业级电路板缺陷检测MLLM系统，支持缺陷识别、定位和维修报告生成。

## 🎯 项目特性

- ✅ **高精度检测**：漏检率 < 1%（工业红线）
- ✅ **快速推理**：BitsAndBytes 4-bit 运行时量化 < 1秒/张
- ✅ **结构化输出**：强制JSON格式，100%解析成功率
- ✅ **低显存占用**：推理时 < 25GB（A100-80GB优化）
- ✅ **完整流程**：从数据准备到部署的8天完整方案
- ✅ **向量数据库**：历史案例存储和相似缺陷检索
- ✅ **LangGraph工作流**：多步骤智能体自动化流程

## 📋 系统要求

- **GPU**: A100 80GB（推荐）或类似规格（**A800 80GB、RTX PRO 6000 96GB完全兼容**✅）
- **存储**: 200GB可用空间（训练阶段），27GB（生产部署）
- **内存**: 建议32GB+（100GB更佳）
- **Python**: 3.8+
- **CUDA**: 11.8+

> 💡 **兼容性说明**: 
> - **Autodl A800用户**: 项目完全兼容A800 80GB，无需修改代码。查看 [AUTODL_A800_COMPATIBILITY.md](docs/AUTODL_A800_COMPATIBILITY.md)
> - **RTX PRO 6000用户**: 项目完全兼容RTX PRO 6000 96GB，显存更大更稳定。查看 [RTX_PRO_6000_COMPATIBILITY.md](docs/RTX_PRO_6000_COMPATIBILITY.md)

## 🚀 快速开始

### ⚠️ 重要：数据集需要自己准备

**本项目不会自动下载数据集**，你需要：
1. 准备PCB图像文件（放在 `data/pcb_defects/images/`）
2. 创建标签文件 `data/pcb_defects/labels.json`

详细说明请查看 [RUN_GUIDE.md](docs/RUN_GUIDE.md)

### 1. 快速检查（推荐先运行）

```bash
python tools/quick_start.py
```

这会检查：
- ✅ 环境配置
- ✅ 数据集配置
- ✅ 模型文件
- ✅ 给出下一步建议

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

#### 方式A：使用DeepPCB数据集（推荐，1500张图像）

如果你有DeepPCB数据集，可以使用转换脚本自动转换：

```bash
# 转换DeepPCB数据集为项目格式
python tools/convert_deeppcb_dataset.py --deeppcb_dir /path/to/DeepPCB-master

# 转换完成后，数据集将保存在 ./data/pcb_defects/ 目录
```

详细转换说明请查看 [DEEPPCB_CONVERSION_GUIDE.md](docs/DEEPPCB_CONVERSION_GUIDE.md)

#### 方式B：使用自定义数据集

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

#### 方式C：生成示例数据（用于测试）

```bash
python -c "from src.data.data_loader import load_pcb_dataset; help(load_pcb_dataset)"  # 查看示例
```

### 3. 训练模型（Day 1-2）

**训练监控（可选）**：若要在训练时监控损失是否出现崩坏，可在另一个终端运行：

```bash
python tools/monitor_training.py --output_dir ./checkpoints/pcb_checkpoints --interval 5
```

**方式A：本地已有完整模型（推荐）**

如果模型已下载到本地（如 `./modelscope_cache/qwen/Qwen3-VL-32B-Instruct`），直接传入本地路径，**不需要** `--use_modelscope`：

```bash
# 单行命令（推荐，避免换行问题）
python src/train/pcb_train.py --data_dir ./data/pcb_defects --output_dir ./checkpoints/pcb_checkpoints --model_name ./modelscope_cache/qwen/Qwen3-VL-32B-Instruct --max_steps 1000 --batch_size 1 --gradient_accumulation_steps 16 --learning_rate 1e-4 --save_steps 50 --no_4bit
```

或者使用多行格式（确保每行末尾有反斜杠 `\`，最后一行不要有）：

```bash
# 训练（不使用4-bit量化，训练完成后会自动合并LoRA权重）
python src/train/pcb_train.py \
    --data_dir ./data/pcb_defects \
    --output_dir ./checkpoints/pcb_checkpoints \
    --model_name ./modelscope_cache/qwen/Qwen3-VL-32B-Instruct \
    --max_steps 1000 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --save_steps 50 \
    --no_4bit
```

> **说明**：
> - 训练脚本会在训练完成后**自动合并LoRA权重**，保存到 `./checkpoints/pcb_checkpoints/final`
> - 训练时**必须使用 `--no_4bit`**，因为4-bit量化与LoRA训练不兼容
> - 训练完成后，可以对合并后的模型进行量化用于推理优化

**方式B：从 ModelScope 下载或检查模型**

如果需要从 ModelScope 下载模型，或让 ModelScope 检查/补全本地缓存，使用 `--use_modelscope`：

```bash
# 单行命令（推荐，避免换行问题）
python src/train/pcb_train.py --data_dir ./data/pcb_defects --output_dir ./checkpoints/pcb_checkpoints --model_name ./modelscope_cache/qwen/Qwen3-VL-32B-Instruct --max_steps 1000 --batch_size 1 --gradient_accumulation_steps 16 --learning_rate 1e-4 --save_steps 50 --no_4bit --use_modelscope
```

或者使用多行格式：

```bash
python src/train/pcb_train.py \
    --data_dir ./data/pcb_defects \
    --output_dir ./checkpoints/pcb_checkpoints \
    --model_name ./modelscope_cache/qwen/Qwen3-VL-32B-Instruct \
    --max_steps 1000 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --save_steps 50 \
    --no_4bit \
    --use_modelscope
```

> **说明**：如果本地已有完整模型，`snapshot_download` 会直接返回本地路径，不会重复下载。但如果本地模型文件不完整（如缺少 `processor_config.json`），建议重新下载或使用 `--use_modelscope` 让 ModelScope 自动补全。

### 4. 量化模型（Day 4，可选但推荐）

**唯一支持：BitsAndBytes 4-bit 运行时量化（兼容Qwen3-VL）**

训练完成后，可以对合并后的模型进行量化，用于推理优化（显存占用更小，速度更快）：

```bash
python src/train/quantize_model_bnb.py \
    --model_path ./checkpoints/pcb_checkpoints/final \
    --output_dir ./models/qwen3-vl-pcb-bnb \
    --use_4bit
```

> **说明**：
> - BitsAndBytes是运行时量化，保存的权重仍为基础权重；加载时必须再次传入相同的 `BitsAndBytesConfig`（脚本会在输出目录生成 `load_quantized_model.py` 示例）
> - 量化是可选的，如果不量化，可以直接使用 `./checkpoints/pcb_checkpoints/final` 进行推理，但显存占用会更大
> - AWQ 脚本已移除（Qwen3-VL 暂不支持标准 AWQ 流程）

> 说明：BitsAndBytes是运行时量化，保存的权重仍为基础权重；加载时必须再次传入相同的 `BitsAndBytesConfig`（脚本会在输出目录生成 `load_quantized_model.py` 示例）。  
> AWQ 脚本已移除（Qwen3-VL 暂不支持标准 AWQ 流程）。

### 5. 验证模型（Day 7）

验证训练后的模型效果：

```bash
# 验证合并后的模型（未量化）
python src/inference/validation_pcb.py \
  --model_path ./checkpoints/pcb_checkpoints/final \
  --test_data_dir ./tools/data/pcb_defects \
  --max_test_samples 10

# 或验证量化后的模型
python src/inference/validation_pcb.py \
    --model_path ./models/qwen3-vl-pcb-bnb \
  --test_data_dir ./tools/data/pcb_defects \
  --max_test_samples 10
```

### 6. 部署服务（Day 8）

#### 方式A：使用部署脚本

```bash
chmod +x deploy_pcb.sh
./deploy_pcb.sh
```

#### 方式B：启动API服务

```bash
python src/inference/mllm_api.py \
    --host 0.0.0.0 \
    --port 8000 \
    --model_path ./models/qwen3-vl-pcb-bnb
```

API文档：http://localhost:8000/docs

#### 方式C：命令行使用

```bash
python src/inference/pcb_agent.py \
    --image_path ./data/test_image.jpg \
    --inspection_type full \
    --model_path ./models/qwen3-vl-pcb-bnb
```

## 📁 项目结构

```
.
├── tools/                       # 工具脚本目录
│   ├── convert_deeppcb_dataset.py  # DeepPCB数据集格式转换工具
│   ├── check_autodl_compatibility.py  # Autodl兼容性检查工具
│   └── quick_start.py          # 快速检查工具
├── src/                         # 核心源代码目录
│   ├── data/                   # 数据处理模块
│   │   ├── data_loader.py      # Day 0: 数据集加载和增强
│   │   └── dataset.py          # 数据集接口（向后兼容）
│   ├── train/                  # 训练相关模块
│   │   ├── pcb_train.py        # Day 1-2: 模型微调（自动合并LoRA）
│   │   ├── merge_model.py      # 手动合并工具（训练脚本已自动完成，此工具用于特殊场景）
│   │   └── quantize_model_bnb.py   # Day 4: BitsAndBytes量化
│   └── inference/              # 推理和部署模块
│       ├── pcb_agent.py        # Day 5-6: LangChain智能体
│       ├── vector_store.py     # 向量数据库模块（历史案例存储）
│       ├── pcb_graph.py        # LangGraph工作流模块
│       ├── validation_pcb.py   # Day 7: 工业级验证
│       └── mllm_api.py         # Day 8: FastAPI服务
├── examples/                    # 示例代码目录
│   ├── example_usage.py        # 使用示例
│   ├── main.py                 # 示例入口
│   └── test.py                 # 测试脚本
├── deploy_pcb.sh               # Day 8: 部署脚本
├── config.yaml                 # 配置文件
├── requirements.txt            # 依赖列表
├── README.md                   # 本文档
└── docs/                        # 文档目录
    ├── RUN_GUIDE.md            # 运行指南
    ├── QUICKSTART.md           # 快速开始
    ├── DEEPPCB_CONVERSION_GUIDE.md  # DeepPCB数据集转换指南
    ├── VECTOR_STORE_GUIDE.md   # 向量数据库和LangGraph使用指南
    ├── AUTODL_A800_COMPATIBILITY.md  # A800兼容性分析
    └── AUTODL_QUICK_START.md   # A800快速启动指南
```

## 🎖️ 性能指标

| 指标 | 目标 | 实际 |
|------|------|------|
| 漏检率 | < 1% | - |
| 误报率 | < 5% | - |
| 推理速度 | < 1秒/张 | - |
| JSON格式正确率 | 100% | - |
| 显存占用 | < 25GB | - |
| 模型大小 | ~25GB (4-bit BnB 运行时) | - |

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
| Day 1-2 | 模型微调 | LoRA检查点 + 自动合并的完整模型（保存在 `./checkpoints/pcb_checkpoints/final`） |
| Day 3 | - | -（合并已自动完成，无需单独步骤） |
| Day 4 | BitsAndBytes量化（可选） | 量化模型（运行时4bit，加载需传入配置） |
| Day 5-6 | 智能体开发 | src/inference/pcb_agent.py |
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
- 使用 BitsAndBytes 4-bit 量化模型（加载时传入相同的 `quantization_config`）
- 减少 `max_new_tokens`
- 使用 `do_sample=False`（贪心解码）

## 💾 存储空间管理

### 自动清理工具

项目提供了自动化清理脚本，**无需手动删除**：

```bash
# 1. 先模拟运行，查看将删除什么（推荐）
python tools/cleanup_storage.py --all --dry-run

# 2. 训练完成后，清理所有不需要的文件
python tools/cleanup_storage.py --all

# 3. 只清理特定项目
python tools/cleanup_storage.py --base-model      # 删除基础模型（~60GB）
python tools/cleanup_storage.py --merged-model   # 删除合并模型（~60GB）
python tools/cleanup_storage.py --checkpoints 2   # 只保留最新2个检查点
python tools/cleanup_storage.py --original-dataset # 删除原始数据集
python tools/cleanup_storage.py --cache           # 清理缓存文件
```

### 存储空间需求

| 阶段 | 所需空间 | 说明 |
|------|---------|------|
| **训练阶段** | ~63 GB | 代码 + 数据集 + 基础模型 + LoRA + 检查点 |
| **生产部署** | ~27 GB | 代码 + 数据集 + 量化模型（删除基础模型） |

**优化建议**：
- ✅ 训练完成后删除基础模型，节省 ~60GB
- ✅ 只保留量化模型用于推理，删除合并模型
- ✅ 定期清理旧检查点，只保留最新的1-2个
- ✅ 转换完成后可删除原始DeepPCB数据集

### 问题4：推理速度慢

**解决方案**：
- 使用 BitsAndBytes 4-bit 量化模型（加载时传入相同的 `quantization_config`）
- 减少 `max_new_tokens`
- 使用 `do_sample=False`（贪心解码）

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

详细使用指南请参考 [VECTOR_STORE_GUIDE.md](docs/VECTOR_STORE_GUIDE.md)

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

