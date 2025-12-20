# 🚀 项目运行指南

## 📋 重要说明：数据集需要自己准备

**本项目不会自动下载数据集**，你需要自己准备PCB缺陷数据集。有两种方式：

### 方式1：使用自己的数据集（推荐）

你需要准备：
1. **电路板图像**：放在 `data/pcb_defects/images/` 目录
2. **标签文件**：`data/pcb_defects/labels.json`

### 方式2：使用公开数据集（需要手动下载）

目前代码中提到的HuggingFace数据集（`hf-internal-testing/pcb-defects`）只是一个示例，实际可能不存在。你需要：
- 从Kaggle、GitHub等平台下载PCB缺陷数据集
- 转换为项目要求的格式

---

## 🎯 快速开始（3步运行）

### 第1步：安装依赖

```bash
pip install -r requirements.txt
```

**注意**：如果某些包安装失败（如autoawq、chromadb），可以先跳过，这些是可选的。

### 第2步：准备数据集

#### 选项A：创建示例数据集（用于测试）

```bash
# 创建目录
mkdir -p data/pcb_defects/images

# 生成示例labels.json（需要先有一些测试图像）
python -c "from data_loader import create_sample_labels_json; create_sample_labels_json('data/pcb_defects/labels.json', 'data/pcb_defects/images', num_samples=5)"
```

#### 选项B：使用真实数据集

1. 准备图像文件，放到 `data/pcb_defects/images/` 目录
2. 创建 `data/pcb_defects/labels.json` 文件，格式如下：

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

**数据集格式说明**：
- `image`: 图像文件名（需要在images目录下）
- `defects`: 缺陷列表
  - `type`: 缺陷类型（"short"/"open"/"missing"/"normal"）
  - `bbox`: 边界框 [x, y, width, height]
  - `repair`: 维修建议

### 第3步：运行项目

根据你的需求选择运行方式：

---

## 🔧 运行方式

### 方式1：完整训练流程（需要GPU和数据集）

```bash
# 1. 训练模型（需要GPU，可能需要几小时到几天）
python pcb_train.py --data_dir ./data/pcb_defects --output_dir ./checkpoints/pcb_checkpoints

# 2. 合并模型
python merge_model.py --base_model Qwen/Qwen3-VL-32B-Instruct --lora_checkpoint ./checkpoints/pcb_checkpoints/final --output_dir ./models/qwen3-vl-pcb

# 3. 量化模型（可选，但推荐）
python quantize_model.py --model_path ./models/qwen3-vl-pcb --output_dir ./models/qwen3-vl-pcb-awq

# 4. 使用智能体检测
python pcb_agent.py --image_path your_image.jpg --model_path ./models/qwen3-vl-pcb-awq
```

### 方式2：直接使用预训练模型（如果有）

如果你已经有训练好的模型：

```bash
# 使用基础智能体
python pcb_agent.py --image_path your_image.jpg --model_path ./models/qwen3-vl-pcb-awq

# 或使用LangGraph工作流（需要向量数据库）
python -c "from pcb_graph import PCBLangGraphAgent; agent = PCBLangGraphAgent(); result = agent.inspect('your_image.jpg'); print(result['repair_report'])"
```

### 方式3：启动API服务

```bash
# 启动FastAPI服务
python mllm_api.py --port 8000 --model_path ./models/qwen3-vl-pcb-awq

# 访问API文档
# http://localhost:8000/docs
```

### 方式4：测试数据加载（不需要模型）

```bash
# 测试数据集加载
python -c "from data_loader import load_pcb_dataset; dataset = load_pcb_dataset('data/pcb_defects'); print(f'数据集大小: {len(dataset)}')"
```

---

## 📊 数据集准备详细说明

### 最小数据集要求

- **至少需要**：10-20张图像（用于测试）
- **推荐数量**：500+张图像（用于训练）
- **图像格式**：JPG/PNG，建议尺寸 448x448 或更大
- **标签文件**：JSON格式，必须包含所有图像的标注

### 数据集目录结构

```
data/
  pcb_defects/
    images/
      board_001.jpg
      board_002.jpg
      ...
    labels.json
```

### 标签文件示例（完整版）

```json
[
  {
    "image": "board_001.jpg",
    "defects": [
      {
        "type": "short",
        "bbox": [120, 350, 45, 12],
        "repair": "清理焊锡桥接，检查相邻焊盘"
      }
    ]
  },
  {
    "image": "board_002.jpg",
    "defects": [
      {
        "type": "open",
        "bbox": [200, 150, 30, 8],
        "repair": "补焊连接，检查线路完整性"
      },
      {
        "type": "missing",
        "bbox": [300, 400, 20, 20],
        "repair": "补装缺失元件R12"
      }
    ]
  },
  {
    "image": "board_003.jpg",
    "defects": []
  }
]
```

---

## ⚠️ 常见问题

### Q1: 我没有数据集怎么办？

**A**: 你可以：
1. 使用 `create_sample_labels_json` 生成示例数据（但需要真实的图像文件）
2. 从公开数据集下载（如Kaggle的PCB数据集）
3. 自己标注一些图像

### Q2: 我没有GPU可以运行吗？

**A**: 
- **训练**：需要GPU（A100 80GB推荐）
- **推理**：可以使用CPU，但会很慢
- **测试数据加载**：不需要GPU，可以测试数据集格式

### Q3: 我没有训练好的模型怎么办？

**A**: 
- 你需要先完成训练流程（Day 1-4）
- 或者使用其他人训练好的模型
- 或者使用基础模型（Qwen3-VL-32B-Instruct）直接推理（效果较差）

### Q4: 如何快速测试项目是否配置正确？

**A**: 运行以下命令测试：

```bash
# 1. 测试数据加载
python -c "from data_loader import create_sample_labels_json; print('数据加载模块正常')"

# 2. 测试依赖
python -c "import torch; import transformers; print('核心依赖正常')"

# 3. 测试向量数据库（可选）
python -c "from vector_store import create_vector_store; store = create_vector_store(); print('向量数据库正常')"
```

### Q5: 训练需要多长时间？

**A**: 
- **数据准备**：1-2天（取决于数据量）
- **模型训练**：2-3天（A100，2000步）
- **模型量化**：3-4小时
- **总计**：约8天（按计划）

---

## 🎯 推荐运行顺序

### 第一次运行（测试环境）

1. ✅ 安装依赖：`pip install -r requirements.txt`
2. ✅ 创建示例数据集：使用 `create_sample_labels_json`
3. ✅ 测试数据加载：`python -c "from data_loader import load_pcb_dataset; ..."`
4. ✅ 检查GPU：`python -c "import torch; print(torch.cuda.is_available())"`

### 正式训练（需要GPU和真实数据）

1. ✅ 准备真实数据集（500+图像）
2. ✅ 开始训练：`python pcb_train.py --data_dir ./data/pcb_defects`
3. ✅ 监控训练过程（查看checkpoints目录）
4. ✅ 训练完成后合并和量化模型
5. ✅ 验证模型：`python validation_pcb.py`

### 生产使用（已有模型）

1. ✅ 启动API服务：`python mllm_api.py`
2. ✅ 或使用命令行：`python pcb_agent.py --image_path xxx.jpg`

---

## 📝 检查清单

运行前请确认：

- [ ] Python 3.8+ 已安装
- [ ] 依赖已安装：`pip install -r requirements.txt`
- [ ] GPU可用（训练必需）：`python -c "import torch; print(torch.cuda.is_available())"`
- [ ] 数据集已准备：`data/pcb_defects/images/` 和 `labels.json`
- [ ] 存储空间充足：至少200GB（训练时）

---

## 🆘 需要帮助？

如果遇到问题：
1. 查看 `QUICKSTART.md` 获取详细步骤
2. 查看 `README.md` 了解项目结构
3. 检查错误日志
4. 确认数据集格式是否正确

