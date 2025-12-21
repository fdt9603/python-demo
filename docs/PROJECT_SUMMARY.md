# PCB电路板缺陷检测项目总结

## 📦 项目概述

这是一个完整的工业级PCB电路板缺陷检测系统，基于Qwen3-VL-32B-Instruct多模态大语言模型（MLLM），实现了从数据准备到部署的完整流程�?
## 🎯 核心功能

1. **缺陷检�?*：识别短�?short)、断�?open)、缺�?missing)等缺�?2. **精确定位**：返回缺陷的边界框坐�?[x, y, w, h]
3. **维修建议**：为每种缺陷生成具体的维修建�?4. **结构化输�?*：强制JSON格式，确�?00%解析成功�?
## 📁 项目文件结构

```
.
├── src/data/data_loader.py          # Day 0: 数据集加载、增强和预处�?├── src/train/pcb_train.py            # Day 1-2: 模型微调（LoRA�?├── src/train/merge_model.py          # Day 3: LoRA权重合并
├── src/train/quantize_model.py       # Day 4: AWQ量化
├── src/inference/pcb_agent.py            # Day 5-6: LangChain智能�?├── src/inference/validation_pcb.py       # Day 7: 工业级验�?├── src/inference/mllm_api.py             # Day 8: FastAPI REST服务
├── deploy_pcb.sh           # Day 8: 部署脚本
├── storage_monitor.sh      # 存储监控脚本
├── config.yaml             # 配置文件
├── requirements.txt        # Python依赖
├── README.md               # 项目文档
├── QUICKSTART.md           # 快速开始指�?└── dataset.py              # 向后兼容的数据加载接�?```

## 🔄 完整工作流程

### Day 0: 数据准备
- **文件**: `src/data/data_loader.py`
- **功能**: 
  - 加载自定义PCB数据�?  - 数据增强（缺陷样本�?0�?  - 图像预处理（448x448�?- **输出**: 预处理后的数据集

### Day 1-2: 模型微调
- **文件**: `src/train/pcb_train.py`
- **功能**:
  - 加载Qwen3-VL-32B-Instruct
  - 配置LoRA（rank=16�?  - 冻结视觉�?  - 训练2000�?- **输出**: LoRA检查点

### Day 3: 模型合并
- **文件**: `src/train/merge_model.py`
- **功能**:
  - 合并LoRA权重到基础模型
  - 固化JSON格式约束到config
- **输出**: 合并后的完整模型

### Day 4: 模型量化
- **文件**: `src/train/quantize_model.py`
- **功能**:
  - AWQ 4-bit量化
  - 保持视觉塔不量化
  - 校准数据准备
- **输出**: 量化模型（~25GB�?
### Day 5-6: 智能体开�?- **文件**: `src/inference/pcb_agent.py`
- **功能**:
  - LangChain智能体封�?  - 强制JSON输出
  - 置信度过�?  - 错误处理
- **输出**: 可用的智能体�?
### Day 7: 验证测试
- **文件**: `src/inference/validation_pcb.py`
- **功能**:
  - 漏检率测试（<1%�?  - 推理速度测试�?1�?张）
  - JSON格式正确率（100%�?  - 显存稳定性测�?- **输出**: 验证报告

### Day 8: 部署交付
- **文件**: `src/inference/mllm_api.py`, `deploy_pcb.sh`
- **功能**:
  - FastAPI REST服务
  - 批量处理脚本
  - 健康检�?- **输出**: 生产环境API服务

## 🎖�?性能指标

| 指标 | 目标�?| 说明 |
|------|--------|------|
| 漏检�?| < 1% | 工业红线 |
| 误报�?| < 5% | 可接受范�?|
| 推理速度 | < 1�?�?| AWQ量化�?|
| JSON格式正确�?| 100% | 强制约束 |
| 显存占用 | < 25GB | 推理�?|
| 模型大小 | ~25GB | 4-bit AWQ |

## ⚙️ 关键配置

### LoRA配置
- **rank**: 16（缺陷模式简单，低秩足够�?- **alpha**: 32
- **dropout**: 0.05
- **target_modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### 训练配置 
- **max_steps**: 2000
- **batch_size**: 1
- **gradient_accumulation_steps**: 16
- **learning_rate**: 5e-4
- **图像尺寸**: 448x448（Qwen3-VL最优）

### 推理配置
- **temperature**: 0.1（低随机性）
- **do_sample**: False（贪心解码）
- **max_new_tokens**: 512
- **confidence_threshold**: 0.7

## 🔧 使用示例

### 1. 训练模型
```bash
python src/train/pcb_train.py \
    --data_dir ./data/pcb_defects \
    --output_dir ./checkpoints/pcb_checkpoints \
    --max_steps 2000
```

### 2. 合并和量�?```bash
python src/train/merge_model.py \
    --base_model Qwen/Qwen3-VL-32B-Instruct \
    --lora_checkpoint ./checkpoints/pcb_checkpoints/final \
    --output_dir ./models/qwen3-vl-pcb

python src/train/quantize_model.py \
    --model_path ./models/qwen3-vl-pcb \
    --output_dir ./models/qwen3-vl-pcb-awq
```

### 3. 使用智能�?```python
from pcb_agent import SimplePCBAgent

agent = SimplePCBAgent(model_path="./models/qwen3-vl-pcb-awq")
defects = agent.inspect("board.jpg", inspection_type="full")
print(defects)
```

### 4. API调用
```bash
# 启动服务
python src/inference/mllm_api.py --port 8000

# 调用API
curl -X POST "http://localhost:8000/inspect" \
     -F "file=@board.jpg" \
     -F "inspection_type=full"
```

## 📊 数据格式

### 输入格式
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
  }
]
```

### 输出格式
```json
[
  {
    "defect": "short",
    "bbox": [120, 350, 45, 12],
    "confidence": 0.98,
    "repair": "清理焊锡桥接"
  }
]
```

## 🚨 注意事项

1. **显存要求**: 至少需要A100 80GB或类似规�?2. **存储空间**: 需�?00GB可用空间
3. **数据质量**: 缺陷样本需要标注准�?4. **模型选择**: 根据实际需求调整LoRA rank和训练步�?5. **量化必�?*: 为了满足<1�?张的要求，AWQ量化是必需�?
## 🔄 后续优化方向

1. **难例挖掘**: 自动识别和标注困难样�?2. **滑动窗口**: 支持大尺寸图像（>448x448�?3. **多尺度检�?*: 处理不同尺寸的缺�?4. **实时监控**: 添加推理监控和性能指标
5. **模型压缩**: 进一步压缩模型大�?
## 📝 维护建议

1. 定期备份checkpoint
2. 监控存储使用情况（使用storage_monitor.sh�?3. 记录训练日志和验证结�?4. 根据实际数据调整超参�?5. 定期更新依赖包版�?
## 🎯 成功标准

项目成功的标志：
- �?漏检�?< 1%
- �?推理速度 < 1�?�?- �?JSON格式100%正确
- �?显存稳定 < 25GB
- �?API服务正常运行

---

**最后更�?*: 2024-12-20
**版本**: 1.0.0

