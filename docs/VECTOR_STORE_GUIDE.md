# 向量数据库和LangGraph使用指南

## 📚 概述

本项目已集成向量数据库（ChromaDB）和LangGraph工作流，提供了更强大的功能：

- **向量数据库**: 存储历史检测结果，支持相似缺陷案例检索
- **LangGraph**: 构建多步骤智能体工作流，自动化检测流程

## 🔧 安装依赖

```bash
pip install chromadb sentence-transformers langgraph
```

或者直接安装所有依赖：

```bash
pip install -r requirements.txt
```

## 🗄️ 向量数据库使用

### 1. 基础使用

```python
from vector_store import create_vector_store
from pcb_agent import SimplePCBAgent

# 创建向量存储
vector_store = create_vector_store(
    collection_name="pcb_defects",
    persist_directory="./vector_db"
)

# 创建带向量存储的智能体
agent = SimplePCBAgent(
    model_path="./models/qwen3-vl-pcb-awq",
    vector_store=vector_store
)

# 执行检测（结果自动保存到向量数据库）
defects = agent.inspect("board.jpg", inspection_type="full")
```

### 2. 搜索相似案例

```python
# 搜索相似缺陷案例
similar_cases = agent.search_similar_cases(defects, top_k=5)

for case in similar_cases:
    similarity = case['similarity']
    print(f"相似度: {similarity:.2%}")
    print(f"历史案例: {case['defects_json']}")
```

### 3. 手动管理向量数据库

```python
from vector_store import PCBVectorStore

# 创建存储实例
store = PCBVectorStore(
    collection_name="pcb_defects",
    persist_directory="./vector_db"
)

# 添加检测结果
doc_id = store.add_detection_result(
    image_path="board.jpg",
    defects=[
        {"defect": "short", "bbox": [100, 200, 50, 20], "repair": "清理焊锡"}
    ],
    metadata={"board_type": "mainboard", "batch": "20241220"}
)

# 搜索相似案例
similar = store.search_similar_defects(
    query_defects=[{"defect": "short", "bbox": [100, 200, 50, 20], "repair": "清理焊锡"}],
    top_k=5,
    min_score=0.7
)

# 获取统计信息
stats = store.get_statistics()
print(f"总案例数: {stats['total_cases']}")

# 导出数据
store.export_to_json("exported_cases.json")
```

## 🔄 LangGraph工作流使用

### 工作流步骤

LangGraph工作流包含以下步骤：

1. **检测缺陷** - 使用MLLM检测图像中的缺陷
2. **检索相似案例** - 从向量数据库检索相似历史案例
3. **生成维修报告** - 基于检测结果和相似案例生成详细报告
4. **质量评估** - 计算质量分数
5. **存储结果** - 将结果保存到向量数据库

### 使用示例

```python
from pcb_graph import PCBLangGraphAgent

# 创建LangGraph智能体
agent = PCBLangGraphAgent(
    model_path="./models/qwen3-vl-pcb-awq",
    collection_name="pcb_defects"
)

# 执行完整工作流
result = agent.inspect(
    image_path="board.jpg",
    inspection_type="full",
    use_graph=True  # 使用LangGraph工作流
)

# 查看结果
print(f"缺陷数量: {len(result['defects'])}")
print(f"相似案例: {len(result['similar_cases'])}")
print(f"质量分数: {result['quality_score']:.2f}")
print(f"维修报告:\n{result['repair_report']}")
```

### 工作流状态

工作流状态包含以下字段：

```python
class PCBInspectionState:
    image_path: str              # 图像路径
    inspection_type: str         # 检测类型
    defects: List[Dict]          # 检测到的缺陷列表
    similar_cases: List[Dict]    # 相似案例列表
    repair_report: str           # 生成的维修报告
    quality_score: float         # 质量分数 (0.0-1.0)
    metadata: Dict               # 元数据
```

## 📊 使用场景

### 场景1: 历史案例库构建

```python
# 批量处理历史图像，构建案例库
vector_store = create_vector_store()
agent = SimplePCBAgent(vector_store=vector_store)

for image_path in historical_images:
    defects = agent.inspect(image_path)
    # 结果自动保存到向量数据库
```

### 场景2: 智能维修建议

```python
# 检测新图像，自动检索相似案例提供维修建议
agent = PCBLangGraphAgent()

result = agent.inspect("new_board.jpg")
# result['repair_report'] 包含基于历史案例的维修建议
```

### 场景3: 质量趋势分析

```python
# 导出所有案例进行分析
vector_store = create_vector_store()
vector_store.export_to_json("all_cases.json")

# 可以分析：
# - 常见缺陷类型
# - 质量分数趋势
# - 维修方案效果
```

## 🔍 向量数据库架构

### 数据存储格式

每个检测结果存储为：

```json
{
  "id": "doc_20241220_143022_123456",
  "text": "缺陷类型: short, 位置: [100, 200, 50, 20], 维修建议: 清理焊锡桥接",
  "embedding": [0.123, 0.456, ...],  // 向量嵌入
  "metadata": {
    "image_path": "board.jpg",
    "defect_count": 1,
    "timestamp": "2024-12-20T14:30:22",
    "defects_json": "[{\"defect\": \"short\", ...}]"
  }
}
```

### 相似度计算

使用余弦相似度计算缺陷案例之间的相似性：

- 相似度范围: 0.0 - 1.0
- 默认最小相似度: 0.7
- 嵌入模型: `paraphrase-multilingual-MiniLM-L12-v2`

## 🚀 性能优化

### 1. 批量导入

```python
# 批量添加可以提高性能
for image_path in image_list:
    defects = agent.inspect(image_path)
    # 向量存储会自动批量处理
```

### 2. 持久化存储

```python
# 使用持久化存储，避免重复构建
vector_store = create_vector_store(
    persist_directory="./vector_db"  # 持久化到磁盘
)
```

### 3. 相似度阈值调整

```python
# 根据需求调整相似度阈值
similar_cases = store.search_similar_defects(
    query_defects=defects,
    top_k=10,
    min_score=0.6  # 降低阈值获取更多结果
)
```

## ⚠️ 注意事项

1. **内存使用**: 向量数据库会占用一定内存，大量数据建议使用持久化存储
2. **嵌入模型**: 首次使用会自动下载嵌入模型（~100MB）
3. **ChromaDB**: 如果ChromaDB不可用，系统会使用内存存储作为后备
4. **LangGraph**: 如果LangGraph未安装，系统会自动使用简化版工作流

## 📝 完整示例

参考 `examples/example_usage.py` 查看完整的使用示例，包括：

- 基础检测
- 向量数据库集成
- LangGraph工作流
- 批量处理
- 高级检索

