# PCB RAG 推理系统使用说明

## 快速启动

### 方式1：从项目根目录运行
```bash
python src/inference/rag.py
```

### 方式2：进入 inference 目录运行
```bash
cd src/inference
python rag.py
```

## 依赖安装

### 依赖冲突已修复

已移除 `pyfunctional` 依赖（与 `dill` 版本冲突），改用标准 Python 代码实现。

如果安装时出现依赖冲突，请按以下步骤操作：

```bash
# 1. 先升级 dill 到兼容版本
pip install "dill>=0.4.0,<0.4.1" --upgrade

# 2. 如果已安装旧版本的 pyfunctional，可以卸载（可选）
pip uninstall pyfunctional -y

# 3. 安装 RAG 系统依赖
pip install -r src/inference/rag_requirements.txt
```

## 文件说明

- `rag.py` - 主推理文件，包含 Gradio UI 界面
- `config.py` - 配置文件（需要设置 Kimi API Key）
- `local_llm_client.py` - 本地 LLM 客户端（使用训练好的模型）
- `text2vec.py` - 文本向量化模块（使用 Kimi API）
- `retrievor.py` - 联网搜索模块
- `knowledge_base/` - 知识库文件夹（用于上传 PCB 缺陷检测指南文档）
- `output_files/` - 临时输出目录

## 配置说明

编辑 `config.py` 设置：
1. **Kimi API Key**: 用于文本向量化
2. **模型路径**: 默认 `./models/qwen3-vl-pcb-bnb`（训练好的模型）
3. **知识库路径**: 默认 `src/inference/knowledge_bases`
4. **输出目录**: 默认 `src/inference/output_files`

## 注意事项

1. 确保训练好的模型位于 `./models/qwen3-vl-pcb-bnb` 目录
2. 需要有效的 Kimi API Key
3. 建议至少 8GB 显存（使用量化模型）
4. 知识库文档建议上传与 PCB 缺陷检测相关的文档

