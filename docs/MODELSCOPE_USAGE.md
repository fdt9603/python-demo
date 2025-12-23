# ModelScope 使用指南

## 问题说明

如果服务器无法访问 HuggingFace（网络连接问题），可以使用 ModelScope 来加载模型。

## 安装 ModelScope

```bash
pip install modelscope
```

## 使用方法

### 训练时使用 ModelScope

**重要：必须添加 `--use_modelscope` 参数**

```bash
python src/train/pcb_train.py \
    --data_dir tools/data/pcb_defects \
    --output_dir ./checkpoints/pcb_checkpoints \
    --max_steps 2000 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-4 \
    --use_modelscope  # ⚠️ 必须添加这个参数
```

### 如果不添加 `--use_modelscope` 参数

代码会尝试从 HuggingFace 加载，如果网络不通会失败。

## 工作原理

1. **ModelScope 下载**：使用 `snapshot_download` 从 ModelScope 下载模型到本地
2. **本地加载**：下载完成后，使用本地路径加载模型
3. **禁用在线检查**：设置环境变量禁用 HuggingFace 的在线检查

## 模型路径映射

代码会自动将 HuggingFace 格式转换为 ModelScope 格式：

- `Qwen/Qwen3-VL-32B-Instruct` → `qwen/Qwen3-VL-32B-Instruct`

## 缓存目录

ModelScope 下载的模型默认保存在：
- `./modelscope_cache/`（当前目录）
- 或通过环境变量 `MODELSCOPE_CACHE` 自定义

## 首次下载

首次使用 ModelScope 下载模型需要一些时间（约 60GB），下载完成后会缓存在本地，后续使用会直接加载本地缓存。

## 验证是否使用 ModelScope

运行训练命令时，如果看到以下输出，说明正在使用 ModelScope：

```
🔄 使用ModelScope下载模型: qwen/Qwen3-VL-32B-Instruct
   这将避免网络连接问题...
✅ 模型已下载到本地: ./modelscope_cache/qwen/Qwen3-VL-32B-Instruct
   现在将使用本地路径加载，不会访问 HuggingFace
📁 检测到本地模型路径，禁用 HuggingFace 在线检查
```

## 常见问题

### Q: 为什么还是尝试连接 HuggingFace？

A: 请确保：
1. ✅ 添加了 `--use_modelscope` 参数
2. ✅ 已安装 `modelscope`：`pip install modelscope`
3. ✅ 检查命令中参数位置正确

### Q: ModelScope 下载失败怎么办？

A: 
1. 检查网络连接：`ping www.modelscope.cn`
2. 检查 ModelScope 是否安装：`python -c "import modelscope; print(modelscope.__version__)"`
3. 如果下载失败，代码会自动回退到 HuggingFace（如果网络允许）

### Q: 如何查看下载进度？

A: ModelScope 会自动显示下载进度，包括文件大小和下载速度。

### Q: 下载的模型在哪里？

A: 默认在 `./modelscope_cache/` 目录，可以通过环境变量自定义：
```bash
export MODELSCOPE_CACHE=/path/to/custom/cache
python src/train/pcb_train.py --use_modelscope ...
```

## 完整示例

```bash
# 1. 安装 ModelScope
pip install modelscope

# 2. 使用 ModelScope 训练
python src/train/pcb_train.py \
    --data_dir tools/data/pcb_defects \
    --output_dir ./checkpoints/pcb_checkpoints \
    --max_steps 2000 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-4 \
    --use_modelscope  # 关键参数
```

## 注意事项

1. **首次下载**：模型约 60GB，需要足够的存储空间和下载时间
2. **缓存管理**：下载的模型会缓存在本地，不会重复下载
3. **网络要求**：需要能够访问 ModelScope（国内网络通常没问题）

