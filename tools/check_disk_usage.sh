#!/bin/bash
# 检查磁盘使用情况，找出占用空间最大的目录和文件

echo "=========================================="
echo "磁盘使用情况分析"
echo "=========================================="
echo ""

# 1. 总体磁盘使用情况
echo "1. 总体磁盘使用情况："
df -h . | tail -1
echo ""

# 2. 检查各个目录的大小
echo "2. 各目录大小（前20个最大的）："
du -h --max-depth=1 . 2>/dev/null | sort -h | tail -20
echo ""

# 3. 检查模型相关目录
echo "3. 模型相关目录大小："
if [ -d "./models" ]; then
    echo "  ./models:"
    du -sh ./models/* 2>/dev/null | sort -h | tail -10
fi
if [ -d "./modelscope_cache" ]; then
    echo "  ./modelscope_cache:"
    du -sh ./modelscope_cache/* 2>/dev/null | sort -h | tail -10
fi
if [ -d "$HOME/.cache/modelscope" ]; then
    echo "  ~/.cache/modelscope:"
    du -sh "$HOME/.cache/modelscope"/* 2>/dev/null | sort -h | tail -10
fi
if [ -d "$HOME/.cache/huggingface" ]; then
    echo "  ~/.cache/huggingface:"
    du -sh "$HOME/.cache/huggingface"/* 2>/dev/null | sort -h | tail -10
fi
echo ""

# 4. 检查checkpoints目录
echo "4. Checkpoints目录大小："
if [ -d "./checkpoints" ]; then
    du -sh ./checkpoints/* 2>/dev/null | sort -h | tail -10
fi
echo ""

# 5. 检查最大的单个文件
echo "5. 最大的单个文件（前20个）："
find . -type f -size +100M 2>/dev/null | xargs du -h 2>/dev/null | sort -h | tail -20
echo ""

# 6. 统计各类型文件大小
echo "6. 各类型文件统计："
echo "  .safetensors 文件:"
find . -name "*.safetensors" -type f -exec du -ch {} + 2>/dev/null | tail -1
echo "  .bin 文件:"
find . -name "*.bin" -type f -exec du -ch {} + 2>/dev/null | tail -1
echo "  .pt 文件:"
find . -name "*.pt" -type f -exec du -ch {} + 2>/dev/null | tail -1
echo "  checkpoint 目录:"
find . -type d -name "checkpoint-*" -exec du -sh {} + 2>/dev/null | sort -h | tail -5
echo ""

echo "=========================================="
echo "分析完成"
echo "=========================================="

