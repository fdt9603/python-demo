#!/bin/bash
# 清理所有缓存和回收站内容，释放存储空间

echo "=========================================="
echo "开始清理存储空间"
echo "=========================================="
echo ""

# 1. 清理Python缓存
echo "1. 清理Python缓存文件..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type f -name "*.pyd" -delete 2>/dev/null
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null
echo "✅ Python缓存已清理"
echo ""

# 2. 清理pip缓存
echo "2. 清理pip缓存..."
pip cache purge 2>/dev/null || echo "⚠️  pip缓存清理失败（可能已清理）"
echo "✅ pip缓存已清理"
echo ""

# 3. 清理HuggingFace缓存
echo "3. 清理HuggingFace缓存..."
if [ -d "$HOME/.cache/huggingface" ]; then
    du -sh "$HOME/.cache/huggingface" 2>/dev/null
    rm -rf "$HOME/.cache/huggingface"/* 2>/dev/null
    echo "✅ HuggingFace缓存已清理"
else
    echo "ℹ️  HuggingFace缓存目录不存在"
fi
echo ""

# 4. 清理ModelScope缓存
echo "4. 清理ModelScope缓存..."
if [ -d "./modelscope_cache" ]; then
    du -sh "./modelscope_cache" 2>/dev/null
    rm -rf "./modelscope_cache"/* 2>/dev/null
    echo "✅ ModelScope缓存已清理"
fi
if [ -d "$HOME/.cache/modelscope" ]; then
    du -sh "$HOME/.cache/modelscope" 2>/dev/null
    rm -rf "$HOME/.cache/modelscope"/* 2>/dev/null
    echo "✅ ModelScope用户缓存已清理"
fi
echo ""

# 5. 清理系统回收站（Linux）
echo "5. 清理系统回收站..."
# 清理用户回收站
if [ -d "$HOME/.local/share/Trash" ]; then
    rm -rf "$HOME/.local/share/Trash"/* 2>/dev/null
    rm -rf "$HOME/.local/share/Trash"/.[^.]* 2>/dev/null
    echo "✅ 用户回收站已清理"
fi
# 清理root回收站
if [ -d "/root/.local/share/Trash" ]; then
    rm -rf "/root/.local/share/Trash"/* 2>/dev/null
    rm -rf "/root/.local/share/Trash"/.[^.]* 2>/dev/null
    echo "✅ root回收站已清理"
fi
# 清理JupyterLab回收站（如果存在）
if [ -d "$HOME/.jupyter/lab/workspaces" ]; then
    find "$HOME/.jupyter" -name "*trash*" -type d -exec rm -rf {} + 2>/dev/null
    echo "✅ JupyterLab回收站已清理"
fi
echo ""

# 6. 清理临时文件
echo "6. 清理临时文件..."
# 清理系统临时文件
rm -rf /tmp/* 2>/dev/null
rm -rf /var/tmp/* 2>/dev/null
# 清理用户临时文件
rm -rf "$HOME/tmp"/* 2>/dev/null
rm -rf "$HOME/temp"/* 2>/dev/null
# 清理项目临时文件
find . -type f -name "*.tmp" -delete 2>/dev/null
find . -type f -name "*.temp" -delete 2>/dev/null
find . -type f -name "*.log" -size +100M -delete 2>/dev/null  # 只删除大于100MB的日志
echo "✅ 临时文件已清理"
echo ""

# 7. 清理Docker缓存（如果安装了Docker）
echo "7. 清理Docker缓存..."
if command -v docker &> /dev/null; then
    docker system prune -af --volumes 2>/dev/null || echo "⚠️  Docker清理需要权限"
    echo "✅ Docker缓存已清理"
else
    echo "ℹ️  Docker未安装，跳过"
fi
echo ""

# 8. 清理conda缓存（如果使用conda）
echo "8. 清理conda缓存..."
if command -v conda &> /dev/null; then
    conda clean --all -y 2>/dev/null || echo "⚠️  conda清理失败"
    echo "✅ conda缓存已清理"
else
    echo "ℹ️  conda未安装，跳过"
fi
echo ""

# 9. 清理大型日志文件
echo "9. 清理大型日志文件..."
find . -type f -name "*.log" -size +50M -ls -delete 2>/dev/null
find ./logs -type f -name "*.log" -size +10M -delete 2>/dev/null 2>/dev/null
echo "✅ 大型日志文件已清理"
echo ""

# 10. 清理checkpoints（可选，谨慎使用）
echo "10. 检查checkpoints目录..."
if [ -d "./checkpoints" ]; then
    du -sh "./checkpoints" 2>/dev/null
    echo "⚠️  发现checkpoints目录，如需删除请手动执行: rm -rf ./checkpoints"
fi
echo ""

# 11. 清理模型文件中的临时文件（保留模型本身）
echo "11. 清理模型目录中的临时文件..."
if [ -d "./models" ]; then
    find ./models -name "*.tmp" -delete 2>/dev/null
    find ./models -name "*.lock" -delete 2>/dev/null
    find ./models -name "*.safetensors.index.json.tmp" -delete 2>/dev/null
    echo "✅ 模型临时文件已清理"
fi
echo ""

echo "=========================================="
echo "清理完成！"
echo "=========================================="
echo ""
echo "当前磁盘使用情况："
df -h . | tail -1
echo ""
echo "如需查看详细目录大小，运行："
echo "  du -sh * | sort -h"

