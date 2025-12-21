#!/bin/bash
# PCB存储守护脚本（训练时最大占用20GB）
# 自动清理旧的checkpoint，防止存储溢出

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/pcb_checkpoints}"
MAX_SIZE_GB="${MAX_SIZE_GB:-20}"
MAX_SIZE_KB=$((MAX_SIZE_GB * 1024 * 1024))

echo "PCB存储监控启动..."
echo "监控目录: $CHECKPOINT_DIR"
echo "最大容量: ${MAX_SIZE_GB}GB"

while true; do
    if [ -d "$CHECKPOINT_DIR" ]; then
        USED_KB=$(du -sk "$CHECKPOINT_DIR" 2>/dev/null | awk '{print $1}')
        USED_GB=$(echo "scale=2; $USED_KB / 1024 / 1024" | bc)
        
        if [ -n "$USED_KB" ] && [ "$USED_KB" -gt "$MAX_SIZE_KB" ]; then
            echo "🚨 存储超限: ${USED_GB}GB > ${MAX_SIZE_GB}GB，开始清理..."
            
            # 删除最旧的checkpoint（保留最新的3个）
            find "$CHECKPOINT_DIR" -name "checkpoint-*" -type d | sort | head -n -3 | xargs rm -rf 2>/dev/null
            
            # 如果还是太大，删除所有非final的checkpoint
            NEW_USED_KB=$(du -sk "$CHECKPOINT_DIR" 2>/dev/null | awk '{print $1}')
            if [ -n "$NEW_USED_KB" ] && [ "$NEW_USED_KB" -gt "$MAX_SIZE_KB" ]; then
                echo "🚨 存储仍超限，清理所有checkpoint（保留final）..."
                find "$CHECKPOINT_DIR" -name "checkpoint-*" -type d | xargs rm -rf 2>/dev/null
            fi
            
            NEW_USED_GB=$(du -sk "$CHECKPOINT_DIR" 2>/dev/null | awk '{print $1}')
            NEW_USED_GB=$(echo "scale=2; $NEW_USED_KB / 1024 / 1024" | bc)
            echo "✅ 存储清理完成，当前: ${NEW_USED_GB}GB"
        else
            echo "✅ 存储正常: ${USED_GB}GB / ${MAX_SIZE_GB}GB"
        fi
    fi
    
    sleep 60  # 每分钟检查一次
done

