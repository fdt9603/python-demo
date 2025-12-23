#!/bin/bash
# Day 8: PCB质检系统部署脚本

set -e

echo "=========================================="
echo "PCB缺陷检测系统部署"
echo "=========================================="

# 配置
MODEL_PATH="${MODEL_PATH:-./models/qwen3-vl-pcb-bnb}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
INPUT_DIR="${INPUT_DIR:-./data/pcb_input}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/pcb_output}"
LOG_DIR="${LOG_DIR:-./logs}"

# 创建目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 检查模型
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型不存在: $MODEL_PATH"
    echo "请先完成模型训练和量化"
    exit 1
fi

echo "✅ 模型路径: $MODEL_PATH"

# 启动API服务（后台）
echo "启动API服务..."
python src/inference/mllm_api.py \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --model_path "$MODEL_PATH" \
    --workers 1 \
    > "$LOG_DIR/api.log" 2>&1 &

API_PID=$!
echo "API服务已启动 (PID: $API_PID)"
echo "  URL: http://$API_HOST:$API_PORT"
echo "  文档: http://$API_HOST:$API_PORT/docs"

# 等待API启动
sleep 5

# 健康检查
echo "检查API健康状态..."
if curl -f "http://$API_HOST:$API_PORT/health" > /dev/null 2>&1; then
    echo "✅ API服务运行正常"
else
    echo "❌ API服务启动失败，请检查日志: $LOG_DIR/api.log"
    kill $API_PID 2>/dev/null || true
    exit 1
fi

# 批量处理脚本
if [ -d "$INPUT_DIR" ] && [ "$(ls -A $INPUT_DIR 2>/dev/null)" ]; then
    echo ""
    echo "开始批量处理图像..."
    echo "  输入目录: $INPUT_DIR"
    echo "  输出目录: $OUTPUT_DIR"
    
    python -c "
import os
import json
import glob
import requests
from pathlib import Path

input_dir = '$INPUT_DIR'
output_dir = '$OUTPUT_DIR'
api_url = 'http://$API_HOST:$API_PORT'

os.makedirs(output_dir, exist_ok=True)

# 支持的图像格式
image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_files = []
for ext in image_exts:
    image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

print(f'找到 {len(image_files)} 张图像')

for i, img_path in enumerate(image_files):
    try:
        print(f'处理 [{i+1}/{len(image_files)}]: {os.path.basename(img_path)}')
        
        # 调用API
        with open(img_path, 'rb') as f:
            files = {'file': f}
            data = {'inspection_type': 'full'}
            response = requests.post(f'{api_url}/inspect', files=files, data=data)
            result = response.json()
        
        # 保存结果
        output_file = os.path.join(output_dir, Path(img_path).stem + '_report.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f'  ✅ 结果已保存: {output_file}')
        if result.get('defects'):
            print(f'  发现 {len(result[\"defects\"])} 个缺陷')
    
    except Exception as e:
        print(f'  ❌ 处理失败: {e}')

print(f'\n✅ 批量处理完成！结果保存在: {output_dir}')
" 2>&1 | tee "$LOG_DIR/batch_process.log"
else
    echo "⚠️  输入目录为空或不存在，跳过批量处理"
    echo "  设置 INPUT_DIR 环境变量指定输入目录"
fi

# 保存PID
echo "$API_PID" > "$LOG_DIR/api.pid"

echo ""
echo "=========================================="
echo "部署完成！"
echo "=========================================="
echo "API服务: http://$API_HOST:$API_PORT"
echo "API文档: http://$API_HOST:$API_PORT/docs"
echo "日志目录: $LOG_DIR"
echo ""
echo "停止服务: kill \$(cat $LOG_DIR/api.pid)"
echo "查看日志: tail -f $LOG_DIR/api.log"
echo "=========================================="

