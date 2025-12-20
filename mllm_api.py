"""
Day 8: MLLM推理API服务
使用FastAPI提供RESTful API接口
"""
import os
import json
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from PIL import Image
import io
from pcb_agent import SimplePCBAgent

app = FastAPI(
    title="PCB缺陷检测API",
    description="基于Qwen3-VL的电路板缺陷检测服务",
    version="1.0.0"
)

# 全局智能体实例（延迟加载）
_agent: Optional[SimplePCBAgent] = None


def get_agent() -> SimplePCBAgent:
    """获取智能体实例（单例模式）"""
    global _agent
    if _agent is None:
        model_path = os.getenv("MODEL_PATH", "./models/qwen3-vl-pcb-awq")
        _agent = SimplePCBAgent(model_path=model_path)
    return _agent


class InspectionRequest(BaseModel):
    """检测请求"""
    inspection_type: str = Field(
        default="full",
        enum=["full", "short", "open", "missing"],
        description="检测类型"
    )


class DefectItem(BaseModel):
    """缺陷项"""
    defect: str = Field(..., description="缺陷类型")
    bbox: List[int] = Field(..., description="边界框 [x, y, w, h]")
    confidence: Optional[float] = Field(None, description="置信度")
    repair: str = Field(..., description="维修建议")


class InspectionResponse(BaseModel):
    """检测响应"""
    success: bool = Field(..., description="是否成功")
    defects: List[DefectItem] = Field(default_factory=list, description="缺陷列表")
    error: Optional[str] = Field(None, description="错误信息")


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "PCB缺陷检测API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        agent = get_agent()
        return {"status": "healthy", "model_loaded": agent.llm._model is not None}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/inspect", response_model=InspectionResponse)
async def inspect_image(
    file: UploadFile = File(..., description="电路板图像"),
    inspection_type: str = "full"
):
    """
    检测电路板缺陷
    
    Args:
        file: 上传的图像文件
        inspection_type: 检测类型（full/short/open/missing）
    
    Returns:
        InspectionResponse: 检测结果
    """
    try:
        # 验证文件类型
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="文件必须是图像格式")
        
        # 读取图像
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 保存临时文件（智能体需要文件路径）
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name, "JPEG")
            tmp_path = tmp_file.name
        
        try:
            # 执行检测
            agent = get_agent()
            result = agent.run({
                "image_path": tmp_path,
                "inspection_type": inspection_type
            })
            
            # 解析结果
            defects_data = json.loads(result)
            
            # 过滤错误项
            defects = [
                DefectItem(**item) for item in defects_data
                if not item.get("error")
            ]
            
            return InspectionResponse(
                success=True,
                defects=defects
            )
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except json.JSONDecodeError as e:
        return InspectionResponse(
            success=False,
            error=f"JSON解析错误: {str(e)}"
        )
    except Exception as e:
        return InspectionResponse(
            success=False,
            error=f"检测失败: {str(e)}"
        )


@app.post("/inspect/batch")
async def inspect_batch(
    files: List[UploadFile] = File(..., description="电路板图像列表"),
    inspection_type: str = "full"
):
    """
    批量检测电路板缺陷
    
    Args:
        files: 上传的图像文件列表
        inspection_type: 检测类型
    
    Returns:
        List[InspectionResponse]: 检测结果列表
    """
    results = []
    
    for file in files:
        try:
            result = await inspect_image(file, inspection_type)
            results.append(result.dict())
        except Exception as e:
            results.append({
                "success": False,
                "defects": [],
                "error": str(e)
            })
    
    return {"results": results}


@app.post("/inspect/path")
async def inspect_by_path(
    request: Dict[str, Any]
):
    """
    通过路径检测（用于内部调用）
    
    Args:
        request: {"image_path": "...", "inspection_type": "..."}
    
    Returns:
        InspectionResponse: 检测结果
    """
    try:
        image_path = request.get("image_path")
        inspection_type = request.get("inspection_type", "full")
        
        if not image_path or not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="图像文件不存在")
        
        agent = get_agent()
        result = agent.run({
            "image_path": image_path,
            "inspection_type": inspection_type
        })
        
        defects_data = json.loads(result)
        defects = [
            DefectItem(**item).dict() for item in defects_data
            if not item.get("error")
        ]
        
        return InspectionResponse(
            success=True,
            defects=defects
        )
    
    except Exception as e:
        return InspectionResponse(
            success=False,
            error=str(e)
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="启动PCB缺陷检测API服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--model_path", type=str, default="./models/qwen3-vl-pcb-awq",
                       help="模型路径")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["MODEL_PATH"] = args.model_path
    
    # 启动服务
    uvicorn.run(
        "mllm_api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )

