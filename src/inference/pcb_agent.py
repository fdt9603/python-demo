"""
Day 5-6: PCB专用LangChain智能体
核心：强制JSON输出，避免幻觉
支持向量数据库和LangGraph工作流
"""
import json
import os
import re
import torch
from typing import Optional, Dict, Any, List
from PIL import Image

# LangChain导入（可选）
try:
    from langchain.llms.base import LLM
    from langchain.pydantic_v1 import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = False

from transformers import AutoModelForVision2Seq, AutoProcessor


class PCBDefectInput(BaseModel):
    """PCB缺陷检测输入参数"""
    image_path: str = Field(..., description="电路板图像路径")
    inspection_type: str = Field(default="full", 
                                 enum=["full", "short", "open", "missing"],
                                 description="检测类型")


class PCBAgentLLM(LLM):
    """PCB缺陷检测专用LLM，强制JSON输出"""
    
    model_path: str = Field(default="./models/qwen3-vl-pcb-awq")
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")
    max_new_tokens: int = Field(default=512)
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.95)
    do_sample: bool = Field(default=False)  # 贪心解码，最稳定
    
    _model: Optional[Any] = None
    _processor: Optional[Any] = None
    
    @property
    def _llm_type(self) -> str:
        return "pcb_defect_detector"
    
    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            print(f"加载模型: {self.model_path}")
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16
            )
            self._model.eval()
            
            self._processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        调用模型进行推理
        
        prompt格式: "pcb:/path/to/image.jpg##inspection_type"
        """
        self._load_model()
        
        try:
            # 解析prompt
            if "##" in prompt:
                img_path, insp_type = prompt.split("##", 1)
                img_path = img_path.replace("pcb:", "")
            else:
                img_path = prompt.replace("pcb:", "")
                insp_type = "full"
            
            # 加载图像
            if not os.path.exists(img_path):
                return json.dumps([{"error": "image_not_found", "path": img_path}])
            
            image = Image.open(img_path).convert('RGB')
            
            # 调整图像大小
            if image.size != (448, 448):
                image = image.resize((448, 448), Image.Resampling.LANCZOS)
            
            # 构造带格式约束的prompt
            inspection_types_map = {
                "full": "所有缺陷",
                "short": "短路缺陷",
                "open": "断路缺陷",
                "missing": "缺件缺陷"
            }
            
            constrained_prompt = f"""检测这张电路板的{inspection_types_map.get(insp_type, '所有缺陷')}。
必须返回严格JSON格式：[{{"defect": "类型", "bbox": [x,y,w,h], "confidence": 0.95, "repair": "维修建议"}}]
无缺陷返回空数组[]。
不要返回任何额外解释，只返回JSON数组。"""
            
            # 处理输入
            inputs = self._processor(
                images=image,
                text=constrained_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # 生成输出
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self._processor.tokenizer.pad_token_id,
                )
            
            # 解码输出
            response = self._processor.decode(outputs[0], skip_special_tokens=True)
            
            # 后处理：强制JSON解析
            return self._extract_json(response)
            
        except Exception as e:
            return json.dumps([{"error": str(e), "type": "inference_error"}])
    
    def _extract_json(self, response: str) -> str:
        """从响应中提取JSON部分"""
        try:
            # 尝试直接解析
            json_obj = json.loads(response)
            if isinstance(json_obj, list):
                return json.dumps(json_obj, ensure_ascii=False)
        except:
            pass
        
        # 尝试提取JSON数组
        json_pattern = r'\[[\s\S]*?\]'
        matches = re.findall(json_pattern, response)
        
        if matches:
            # 取最长的匹配（通常是完整的JSON）
            json_str = max(matches, key=len)
            try:
                json_obj = json.loads(json_str)
                # 置信度过滤
                if isinstance(json_obj, list):
                    filtered = [d for d in json_obj if d.get("confidence", 1.0) > 0.7]
                    return json.dumps(filtered, ensure_ascii=False)
                return json_str
            except:
                pass
        
        # 如果都失败了，返回错误标记
        return json.dumps([{"error": "parse_failed", "raw_response": response[:200]}])
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "temperature": self.temperature,
        }


def pcb_inspection_tool(input_data: PCBDefectInput) -> str:
    """
    电路板缺陷检测工具
    
    Args:
        input_data: PCB缺陷检测输入参数
    
    Returns:
        JSON格式的缺陷列表
    """
    llm = PCBAgentLLM()
    prompt = f"pcb:{input_data.image_path}##{input_data.inspection_type}"
    return llm(prompt)


def create_pcb_agent(model_path: str = "./models/qwen3-vl-pcb-awq"):
    """
    创建PCB质检智能体
    
    Args:
        model_path: 模型路径
    
    Returns:
        LangChain智能体
    """
    try:
        from langchain.agents import initialize_agent, Tool
        from langchain.agents.agent_types import AgentType
    except ImportError:
        print("警告: LangChain未安装，使用简化版本")
        return SimplePCBAgent(model_path)
    
    tools = [
        Tool(
            name="pcb_full_inspection",
            func=lambda x: pcb_inspection_tool(PCBDefectInput.parse_raw(x) if isinstance(x, str) else x),
            description="全检：检测短路、断路、缺件所有缺陷，返回JSON格式的缺陷列表",
            args_schema=PCBDefectInput
        )
    ]
    
    llm = PCBAgentLLM(model_path=model_path)
    
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=1,  # 一次推理，不循环
        handle_parsing_errors=True
    )
    
    return agent


class SimplePCBAgent:
    """简化版PCB智能体（不依赖LangChain）"""
    
    def __init__(self, model_path: str = "./models/qwen3-vl-pcb-awq", vector_store=None):
        """
        初始化简化版智能体
        
        Args:
            model_path: 模型路径
            vector_store: 向量存储实例（可选）
        """
        self.llm = PCBAgentLLM(model_path=model_path)
        self.vector_store = vector_store
    
    def run(self, input_data: Dict[str, Any]) -> str:
        """运行检测"""
        if isinstance(input_data, dict):
            input_obj = PCBDefectInput(**input_data)
        else:
            input_obj = input_data
        
        prompt = f"pcb:{input_obj.image_path}##{input_obj.inspection_type}"
        return self.llm(prompt)
    
    def inspect(self, image_path: str, inspection_type: str = "full") -> List[Dict[str, Any]]:
        """检测接口（返回Python对象）"""
        result = self.run({"image_path": image_path, "inspection_type": inspection_type})
        defects = json.loads(result)
        
        # 如果配置了向量存储，自动保存结果
        if self.vector_store is not None:
            try:
                self.vector_store.add_detection_result(
                    image_path=image_path,
                    defects=defects
                )
            except Exception as e:
                print(f"保存检测结果到向量数据库失败: {e}")
        
        return defects
    
    def search_similar_cases(
        self,
        defects: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        搜索相似案例
        
        Args:
            defects: 缺陷列表
            top_k: 返回最相似的k个结果
        
        Returns:
            相似案例列表
        """
        if self.vector_store is None:
            print("警告: 向量存储未配置，无法搜索相似案例")
            return []
        
        return self.vector_store.search_similar_defects(defects, top_k=top_k)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PCB缺陷检测智能体")
    parser.add_argument("--image_path", type=str, required=True, help="电路板图像路径")
    parser.add_argument("--inspection_type", type=str, default="full",
                       choices=["full", "short", "open", "missing"],
                       help="检测类型")
    parser.add_argument("--model_path", type=str, default="./models/qwen3-vl-pcb-awq",
                       help="模型路径")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"错误: 图像文件不存在: {args.image_path}")
        exit(1)
    
    # 创建智能体
    agent = SimplePCBAgent(model_path=args.model_path)
    
    # 执行检测
    print(f"检测图像: {args.image_path}")
    print(f"检测类型: {args.inspection_type}")
    
    defects = agent.inspect(args.image_path, args.inspection_type)
    
    print("\n检测结果:")
    print(json.dumps(defects, ensure_ascii=False, indent=2))

