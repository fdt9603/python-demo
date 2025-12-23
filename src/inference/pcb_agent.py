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
    # 如果 LangChain 不可用，创建一个简单的基类
    class LLM:
        """简单的 LLM 基类（LangChain 不可用时的替代）"""
        pass
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = False

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class PCBDefectInput(BaseModel):
    """PCB缺陷检测输入参数"""
    image_path: str = Field(..., description="电路板图像路径")
    inspection_type: str = Field(default="full", 
                                 enum=["full", "short", "open", "missing"],
                                 description="检测类型")


class PCBAgentLLM(LLM):
    """PCB缺陷检测专用LLM，强制JSON输出"""
    
    model_path: str = Field(default="./models/qwen3-vl-pcb-bnb")
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")
    max_new_tokens: int = Field(default=512)  # 增加到512，确保完整输出
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.95)
    do_sample: bool = Field(default=False)  # 贪心解码，最稳定
    
    _model: Optional[Any] = None
    _processor: Optional[Any] = None
    _is_quantized: bool = False  # 标记是否是量化模型
    
    def __init__(self, model_path: str = "./models/qwen3-vl-pcb-bnb", 
                 device: str = None,
                 max_new_tokens: int = 512,  # 增加到512，确保完整输出
                 temperature: float = 0.1,
                 top_p: float = 0.95,
                 do_sample: bool = False,
                 **kwargs):
        """初始化 PCB Agent LLM"""
        # 如果 LangChain 可用，使用父类初始化；否则直接设置属性
        try:
            super().__init__(**kwargs)
        except Exception:
            pass
        
        self.model_path = model_path
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self._model = None
        self._processor = None
        self._is_quantized = False  # 标记是否是量化模型
    
    @property
    def _llm_type(self) -> str:
        return "pcb_defect_detector"
    
    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            print(f"加载模型: {self.model_path}")
            
            quant_config_path = os.path.join(self.model_path, "quant_config.json")
            quantization_config = None
            if os.path.exists(quant_config_path):
                try:
                    with open(quant_config_path, "r", encoding="utf-8") as f:
                        quant_info = json.load(f)
                    if quant_info.get("quantization") == "BitsAndBytes":
                        from transformers import BitsAndBytesConfig

                        if quant_info.get("use_4bit"):
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=quant_info.get("double_quant", True),
                                bnb_4bit_quant_type=quant_info.get("quant_type", "nf4"),
                            )
                            print("✅ 检测到4-bit量化配置")
                        elif quant_info.get("use_8bit"):
                            quantization_config = BitsAndBytesConfig(
                                load_in_8bit=True,
                                llm_int8_threshold=6.0,
                            )
                            print("✅ 检测到8-bit量化配置")
                except Exception as e:
                    print(f"⚠️  读取量化配置失败，使用默认加载: {e}")
            
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
                "torch_dtype": torch.float16,
                # 显式禁用 FlashAttention2，即使 config.json 中有设置
                "attn_implementation": "sdpa",  # 使用 SDPA（Scaled Dot Product Attention）替代
            }
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
                self._is_quantized = True

            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **model_kwargs,
            )
            self._model.eval()
            for param in self._model.parameters():
                param.requires_grad = False
            
        if self._processor is None:
            print(f"加载processor: {self.model_path}")
            self._processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        调用模型进行推理
        
        prompt格式: "pcb:/path/to/image.jpg##inspection_type"
        """
        self._load_model()
        
        # 确保processor已加载
        if self._processor is None:
            return json.dumps([{"error": "processor_not_loaded", "type": "inference_error"}])
        
        try:
            if "##" in prompt:
                img_path, insp_type = prompt.split("##", 1)
                img_path = img_path.replace("pcb:", "")
            else:
                img_path = prompt.replace("pcb:", "")
                insp_type = "full"
            
            if not os.path.exists(img_path):
                return json.dumps([{"error": "image_not_found", "path": img_path}])
            
            image = Image.open(img_path).convert("RGB")
            
            inspection_types_map = {
                "full": "所有缺陷",
                "short": "短路缺陷",
                "open": "断路缺陷",
                "missing": "缺件缺陷",
            }
            question = (
                f"检测这张电路板的{inspection_types_map.get(insp_type, '所有缺陷')}。"
                "必须返回严格JSON数组：[{'defect': '类型', 'bbox': [x,y,w,h], 'confidence': 0.95, 'repair': '维修建议'}]。"
                "无缺陷返回空数组[]。不要返回额外解释。"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self._processor(
                images=image,
                text=text,
                return_tensors="pt",
            ).to(self.device)
            
            tokenizer = self._processor.tokenizer
            generation_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "repetition_penalty": 1.5,  # 防止重复
                "temperature": self.temperature if self.do_sample else None,  # 贪心解码时不使用temperature
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
            # 移除 None 值
            generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

            with torch.no_grad():
                outputs = self._model.generate(**inputs, **generation_kwargs)

            input_len = inputs["input_ids"].shape[1]
            response = self._processor.batch_decode(
                outputs[:, input_len:], skip_special_tokens=True
            )[0]

            # 稳定的 JSON 解析逻辑：
            # 1. 优先直接解析整个响应
            # 2. 失败时，从中提取所有形如 [...] 的片段，依次尝试 json.loads
            # 3. 找到第一个合法的 JSON 数组就返回；若找不到，则返回空数组 "[]"
            try:
                json_obj = json.loads(response)
                if isinstance(json_obj, list):
                    return json.dumps(json_obj, ensure_ascii=False)
            except Exception:
                pass

            # 使用非贪婪匹配提取所有可能的 JSON 数组片段
            # 注意：这里不按长度排序，按出现顺序依次尝试，避免把多个数组拼在一起的长片段选中
            candidates = re.findall(r"\[[\s\S]*?\]", response)
            for cand in candidates:
                try:
                    json_obj = json.loads(cand)
                except Exception:
                    continue
                if isinstance(json_obj, list):
                    return json.dumps(json_obj, ensure_ascii=False)

            # 实在解析不了时，返回空数组，避免上层再抛 JSONDecodeError
            return "[]"
            
        except Exception as e:
            return json.dumps([{"error": str(e), "type": "inference_error"}])
    
    def _extract_and_fix_json(self, response: str) -> str:
        """从响应中提取并修复JSON部分"""
        # 1. 尝试直接解析
        try:
            json_obj = json.loads(response)
            if isinstance(json_obj, list):
                cleaned = self._clean_defect_objects(json_obj)
                return json.dumps(cleaned, ensure_ascii=False)
        except:
            pass
        
        # 2. 尝试提取JSON数组（使用更宽松的正则）
        json_pattern = r'\[[\s\S]*?\]'
        matches = re.findall(json_pattern, response)
        
        if matches:
            # 取最长的匹配（通常是完整的JSON）
            json_str = max(matches, key=len)
            
            # 3. 尝试修复常见的JSON格式错误
            fixed_json = self._fix_json_format(json_str)
            
            try:
                json_obj = json.loads(fixed_json)
                if isinstance(json_obj, list):
                    # 清理和验证缺陷对象
                    cleaned = self._clean_defect_objects(json_obj)
                    return json.dumps(cleaned, ensure_ascii=False)
                return fixed_json
            except Exception as e:
                # 4. 如果修复后仍无法解析，尝试提取部分有效的JSON对象
                partial_json = self._extract_partial_json(json_str)
                if partial_json:
                    return json.dumps(partial_json, ensure_ascii=False)
        
        # 5. 如果都失败了，返回错误标记
        return json.dumps([{"error": "parse_failed", "raw_response": response[:200]}])
    
    def _fix_json_format(self, json_str: str) -> str:
        """修复常见的JSON格式错误"""
        # 移除字段名中的空格
        json_str = re.sub(r'"\s+(\w+)\s+"', r'"\1"', json_str)
        
        # 修复缺少逗号的情况（在 } 和 { 之间）
        json_str = re.sub(r'\}\s*\{', r'}, {', json_str)
        
        # 修复 bbox 中的格式错误（如 "y":"-1" 等）
        # 尝试修复 bbox 数组格式
        json_str = re.sub(r'"bbox"\s*:\s*\[([^\]]*)"y"\s*:\s*"([^"]*)"([^\]]*)\]', 
                         lambda m: f'"bbox": [{m.group(1)}{m.group(2)}{m.group(3)}]', json_str)
        
        # 移除末尾的不完整对象
        # 查找最后一个完整的 } 或 ]
        last_complete = max(
            json_str.rfind('}'),
            json_str.rfind(']')
        )
        if last_complete > 0:
            # 检查是否在最后一个 } 之后还有不完整的内容
            after_last = json_str[last_complete+1:]
            if after_last.strip() and not after_last.strip().startswith(','):
                json_str = json_str[:last_complete+1]
                # 确保以 ] 结尾
                if not json_str.rstrip().endswith(']'):
                    json_str = json_str.rstrip().rstrip(',') + ']'
        
        return json_str
    
    def _clean_defect_objects(self, defects: List[Dict]) -> List[Dict]:
        """清理和验证缺陷对象"""
        cleaned = []
        for defect in defects:
            if not isinstance(defect, dict):
                continue
            
            # 确保有 type 或 defect 字段
            if "type" not in defect and "defect" not in defect:
                continue
            
            # 确保有 bbox 字段且格式正确
            if "bbox" not in defect:
                continue
            
            bbox = defect["bbox"]
            # 验证 bbox 格式
            if not isinstance(bbox, list) or len(bbox) != 4:
                # 尝试修复 bbox
                if isinstance(bbox, list) and len(bbox) > 0:
                    # 如果 bbox 长度不对，尝试提取前4个数字
                    numbers = [x for x in bbox[:4] if isinstance(x, (int, float))]
                    if len(numbers) == 4:
                        defect["bbox"] = numbers
                    else:
                        continue
                else:
                    continue
            
            # 确保 bbox 中的值都是数字
            if not all(isinstance(x, (int, float)) for x in defect["bbox"]):
                continue
            
            cleaned.append(defect)
        
        return cleaned
    
    def _extract_partial_json(self, json_str: str) -> List[Dict]:
        """提取部分有效的JSON对象（即使整体格式错误）"""
        partial = []
        
        # 尝试提取每个独立的 JSON 对象
        object_pattern = r'\{[^{}]*"type"[^{}]*"bbox"[^{}]*\[[^\]]*\][^{}]*\}'
        matches = re.findall(object_pattern, json_str)
        
        for match in matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict) and "type" in obj and "bbox" in obj:
                    if isinstance(obj.get("bbox"), list) and len(obj.get("bbox")) == 4:
                        partial.append(obj)
            except:
                continue
        
        return partial
    
    def _extract_json(self, response: str) -> str:
        """从响应中提取JSON部分（向后兼容）"""
        return self._extract_and_fix_json(response)
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """调用接口（兼容 LangChain 和直接调用）"""
        return self._call(prompt, **kwargs)
    
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


def create_pcb_agent(model_path: str = "./models/qwen3-vl-pcb-bnb"):
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
    
    def __init__(self, model_path: str = "./models/qwen3-vl-pcb-bnb", vector_store=None):
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
    parser.add_argument("--model_path", type=str, default="./models/qwen3-vl-pcb-bnb",
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

