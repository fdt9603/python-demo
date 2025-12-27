"""
本地LLM客户端 - 使用训练好的 qwen3-VL 模型进行文本生成
"""
import torch
import json
import os
from typing import Optional
from PIL import Image
import numpy as np
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
# 支持相对导入和绝对导入
try:
    from config import Config
except ImportError:
    from src.inference.config import Config


class LocalLLMClient:
    """使用本地训练好的 qwen3-VL 模型进行文本生成"""
    
    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or Config.llm_model_path
        self.device = device or (Config.llm_device if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        self._is_quantized = False
        
    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            print(f"加载模型: {self.model_path}")
            
            quant_config_path = os.path.join(self.model_path, "quant_config.json")
            quantization_config = None
            if os.path.exists(quant_config_path):
                try:
                    # 使用安全的 JSON 读取方式
                    try:
                        with open(quant_config_path, "r", encoding="utf-8") as f:
                            quant_info = json.load(f)
                    except UnicodeDecodeError:
                        # 如果 UTF-8 失败，尝试其他编码
                        with open(quant_config_path, "rb") as f:
                            content = f.read()
                        for encoding in ['utf-8', 'gbk', 'gb18030', 'latin-1']:
                            try:
                                quant_info = json.loads(content.decode(encoding, errors='ignore'))
                                break
                            except (UnicodeDecodeError, json.JSONDecodeError):
                                continue
                        else:
                            quant_info = json.loads(content.decode('utf-8', errors='ignore'))
                    if quant_info.get("quantization") == "BitsAndBytes":
                        from transformers import BitsAndBytesConfig
                        if quant_info.get("use_4bit"):
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=quant_info.get("bnb_4bit_use_double_quant", True),
                                bnb_4bit_quant_type=quant_info.get("bnb_4bit_quant_type", "nf4"),
                            )
                            print("✅ 检测到4-bit量化配置")
                except Exception as e:
                    print(f"⚠️  读取量化配置失败，使用默认加载: {e}")
            
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
                "dtype": torch.float16,  # 使用 dtype 替代已弃用的 torch_dtype
                "attn_implementation": "sdpa",
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
    
    def generate_answer(self, system_prompt: str, user_prompt: str, image: Optional[Image.Image] = None, **kwargs) -> str:
        """
        生成回答
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词（包含背景知识）
            image: 可选的图像输入（PIL Image对象）
            **kwargs: 其他生成参数
        
        Returns:
            生成的文本回答
        """
        self._load_model()
        
        try:
            # 构建消息 - Qwen3VL 的消息格式（参考训练代码）
            if image is not None:
                # 有图像时，构建消息（不包含 system，因为训练代码中也没有）
                # 参考训练代码：只构建 user 和 assistant 消息
                user_msg = {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # 只包含类型占位符
                        {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt}
                    ]
                }
                
                # 构建消息列表（只包含 user 消息，用于生成）
                messages = [user_msg]
                
                # 应用聊天模板（参考训练代码的方式）
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # 处理输入（图像单独传递，参考训练代码）
                inputs = self._processor(
                    images=image,
                    text=text,
                    return_tensors="pt",
                ).to(self.device)
            else:
                # 无图像时，使用纯文本格式
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
                # 应用聊天模板
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                # 处理输入（不包含图像）
                inputs = self._processor(
                    text=text,
                    return_tensors="pt",
                ).to(self.device)
            
            # 生成参数
            tokenizer = self._processor.tokenizer
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", Config.llm_max_new_tokens),
                "do_sample": kwargs.get("do_sample", Config.llm_do_sample),
                "temperature": kwargs.get("temperature", Config.llm_temperature) if kwargs.get("do_sample", Config.llm_do_sample) else None,
                "top_p": kwargs.get("top_p", Config.llm_top_p) if kwargs.get("do_sample", Config.llm_do_sample) else None,
                "repetition_penalty": kwargs.get("repetition_penalty", 1.2),
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
            # 移除 None 值
            generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
            
            # 生成
            with torch.no_grad():
                outputs = self._model.generate(**inputs, **generation_kwargs)
            
            # 解码
            input_len = inputs["input_ids"].shape[1]
            response = self._processor.batch_decode(
                outputs[:, input_len:], skip_special_tokens=True
            )[0]
            
            return response.strip()
            
        except Exception as e:
            print(f"生成回答时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"生成回答时出错：{str(e)}"

