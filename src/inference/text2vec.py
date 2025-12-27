"""
文本向量化模块 - 支持 DashScope、DeepSeek 和 Kimi API 进行 embedding
"""
import os
import torch
import numpy as np
import torch.nn.functional as F  
from torch import cosine_similarity
# 支持相对导入和绝对导入
try:
    from config import Config
except ImportError:
    from src.inference.config import Config
from openai import OpenAI

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class TextVector():
    def __init__(self, cfg):
        self.bert_path = getattr(cfg, 'bert_path', None)
        
        # 从配置文件读取API相关设置
        self.use_api = getattr(cfg, 'use_api', True)
        self.api_provider = getattr(cfg, 'api_provider', 'dashscope')  # 默认使用 DashScope
        self.api_key = getattr(cfg, 'api_key', "")
        self.base_url = getattr(cfg, 'base_url', "")
        self.model_name = getattr(cfg, 'model_name', "qwen2.5-vl-embedding")
        self.dimensions = getattr(cfg, 'dimensions', 1024)
        self.batch_size = getattr(cfg, 'batch_size', 10)
        
        # DashScope 特殊处理：从环境变量读取 API Key（如果配置文件中未设置）
        if self.api_provider == "dashscope":
            if not self.api_key:
                self.api_key = os.getenv('DASHSCOPE_API_KEY', '')
            # 导入 dashscope（延迟导入，避免未安装时出错）
            try:
                import dashscope
                dashscope.api_key = self.api_key
                self.dashscope = dashscope
            except ImportError:
                print("⚠️  警告：未安装 dashscope 包，请运行: pip install dashscope")
                self.dashscope = None
        
        # 只有在不使用API时才加载本地模型
        if not self.use_api and self.bert_path:
            self.load_model()

    def load_model(self):
        """载入模型（如果使用本地模型）"""
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.model = AutoModel.from_pretrained(self.bert_path)

    def mean_pooling(self, model_output, attention_mask):
        """采用序列mean-pooling获得句子的表征向量"""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_vec(self, sentences):
        """通过模型获取句子的向量"""
        if self.use_api:
            # 如果使用API，重定向到API方法
            return self.get_vec_api(sentences)
            
        # 否则使用原始BERT方法
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = sentence_embeddings.data.cpu().numpy().tolist()
        return sentence_embeddings
    
    def get_vec_api(self, query, batch_size=None):
        """通过 API（DashScope、DeepSeek 或 Kimi）获取句子的向量"""
        if batch_size is None:
            batch_size = self.batch_size
            
        # 空查询检查
        if not query:
            print("Warning: Empty query provided to get_vec_api")
            return []
        
        if isinstance(query, str):
            query = [query]
        
        # 移除空字符串和None值，确保输入数据有效
        query = [q for q in query if q and isinstance(q, str) and q.strip()]
        if not query:
            print("Warning: No valid text to vectorize after filtering")
            return []
        
        # DashScope API 特殊处理
        if self.api_provider == "dashscope":
            return self._get_vec_dashscope(query, batch_size)
        
        # DeepSeek 和 Kimi API 使用 OpenAI 兼容接口
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        all_vectors = []
        retry_count = 0
        max_retries = 2  # 允许重试几次
        
        # 获取 API 提供商（用于不同的 API 调用方式）
        api_provider = self.api_provider
        
        while retry_count <= max_retries and not all_vectors:
            try:
                for i in range(0, len(query), batch_size):
                    batch = query[i:i + batch_size]
                    try:
                        # DeepSeek 和 Kimi API 都兼容 OpenAI 格式
                        # DeepSeek 不需要 dimensions 参数，Kimi 需要
                        if api_provider == "deepseek":
                            # DeepSeek API 调用（不指定 dimensions 和 encoding_format）
                            completion = client.embeddings.create(
                                model=self.model_name,
                                input=batch
                            )
                        else:
                            # Kimi API 调用（需要指定 dimensions）
                            completion = client.embeddings.create(
                                model=self.model_name,
                                input=batch,
                                dimensions=self.dimensions,
                                encoding_format="float"
                            )
                        
                        vectors = [embedding.embedding for embedding in completion.data]
                        all_vectors.extend(vectors)
                    except Exception as e:
                        error_msg = str(e)
                        # 检查是否是 API Key 权限问题
                        if "403" in error_msg or "permission_denied" in error_msg.lower():
                            print(f"⚠️  API Key 权限错误：请检查 config.py 中的 api_key 是否正确，或 API 是否已开通")
                            print(f"   错误详情：{error_msg}")
                            # API Key 错误时不再重试
                            break
                        else:
                            print(f"向量化批次 {i//batch_size + 1} 失败：{error_msg}")
                        # 不立即返回空数组，继续处理其他批次
                        continue
                
                # 检查是否有成功获取的向量
                if all_vectors:
                    break
                else:
                    retry_count += 1
                    if retry_count <= max_retries:
                        print(f"未获取到任何向量，第 {retry_count} 次重试...")
                    else:
                        print(f"⚠️  向量化失败：已达到最大重试次数。请检查 API Key 配置。")
                    
            except Exception as outer_e:
                error_msg = str(outer_e)
                if "403" in error_msg or "permission_denied" in error_msg.lower():
                    print(f"⚠️  API Key 权限错误：请检查 config.py 中的 api_key 配置")
                    break
                print(f"向量化过程中发生错误：{error_msg}")
                retry_count += 1
                if retry_count <= max_retries:
                    print(f"第 {retry_count} 次重试...")
                
        # 返回向量数组，如果仍然为空，确保返回一个正确形状的空数组
        if not all_vectors and self.dimensions > 0:
            print("Warning: 返回一个空的向量数组，形状为 [0, dimensions]")
            return np.zeros((0, self.dimensions))
            
        return all_vectors
    
    def _get_vec_dashscope(self, query, batch_size):
        """使用 DashScope API 获取向量（支持文本、图片、视频）"""
        if self.dashscope is None:
            print("⚠️  错误：dashscope 未安装或未正确初始化")
            return []
        
        all_vectors = []
        retry_count = 0
        max_retries = 2
        
        while retry_count <= max_retries and not all_vectors:
            try:
                for i in range(0, len(query), batch_size):
                    batch = query[i:i + batch_size]
                    try:
                        # DashScope 有两种 API：
                        # 1. TextEmbedding - 专门用于文本 embedding（推荐用于纯文本）
                        # 2. MultiModalEmbedding - 用于多模态（图片、视频），也支持文本
                        # 
                        # 对于纯文本，优先尝试 TextEmbedding API
                        # 如果模型不支持 TextEmbedding（如 qwen2.5-vl-embedding），使用 MultiModalEmbedding
                        
                        resp = None
                        use_text_embedding = False
                        
                        # 首先尝试 TextEmbedding API（更适合纯文本）
                        try:
                            resp = self.dashscope.TextEmbedding.call(
                                model=self.model_name,
                                input=batch  # TextEmbedding 直接接受文本列表
                            )
                            # 检查状态码，如果非 200 则回退到 MultiModalEmbedding
                            if resp.status_code == 200:
                                use_text_embedding = True
                            else:
                                # 状态码非 200，说明 TextEmbedding 不支持该模型
                                use_text_embedding = False
                                resp = None
                        except Exception as text_e:
                            # TextEmbedding 不支持，尝试 MultiModalEmbedding
                            use_text_embedding = False
                            resp = None
                        
                        # 如果 TextEmbedding 不支持或失败，使用 MultiModalEmbedding
                        if not use_text_embedding:
                            # 注意：MultiModalEmbedding 的 input 格式要求每种类型只能出现一次
                            # 对于多个文本，需要逐个处理
                            batch_vectors = []
                            for text in batch:
                                try:
                                    single_resp = self.dashscope.MultiModalEmbedding.call(
                                        model=self.model_name,
                                        input=[{'text': text}]  # 每个文本单独处理
                                    )
                                    
                                    if single_resp.status_code == 200:
                                        vector = self._extract_dashscope_embedding(single_resp, is_text_embedding=False)
                                        if vector and len(vector) > 0:
                                            batch_vectors.append(vector[0])
                                    else:
                                        error_msg = f"MultiModalEmbedding API 错误: {single_resp.status_code}"
                                        if hasattr(single_resp, 'message'):
                                            error_msg += f" - {single_resp.message}"
                                        print(f"   处理单个文本失败: {error_msg}")
                                        # 继续处理下一个文本
                                        continue
                                except Exception as single_e:
                                    print(f"   处理单个文本失败: {str(single_e)}")
                                    # 继续处理下一个文本
                                    continue
                            
                            # 将批次结果添加到总结果
                            if batch_vectors:
                                all_vectors.extend(batch_vectors)
                            continue  # 跳过下面的响应处理
                        
                        # 处理 TextEmbedding API 的响应
                        if resp and resp.status_code == 200:
                            vectors = self._extract_dashscope_embedding(resp, is_text_embedding=True)
                            if vectors:
                                all_vectors.extend(vectors)
                            else:
                                print(f"⚠️  DashScope API 返回格式无法解析：{resp.output}")
                        elif resp:
                            error_msg = f"DashScope API 错误 (状态码: {resp.status_code})"
                            if hasattr(resp, 'message'):
                                error_msg += f": {resp.message}"
                            elif hasattr(resp, 'output') and isinstance(resp.output, dict) and 'message' in resp.output:
                                error_msg += f": {resp.output['message']}"
                            print(f"向量化批次 {i//batch_size + 1} 失败：{error_msg}")
                            if resp.status_code == 403 or resp.status_code == 401:
                                print(f"⚠️  API Key 权限错误：请检查 config.py 中的 api_key 或环境变量 DASHSCOPE_API_KEY")
                                break
                            continue
                    except Exception as e:
                        error_msg = str(e)
                        print(f"向量化批次 {i//batch_size + 1} 失败：{error_msg}")
                        if "403" in error_msg or "401" in error_msg or "permission" in error_msg.lower():
                            print(f"⚠️  API Key 权限错误：请检查 config.py 中的 api_key 或环境变量 DASHSCOPE_API_KEY")
                            break
                        continue
                
                # 检查是否有成功获取的向量
                if all_vectors:
                    break
                else:
                    retry_count += 1
                    if retry_count <= max_retries:
                        print(f"未获取到任何向量，第 {retry_count} 次重试...")
                    else:
                        print(f"⚠️  向量化失败：已达到最大重试次数。请检查 API Key 配置。")
                    
            except Exception as outer_e:
                error_msg = str(outer_e)
                if "403" in error_msg or "401" in error_msg or "permission" in error_msg.lower():
                    print(f"⚠️  API Key 权限错误：请检查 config.py 中的 api_key 或环境变量 DASHSCOPE_API_KEY")
                    break
                print(f"向量化过程中发生错误：{error_msg}")
                retry_count += 1
                if retry_count <= max_retries:
                    print(f"第 {retry_count} 次重试...")
        
        # 返回向量数组，如果仍然为空，确保返回一个正确形状的空数组
        if not all_vectors and self.dimensions > 0:
            print("Warning: 返回一个空的向量数组，形状为 [0, dimensions]")
            return np.zeros((0, self.dimensions))
        
        return all_vectors
    
    def _extract_dashscope_embedding(self, resp, is_text_embedding=False):
        """从 DashScope API 响应中提取 embedding 向量"""
        if not hasattr(resp, 'output') or not resp.output:
            return []
        
        output = resp.output
        vectors = []
        
        if is_text_embedding:
            # TextEmbedding API 的响应格式
            # 通常是一个字典，包含 'embeddings' 字段，每个元素有 'embedding' 字段
            if isinstance(output, dict):
                if 'embeddings' in output:
                    for item in output['embeddings']:
                        if isinstance(item, dict) and 'embedding' in item:
                            vectors.append(item['embedding'])
                        elif isinstance(item, (list, np.ndarray)):
                            vectors.append(item)
                elif 'output' in output:
                    # 可能是嵌套的 output
                    nested = output['output']
                    if isinstance(nested, list):
                        for item in nested:
                            if isinstance(item, dict) and 'embedding' in item:
                                vectors.append(item['embedding'])
                            elif isinstance(item, (list, np.ndarray)):
                                vectors.append(item)
            elif isinstance(output, list):
                # 直接是列表格式
                for item in output:
                    if isinstance(item, dict) and 'embedding' in item:
                        vectors.append(item['embedding'])
                    elif isinstance(item, (list, np.ndarray)):
                        vectors.append(item)
        else:
            # MultiModalEmbedding API 的响应格式
            # 根据实际响应，可能是 {'embeddings': [{'embedding': [...]}]} 格式
            if isinstance(output, dict):
                if 'embeddings' in output:
                    # 格式: {'embeddings': [{'embedding': [...]}]}
                    embeddings_list = output['embeddings']
                    if isinstance(embeddings_list, list) and len(embeddings_list) > 0:
                        # 取第一个 embedding（因为每次只处理一个输入）
                        item = embeddings_list[0]
                        if isinstance(item, dict) and 'embedding' in item:
                            vectors.append(item['embedding'])
                        elif isinstance(item, (list, np.ndarray)):
                            vectors.append(item)
                elif 'embedding' in output:
                    # 格式: {'embedding': [...]}
                    vectors.append(output['embedding'])
            elif isinstance(output, list) and len(output) > 0:
                # 列表格式，取第一个元素
                item = output[0]
                if isinstance(item, dict) and 'embedding' in item:
                    vectors.append(item['embedding'])
                elif isinstance(item, (list, np.ndarray)):
                    vectors.append(item)
        
        return vectors
    
    def get_vec_batch(self, data, bs=None):
        """batch方式获取，提高效率"""
        if bs is None:
            bs = self.batch_size
            
        if self.use_api:
            # 如果使用API，直接调用API方法
            vectors = self.get_vec_api(data, bs)
            return torch.tensor(np.array(vectors)) if len(vectors) > 0 else torch.tensor(np.array([]))
        
        # 否则使用原始BERT方法
        # 将数据分批处理（替换 pyfunctional 的 seq().grouped()）
        all_vectors = []
        for i in range(0, len(data), bs):
            batch = data[i:i + bs]
            vecs = self.get_vec(batch)
            all_vectors.extend(vecs)
        all_vectors = torch.tensor(np.array(all_vectors))
        return all_vectors

    def vector_similarity(self, vectors):
        """以[query，text1，text2...]来计算query与text1，text2,...的cosine相似度"""
        # Add dimension checking to prevent errors
        if vectors.size(0) <= 1:
            print("Warning: Not enough vectors for similarity calculation")
            return []
            
        if len(vectors.shape) < 2:
            print("Warning: Vectors must be 2-dimensional")
            return []
        
        vectors = F.normalize(vectors, p=2, dim=1)
        q_vec = vectors[0,:]
        o_vec = vectors[1:,:]
        sim = cosine_similarity(q_vec, o_vec)
        sim = sim.data.cpu().numpy().tolist()
        return sim

# 初始化全局实例（延迟初始化，避免导入时Config未准备好）
_cfg_instance = None
_tv_instance = None

def _get_text_vector_instance():
    """获取TextVector实例（单例模式）"""
    global _cfg_instance, _tv_instance
    if _tv_instance is None:
        _cfg_instance = Config()
        _tv_instance = TextVector(_cfg_instance)
    return _tv_instance

def get_vector(data, bs=None):
    """batch方式获取向量"""
    tv = _get_text_vector_instance()
    return tv.get_vec_batch(data, bs)

def get_sim(vectors):
    """计算相似度"""
    tv = _get_text_vector_instance()
    return tv.vector_similarity(vectors)

