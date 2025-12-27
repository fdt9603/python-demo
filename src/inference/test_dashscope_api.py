"""
测试 DashScope API 和 qwen2.5-vl-embedding 模型
"""
import os
import json
import sys

# 添加路径以便导入 config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import dashscope
    from config import Config
except ImportError as e:
    print(f"❌ 导入错误：{e}")
    print("请确保已安装 dashscope: pip install dashscope")
    sys.exit(1)

def test_dashscope_embedding():
    """测试 DashScope MultiModalEmbedding API"""
    print("=" * 60)
    print("测试 DashScope qwen2.5-vl-embedding API")
    print("=" * 60)
    
    # 读取配置
    cfg = Config()
    
    # 获取 API Key
    api_key = cfg.api_key or os.getenv('DASHSCOPE_API_KEY', '')
    if not api_key:
        print("❌ 错误：未找到 DashScope API Key")
        print("   请在 config.py 中设置 api_key，或设置环境变量 DASHSCOPE_API_KEY")
        return False
    
    # 设置 API Key
    dashscope.api_key = api_key
    
    # 测试文本 embedding
    print("\n1. 测试文本 embedding...")
    test_texts = [
        "这是一个测试文本",
        "PCB缺陷检测",
        "工业质检系统"
    ]
    
    try:
        # 首先尝试 TextEmbedding API（更适合纯文本）
        print("   尝试使用 TextEmbedding API...")
        resp = None
        use_text_embedding = False
        
        try:
            resp = dashscope.TextEmbedding.call(
                model=cfg.model_name,
                input=test_texts  # TextEmbedding 直接接受文本列表
            )
            # 检查状态码，如果非 200 则回退到 MultiModalEmbedding
            if resp.status_code == 200:
                use_text_embedding = True
            else:
                # 状态码非 200，说明 TextEmbedding 不支持该模型
                error_msg = resp.message if hasattr(resp, 'message') else f'状态码 {resp.status_code}'
                print(f"   TextEmbedding API 返回错误: {error_msg}")
                use_text_embedding = False
        except Exception as text_e:
            print(f"   TextEmbedding API 不支持: {text_e}")
            use_text_embedding = False
        
        # 如果 TextEmbedding 不支持或失败，使用 MultiModalEmbedding API
        if not use_text_embedding:
            print("   尝试使用 MultiModalEmbedding API...")
            # 使用 MultiModalEmbedding API（需要逐个处理）
            all_responses = []
            for idx, text in enumerate(test_texts, 1):
                try:
                    single_resp = dashscope.MultiModalEmbedding.call(
                        model=cfg.model_name,
                        input=[{'text': text}]  # 每个文本单独处理
                    )
                    all_responses.append(single_resp)
                    if single_resp.status_code != 200:
                        error_msg = single_resp.message if hasattr(single_resp, 'message') else f'状态码 {single_resp.status_code}'
                        print(f"   ⚠️  文本 {idx} 处理失败: {error_msg}")
                except Exception as e:
                    print(f"   ⚠️  文本 {idx} 处理异常: {e}")
                    all_responses.append(None)
            
            # 检查所有响应
            successful_responses = [r for r in all_responses if r and r.status_code == 200]
            if successful_responses:
                print(f"   ✅ MultiModalEmbedding API 调用成功 ({len(successful_responses)}/{len(test_texts)})")
                # 处理所有响应
                vectors = []
                for single_resp in successful_responses:
                    if hasattr(single_resp, 'output') and single_resp.output:
                        output = single_resp.output
                        # 检查响应格式：可能是 {'embeddings': [{'embedding': [...]}]} 或 {'embedding': [...]}
                        if isinstance(output, dict):
                            if 'embeddings' in output:
                                # 格式: {'embeddings': [{'embedding': [...]}]}
                                embeddings_list = output['embeddings']
                                if isinstance(embeddings_list, list) and len(embeddings_list) > 0:
                                    item = embeddings_list[0]
                                    if isinstance(item, dict) and 'embedding' in item:
                                        vectors.append(item['embedding'])
                                    elif isinstance(item, (list, np.ndarray)):
                                        vectors.append(item)
                            elif 'embedding' in output:
                                # 格式: {'embedding': [...]}
                                vectors.append(output['embedding'])
                        elif isinstance(output, list) and len(output) > 0:
                            # 格式: [{'embedding': [...]}] 或 [[...]]
                            item = output[0]
                            if isinstance(item, dict) and 'embedding' in item:
                                vectors.append(item['embedding'])
                            elif isinstance(item, (list, np.ndarray)):
                                vectors.append(item)
                
                if vectors:
                    print(f"   ✅ 成功提取 {len(vectors)} 个 embedding 向量")
                    print(f"   向量维度: {len(vectors[0]) if vectors else 0}")
                    return True
                else:
                    print("   ⚠️  无法从响应中提取 embedding 向量")
                    if successful_responses:
                        print(f"   响应示例: {json.dumps(successful_responses[0].output, indent=2, ensure_ascii=False)[:300]}")
                    return False
            else:
                resp = all_responses[0] if all_responses and all_responses[0] else None
                if resp:
                    error_msg = resp.message if hasattr(resp, 'message') else f'状态码 {resp.status_code}'
                    print(f"   ❌ MultiModalEmbedding API 调用失败: {error_msg}")
                else:
                    print("   ❌ 无法调用 API")
                return False
        
        # 处理 TextEmbedding API 的响应
        if use_text_embedding and resp:
            print(f"   状态码: {resp.status_code}")
            
            if resp.status_code == 200:
                print("   ✅ TextEmbedding API 调用成功")
                print(f"   响应格式: {type(resp.output)}")
                
                # 打印响应结构（用于调试）
                print("\n   响应内容预览:")
                print(json.dumps(resp.output, indent=2, ensure_ascii=False)[:500])
                
                # 尝试提取 embedding
                vectors = []
                output = resp.output
                
                if isinstance(output, dict):
                    if 'embeddings' in output:
                        for item in output['embeddings']:
                            if isinstance(item, dict) and 'embedding' in item:
                                vectors.append(item['embedding'])
                                print(f"   ✅ 成功提取 embedding，维度: {len(item['embedding'])}")
                    elif 'output' in output:
                        nested = output['output']
                        if isinstance(nested, list):
                            for item in nested:
                                if isinstance(item, dict) and 'embedding' in item:
                                    vectors.append(item['embedding'])
                                    print(f"   ✅ 成功提取 embedding，维度: {len(item['embedding'])}")
                elif isinstance(output, list):
                    for item in output:
                        if isinstance(item, dict) and 'embedding' in item:
                            vectors.append(item['embedding'])
                            print(f"   ✅ 成功提取 embedding，维度: {len(item['embedding'])}")
                
                if vectors:
                    print(f"\n   ✅ 成功获取 {len(vectors)} 个 embedding 向量")
                    return True
                else:
                    print("   ⚠️  无法从响应中提取 embedding 向量")
                    print(f"   完整响应: {json.dumps(resp.output, indent=2, ensure_ascii=False)}")
                    return False
            else:
                print(f"   ❌ TextEmbedding API 调用失败")
                error_msg = resp.message if hasattr(resp, 'message') else f'状态码 {resp.status_code}'
                print(f"   错误信息: {error_msg}")
                
                if resp.status_code == 403 or resp.status_code == 401:
                    print("\n   ⚠️  API Key 权限错误")
                    print("   请检查：")
                    print("   1. API Key 是否正确")
                    print("   2. 是否已开通 DashScope 服务")
                    print("   3. 是否已开通 TextEmbedding 服务权限")
                
                return False
            
    except Exception as e:
        print(f"   ❌ 发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text2vec_integration():
    """测试 text2vec.py 中的 DashScope 集成"""
    print("\n" + "=" * 60)
    print("测试 text2vec.py 集成")
    print("=" * 60)
    
    try:
        from text2vec import TextVector
        
        cfg = Config()
        tv = TextVector(cfg)
        
        test_texts = ["测试文本1", "测试文本2"]
        print(f"\n测试文本: {test_texts}")
        
        vectors = tv.get_vec_api(test_texts)
        
        if vectors and len(vectors) > 0:
            print(f"✅ 成功获取 {len(vectors)} 个向量")
            print(f"   向量维度: {len(vectors[0]) if vectors else 0}")
            return True
        else:
            print("❌ 未能获取向量")
            return False
            
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import numpy as np
    
    # 测试 1: 直接调用 DashScope API
    success1 = test_dashscope_embedding()
    
    # 测试 2: 测试 text2vec 集成
    if success1:
        success2 = test_text2vec_integration()
    else:
        print("\n⚠️  跳过集成测试（API 调用失败）")
        success2 = False
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ 所有测试通过！")
    elif success1:
        print("⚠️  API 调用成功，但集成测试失败")
    else:
        print("❌ 测试失败，请检查配置")
    print("=" * 60)

