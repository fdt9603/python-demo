"""
测试所有导入是否正确
"""
import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 60)
print("测试导入")
print("=" * 60)

# 测试 config
try:
    from config import Config
    print("✅ config 导入成功")
    cfg = Config()
    print(f"   - kb_base_dir: {cfg.kb_base_dir}")
    print(f"   - output_dir: {cfg.output_dir}")
    print(f"   - llm_model_path: {cfg.llm_model_path}")
except Exception as e:
    print(f"❌ config 导入失败: {e}")
    try:
        from src.inference.config import Config
        print("✅ 使用绝对导入成功")
    except Exception as e2:
        print(f"❌ 绝对导入也失败: {e2}")

# 测试 local_llm_client
try:
    from local_llm_client import LocalLLMClient
    print("✅ local_llm_client 导入成功")
except Exception as e:
    print(f"❌ local_llm_client 导入失败: {e}")
    try:
        from src.inference.local_llm_client import LocalLLMClient
        print("✅ 使用绝对导入成功")
    except Exception as e2:
        print(f"❌ 绝对导入也失败: {e2}")

# 测试 text2vec
try:
    from text2vec import get_vector, get_sim
    print("✅ text2vec 导入成功")
except Exception as e:
    print(f"❌ text2vec 导入失败: {e}")
    try:
        from src.inference.text2vec import get_vector, get_sim
        print("✅ 使用绝对导入成功")
    except Exception as e2:
        print(f"❌ 绝对导入也失败: {e2}")

# 测试 retrievor
try:
    from retrievor import q_searching
    print("✅ retrievor 导入成功")
except Exception as e:
    print(f"❌ retrievor 导入失败: {e}")
    try:
        from src.inference.retrievor import q_searching
        print("✅ 使用绝对导入成功")
    except Exception as e2:
        print(f"❌ 绝对导入也失败: {e2}")

# 测试 rag 模块中的关键函数
try:
    from rag import get_knowledge_bases, vectorize_query
    print("✅ rag 模块导入成功")
except Exception as e:
    print(f"❌ rag 模块导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

