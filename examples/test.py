# 简化测试代码
import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-ccPg9NQUefwNguNEO56BMLnk4ShO8eBtLLLO1sGfirb0Ey0I",  # 确保是完整的有效密钥
    base_url="https://api.moonshot.cn/v1"
)

# 先测试普通聊天接口
try:
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[{"role": "user", "content": "你好"}]
    )
    print("聊天API正常")
except Exception as e:
    print(f"密钥无效: {e}")
    