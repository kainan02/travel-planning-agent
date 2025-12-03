"""测试DeepSeek API连接"""
from openai import OpenAI
import os

# 使用你的API密钥
api_key = ''

print("正在测试DeepSeek API连接...")
print(f"API密钥: {api_key[:10]}...")

try:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    
    print("\n正在发送测试请求...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": "你好，请回复'API连接成功'"
            }
        ],
        max_tokens=50,
        timeout=30
    )
    
    result = response.choices[0].message.content
    print(f"\n✅ API连接成功！")
    print(f"回复: {result}")
    
except Exception as e:
    print(f"\n❌ API连接失败！")
    print(f"错误信息: {str(e)}")
    print(f"错误类型: {type(e).__name__}")
    import traceback
    print("\n详细错误:")
    traceback.print_exc()

