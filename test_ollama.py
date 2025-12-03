"""
测试 Ollama 连接的脚本
运行此脚本来检查 Ollama 是否正常运行
"""
import requests
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')

print("=" * 60)
print("Ollama 连接测试")
print("=" * 60)
print(f"Ollama URL: {OLLAMA_BASE_URL}")
print(f"模型名称: {OLLAMA_MODEL}")
print()

# 测试1: 检查 Ollama 服务是否运行
print("1. 检查 Ollama 服务是否运行...")
try:
    health_url = f"{OLLAMA_BASE_URL}/api/tags"
    print(f"   请求 URL: {health_url}")
    response = requests.get(health_url, timeout=5)
    response.raise_for_status()
    print("   [OK] Ollama service is running")
except requests.exceptions.ConnectionError:
    print("   [ERROR] Cannot connect to Ollama service")
    print()
    print("   解决方法:")
    print("   1. 确保 Ollama 已安装")
    print("   2. 启动 Ollama:")
    print("      - Windows: 运行 Ollama 应用程序")
    print("      - 命令行: ollama serve")
    print("   3. 检查 URL 是否正确:", OLLAMA_BASE_URL)
    exit(1)
except Exception as e:
    print(f"   [ERROR] {e}")
    exit(1)

# 测试2: 检查已安装的模型
print()
print("2. 检查已安装的模型...")
try:
    models_data = response.json()
    models = models_data.get('models', [])
    if models:
        print(f"   已安装的模型 ({len(models)} 个):")
        for model in models:
            model_name = model.get('name', 'Unknown')
            print(f"     - {model_name}")
    else:
        print("   [WARNING] No models installed")
        print()
        print("   解决方法:")
        print(f"   运行: ollama pull {OLLAMA_MODEL}")
        exit(1)
except Exception as e:
    print(f"   [ERROR] Failed to parse model list: {e}")
    exit(1)

# 测试3: 检查请求的模型是否存在
print()
print(f"3. 检查模型 '{OLLAMA_MODEL}' 是否可用...")
available_models = [m.get('name', '') for m in models]
model_base = OLLAMA_MODEL.split(':')[0]

# 检查完整名称或基础名称
model_found = False
for model_name in available_models:
    if model_name == OLLAMA_MODEL or model_name.startswith(model_base + ':'):
        model_found = True
        print(f"   [OK] Found model: {model_name}")
        break

if not model_found:
    print(f"   [ERROR] Model '{OLLAMA_MODEL}' not found")
    print()
    print("   解决方法:")
    print(f"   运行: ollama pull {OLLAMA_MODEL}")
    print()
    print("   或者使用已安装的模型，在 .env 文件中设置:")
    if available_models:
        print(f"   OLLAMA_MODEL={available_models[0]}")
    exit(1)

# 测试4: 测试简单的 API 调用
print()
print("4. 测试 API 调用...")
try:
    test_url = f"{OLLAMA_BASE_URL}/api/chat"
    test_payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": "Say 'Hello' in one word."}
        ],
        "stream": False
    }
    print(f"   发送测试请求到: {test_url}")
    test_response = requests.post(test_url, json=test_payload, timeout=30)
    
    if test_response.status_code != 200:
        print(f"   [ERROR] API returned status code: {test_response.status_code}")
        try:
            error_detail = test_response.json()
            print(f"   Error details: {error_detail}")
        except:
            print(f"   Error text: {test_response.text[:200]}")
        exit(1)
    
    test_response.raise_for_status()
    test_result = test_response.json()
    
    if 'message' in test_result and 'content' in test_result['message']:
        content = test_result['message']['content']
        print(f"   [OK] API call successful")
        print(f"   响应: {content[:50]}...")
    else:
        print(f"   [ERROR] API returned unexpected format: {test_result}")
        exit(1)
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        print(f"   [ERROR] API endpoint not found (404)")
        print("   这可能是因为:")
        print("   1. Ollama 版本过旧")
        print("   2. API 端点路径不正确")
        print("   请尝试更新 Ollama 到最新版本")
    else:
        print(f"   [ERROR] HTTP error: {e.response.status_code}")
    exit(1)
except Exception as e:
    print(f"   [ERROR] Test failed: {e}")
    exit(1)

print()
print("=" * 60)
print("[SUCCESS] All tests passed! Ollama is configured correctly")
print("=" * 60)
print()
print("现在可以在应用中使用本地 LLM 模式了。")

