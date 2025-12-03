from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import json
import warnings
from dotenv import load_dotenv
from openai import OpenAI
import requests
import logging
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
from ics import Calendar, Event
from io import BytesIO

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 抑制 multiprocessing 资源跟踪器的警告（FAISS 相关）
# 使用更全面的过滤规则
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')
warnings.filterwarnings('ignore', message='.*resource_tracker.*')
warnings.filterwarnings('ignore', message='.*leaked.*semaphore.*')
warnings.filterwarnings('ignore', message='.*There appear to be.*leaked.*objects.*')
from rag_utils import (
    manual_rag_inference,          # <--- 必须有这个
    create_ollama_llm,
    create_documents_from_text,
    create_vectorstore_from_documents,
    load_destination_knowledge_base,
    save_knowledge_base,
    call_local_llm as utils_call_local_llm # 引入工具库里的简单调用函数
)

# 加载环境变量
try:
    load_dotenv()
except Exception as e:
    print(f"警告: 加载.env文件时出错: {e}，将使用代码中的默认值")

app = Flask(__name__)
# 配置CORS，允许所有来源的请求（包括不同IP地址）
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# 配置日志记录到文件
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)
logger.info("=" * 50)
logger.info("应用启动")
logger.info("=" * 50)

# LLM模式配置：'cloud' 或 'local'
LLM_MODE = os.getenv('LLM_MODE', 'cloud').lower()  # 默认使用云端

# 初始化DeepSeek客户端（兼容OpenAI SDK）- 仅云端模式使用
client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY', ''),
    base_url="https://api.deepseek.com"
)

# 本地LLM配置（Ollama）
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')  # 默认模型

# Google Custom Search API 配置
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
GOOGLE_SEARCH_ENGINE_ID = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '5299e07176b844ae6')

# 启动时打印配置信息
logger.info(f"LLM模式: {LLM_MODE}")
logger.info(f"Google API配置: API_KEY={GOOGLE_API_KEY[:10]}..., SEARCH_ENGINE_ID={GOOGLE_SEARCH_ENGINE_ID}")
if LLM_MODE == 'local':
    logger.info(f"本地LLM配置: URL={OLLAMA_BASE_URL}, MODEL={OLLAMA_MODEL}")
print(f"LLM模式: {LLM_MODE}")
print(f"Google API配置: API_KEY={GOOGLE_API_KEY[:10]}..., SEARCH_ENGINE_ID={GOOGLE_SEARCH_ENGINE_ID}")
if LLM_MODE == 'local':
    print(f"本地LLM配置: URL={OLLAMA_BASE_URL}, MODEL={OLLAMA_MODEL}")

def google_search(query, num_results=5):
    """
    使用Google Custom Search API进行搜索
    返回搜索结果列表
    """
    if not GOOGLE_API_KEY:
        logger.warning("警告: 未配置Google API密钥，跳过搜索")
        return []
    
    # 如果没有搜索引擎ID，尝试使用默认的
    if not GOOGLE_SEARCH_ENGINE_ID:
        logger.warning("警告: 未配置Google搜索引擎ID，尝试使用API密钥直接搜索")
        # 注意：Google Custom Search API 需要搜索引擎ID，如果没有则无法搜索
        return []
    
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': GOOGLE_API_KEY,
            'cx': GOOGLE_SEARCH_ENGINE_ID,
            'q': query,
            'num': min(num_results, 10)  # Google API最多返回10个结果
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        if 'items' in data:
            for item in data['items']:
                results.append({
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'link': item.get('link', '')
                })
        
        logger.info(f"Google搜索成功，找到 {len(results)} 个结果")
        print(f"Google搜索成功，找到 {len(results)} 个结果")
        return results
    except Exception as e:
        logger.error(f"Google搜索出错: {str(e)}")
        print(f"Google搜索出错: {str(e)}")
        return []

def simple_search(query, num_results=5):
    """
    简化版搜索：直接返回Google搜索链接（不需要API）
    这是一个备用方案，当没有配置API时使用
    """
    # 生成Google搜索链接
    search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
    
    # 返回一个包含搜索链接的结果
    # 注意：这只是一个链接，不是实际的搜索结果
    return [{
        'title': f'在Google中搜索: {query}',
        'snippet': '点击下方链接在Google中查看搜索结果（需要手动访问）',
        'link': search_url,
        'is_link_only': True
    }]

def extract_webpage_content(url, max_length=2000):
    """
    从网页URL提取主要内容
    返回网页的文本内容
    """
    try:
        # 使用更现代的 User-Agent 和更多请求头来避免被阻止
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none'
        }
        
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        
        # 检查状态码
        if response.status_code == 403:
            logger.warning(f"网页访问被阻止 (403): {url} - 可能是反爬虫机制")
            return None
        elif response.status_code == 404:
            logger.warning(f"网页不存在 (404): {url}")
            return None
        
        response.raise_for_status()
        response.encoding = response.apparent_encoding or 'utf-8'
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # 移除脚本和样式标签
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # 提取主要内容
        # 优先提取article、main、content等标签
        content = None
        for tag in ['article', 'main', '[role="main"]', '.content', '.post', '.entry-content']:
            elements = soup.select(tag)
            if elements:
                content = elements[0]
                break
        
        # 如果没有找到特定标签，使用body
        if not content:
            content = soup.find('body') or soup
        
        # 提取文本
        text = content.get_text(separator='\n', strip=True)
        
        # 清理文本：移除多余空白
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # 限制长度
        if len(text) > max_length:
            text = text[:max_length] + '...'
        
        return text
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.warning(f"网页访问被阻止 (403): {url} - 可能是反爬虫机制")
        else:
            logger.warning(f"HTTP错误 {e.response.status_code} 提取网页内容失败 {url}: {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"网络错误提取网页内容失败 {url}: {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"提取网页内容失败 {url}: {str(e)}")
        return None

def search_destination_info(destination, days, preferences=''):
    """
    搜索目的地的相关信息，并提取网页内容
    返回包含网页内容的搜索结果
    """
    search_queries = [
        f"{destination} {days}天 旅游攻略 景点推荐",
        f"{destination} 美食推荐 餐厅",
        f"{destination} 住宿推荐 酒店"
    ]
    
    if preferences:
        search_queries.append(f"{destination} {preferences}")
    
    all_results = []
    for query in search_queries:
        results = google_search(query, num_results=3)
        all_results.extend(results)
    
    # 去重（基于链接）
    seen_links = set()
    unique_results = []
    for result in all_results:
        if result['link'] not in seen_links:
            seen_links.add(result['link'])
            unique_results.append(result)
    
    if not unique_results:
        logger.warning("没有找到搜索结果")
        return []
    
    # 提取网页内容（最多5个，避免太慢）
    logger.info(f"开始提取 {len(unique_results[:5])} 个网页的内容...")
    print(f"开始提取 {len(unique_results[:5])} 个网页的内容...")
    enriched_results = []
    for i, result in enumerate(unique_results[:5], 1):
        logger.info(f"正在提取网页 {i}/{min(5, len(unique_results))}: {result['link']}")
        print(f"正在提取网页 {i}/{min(5, len(unique_results))}: {result['link']}")
        content = extract_webpage_content(result['link'], max_length=1500)
        if content:
            result['content'] = content
            logger.info(f"成功提取网页内容，长度: {len(content)} 字符")
            print(f"成功提取网页内容，长度: {len(content)} 字符")
            enriched_results.append(result)
        else:
            logger.warning(f"提取网页内容失败，使用摘要: {result.get('snippet', '无摘要')[:100]}")
            print(f"提取网页内容失败，使用摘要")
            # 即使提取失败，也保留搜索结果（至少有用摘要）
            enriched_results.append(result)
    
    logger.info(f"成功提取 {len(enriched_results)} 个网页的内容")
    print(f"成功提取 {len(enriched_results)} 个网页的内容")
    return enriched_results

def check_ollama_connection():
    """
    检查Ollama连接和模型是否可用
    """
    try:
        # 检查Ollama服务是否运行
        health_url = f"{OLLAMA_BASE_URL}/api/tags"
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
        
        # 检查模型是否存在
        models_data = response.json()
        available_models = [model.get('name', '').split(':')[0] for model in models_data.get('models', [])]
        
        # 检查请求的模型是否可用（支持带版本号和不带版本号）
        model_base = OLLAMA_MODEL.split(':')[0]
        if model_base not in available_models and OLLAMA_MODEL not in [m.get('name', '') for m in models_data.get('models', [])]:
            available_models_str = ', '.join(available_models) if available_models else 'None'
            raise Exception(f"模型 '{OLLAMA_MODEL}' 未安装。已安装的模型: {available_models_str}。请运行: ollama pull {OLLAMA_MODEL}")
        
        return True
    except requests.exceptions.ConnectionError:
        raise Exception(f"无法连接到Ollama服务 ({OLLAMA_BASE_URL})。请确保Ollama正在运行。启动方法: 运行 'ollama serve' 或启动Ollama应用程序。")
    except requests.exceptions.Timeout:
        raise Exception(f"连接Ollama服务超时 ({OLLAMA_BASE_URL})。请检查Ollama是否正常运行。")
    except requests.exceptions.RequestException as e:
        raise Exception(f"检查Ollama连接时出错: {str(e)}。请确保Ollama正在运行。")

def call_local_llm(prompt, system_prompt="You are a helpful assistant."):
    """
    调用本地LLM (Ollama)
    """
    try:
        # 首先检查连接和模型
        logger.info(f"检查Ollama连接和模型: {OLLAMA_MODEL}")
        check_ollama_connection()
        
        logger.info(f"调用本地LLM: {OLLAMA_MODEL}")
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "options": {
                "temperature": 0.5,  # 降低temperature加快速度
                "num_predict": 2000,  # 限制最大输出token数，加快速度
                "top_p": 0.9,  # 优化采样参数
            },
            "stream": False
        }
        logger.info(f"发送请求到: {url}")
        response = requests.post(url, json=payload, timeout=300)  # 5分钟超时
        
        if response.status_code == 404:
            raise Exception(f"API端点不存在 (404)。请检查Ollama版本。尝试访问: {OLLAMA_BASE_URL}/api/tags 查看是否可访问。")
        
        response.raise_for_status()
        result = response.json()
        
        if 'message' not in result or 'content' not in result.get('message', {}):
            logger.error(f"Ollama返回格式异常: {result}")
            raise Exception("Ollama返回的数据格式不正确")
        
        content = result.get('message', {}).get('content', '')
        if not content:
            raise Exception("Ollama返回的内容为空")
        
        return content
    except requests.exceptions.ConnectionError as e:
        error_msg = f"无法连接到Ollama服务 ({OLLAMA_BASE_URL})。请确保Ollama正在运行。\n解决方法:\n1. 启动Ollama: 运行 'ollama serve' 或启动Ollama应用程序\n2. 检查URL是否正确: {OLLAMA_BASE_URL}\n3. 检查防火墙设置"
        logger.error(f"本地LLM连接失败: {error_msg}")
        raise Exception(error_msg)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            error_msg = f"模型 '{OLLAMA_MODEL}' 未找到 (404)。请运行: ollama pull {OLLAMA_MODEL}"
            logger.error(f"本地LLM调用失败: {error_msg}")
            raise Exception(error_msg)
        else:
            error_msg = f"HTTP错误 ({e.response.status_code}): {str(e)}"
            logger.error(f"本地LLM调用失败: {error_msg}")
            raise Exception(error_msg)
    except requests.exceptions.Timeout:
        error_msg = "请求超时（超过5分钟）。模型可能正在处理，请稍后重试或使用更小的提示词。"
        logger.error(f"本地LLM调用超时: {error_msg}")
        raise Exception(error_msg)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"本地LLM调用失败: {error_msg}")
        raise Exception(f"本地LLM调用失败: {error_msg}")

def parse_plan_to_ics(plan_text, destination, start_date=None):
    """
    解析行程文本并生成.ics日历文件
    """
    try:
        # 如果没有指定开始日期，使用明天
        if not start_date:
            start_date = datetime.now() + timedelta(days=1)
        else:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # 创建日历
        calendar = Calendar()
        
        # 解析每日行程
        lines = plan_text.split('\n')
        current_day = None
        current_events = []
        
        for line in lines:
            line = line.strip()
            # 匹配 "Day X:" 或 "### Day X:"
            day_match = re.search(r'Day\s+(\d+):', line, re.IGNORECASE)
            if day_match:
                # 保存之前一天的事件
                if current_day is not None and current_events:
                    day_date = start_date + timedelta(days=current_day - 1)
                    event_text = '\n'.join(current_events)
                    event = Event()
                    event.name = f"Day {current_day}: {destination}"
                    event.begin = day_date.replace(hour=9, minute=0)  # 默认早上9点
                    event.end = day_date.replace(hour=18, minute=0)  # 默认晚上6点
                    event.description = event_text[:1000]  # 限制描述长度
                    calendar.events.add(event)
                
                current_day = int(day_match.group(1))
                current_events = []
            elif current_day and line and not line.startswith('#'):
                # 收集当天的活动信息
                if line.startswith('**') or line.startswith('-') or line.startswith('*'):
                    current_events.append(line)
        
        # 保存最后一天的事件
        if current_day is not None and current_events:
            day_date = start_date + timedelta(days=current_day - 1)
            event_text = '\n'.join(current_events)
            event = Event()
            event.name = f"Day {current_day}: {destination}"
            event.begin = day_date.replace(hour=9, minute=0)
            event.end = day_date.replace(hour=18, minute=0)
            event.description = event_text[:1000]
            calendar.events.add(event)
        
        # 如果没有解析到任何事件，创建一个默认事件
        if len(calendar.events) == 0:
            event = Event()
            event.name = f"Travel Plan: {destination}"
            event.begin = start_date.replace(hour=9, minute=0)
            event.end = start_date.replace(hour=18, minute=0)
            event.description = plan_text[:1000]
            calendar.events.add(event)
        
        return calendar
    except Exception as e:
        logger.error(f"解析行程到ICS失败: {str(e)}")
        raise

@app.before_request
def handle_preflight():
    """处理CORS预检请求"""
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.after_request
def after_request(response):
    """在所有响应中添加CORS头"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def index():
    """返回主页面"""
    return render_template('index.html')

@app.route('/api/search', methods=['POST', 'OPTIONS'])
def search():
    """搜索API端点：搜索、提取内容、AI总结"""
    # 处理CORS预检请求
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        logger.info(f"收到搜索请求: {request.method}, Content-Type: {request.content_type}")
        
        if not request.is_json:
            logger.warning("请求不是JSON格式")
            return jsonify({
                'success': False,
                'error': '请求格式错误，需要JSON格式'
            }), 400
        
        data = request.json
        if not data:
            logger.warning("请求数据为空")
            return jsonify({
                'success': False,
                'error': '请求数据为空'
            }), 400
            
        query = data.get('query', '')
        logger.info(f"搜索关键词: {query}")
        
        if not query:
            return jsonify({
                'success': False,
                'error': '请输入搜索关键词'
            }), 400
        
        # 搜索相关网页
        has_api = bool(GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID)
        logger.info(f"搜索请求: query={query}, has_api={has_api}")
        
        if not has_api:
            return jsonify({
                'success': False,
                'error': '未配置Google API，无法进行搜索'
            }), 400
        
        # 搜索网页
        logger.info("使用Google Custom Search API进行搜索")
        search_results = google_search(query, num_results=5)
        
        if not search_results:
            return jsonify({
                'success': False,
                'error': '未找到相关搜索结果'
            }), 404
        
        # 提取网页内容（限制数量，避免超时）
        logger.info(f"开始提取 {min(len(search_results), 3)} 个网页的内容...")
        print(f"开始提取 {min(len(search_results), 3)} 个网页的内容...")
        enriched_results = []
        max_extract = min(len(search_results), 3)  # 最多提取3个，避免超时
        successful_extracts = 0
        for i, result in enumerate(search_results[:max_extract], 1):
            try:
                logger.info(f"正在提取网页 {i}/{max_extract}: {result['link']}")
                print(f"正在提取网页 {i}/{max_extract}: {result['link']}")
                content = extract_webpage_content(result['link'], max_length=1000)  # 减少内容长度
                if content:
                    result['content'] = content
                    successful_extracts += 1
                    logger.info(f"成功提取网页 {i} 的内容，长度: {len(content)} 字符")
                else:
                    # 即使提取失败，也保留搜索结果（至少有用摘要）
                    logger.info(f"网页 {i} 提取失败，将使用搜索结果摘要")
                enriched_results.append(result)
            except Exception as extract_error:
                logger.warning(f"提取网页 {i} 失败: {str(extract_error)}")
                # 继续处理下一个，保留搜索结果
                enriched_results.append(result)
        
        logger.info(f"成功提取 {successful_extracts}/{max_extract} 个网页的内容，共 {len(enriched_results)} 个结果")
        print(f"成功提取 {successful_extracts}/{max_extract} 个网页的内容，共 {len(enriched_results)} 个结果")
        
        # 构建AI总结提示词（英文）
        summary_prompt = f"Please summarize the following search results about '{query}', extract key information and organize it into a clear summary:\n\n"
        for i, result in enumerate(enriched_results, 1):
            summary_prompt += f"=== Source {i} ===\n"
            summary_prompt += f"Title: {result.get('title', 'No title')}\n"
            if 'content' in result and result['content']:
                summary_prompt += f"Content: {result['content']}\n\n"
            else:
                summary_prompt += f"Summary: {result.get('snippet', 'No summary')}\n\n"
        
        summary_prompt += "Please generate a concise and comprehensive summary based on the above information, including main points and key content. Write in English."
        
        # 调用AI进行总结
        logger.info("正在使用AI总结搜索结果...")
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional information analyst expert, skilled at extracting and summarizing key information from multiple sources. Your summaries should be concise, accurate, and well-organized. Always respond in English."
                    },
                    {
                        "role": "user",
                        "content": summary_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500,
                timeout=120  # 增加超时时间到120秒
            )
            
            if not response or not response.choices:
                raise Exception("API返回数据格式错误")
            
            summary = response.choices[0].message.content
            logger.info("AI总结完成")
            
            # 尝试从搜索查询中提取目的地，或使用传入的目的地参数
            destination = data.get('destination', '')
            if not destination:
                # 尝试从查询中提取目的地（改进的启发式方法）
                # 将多词目的地和单词目的地分开处理，提高识别准确率
                query_words = query.split()
                multi_word_destinations = ['hong kong', 'new york']
                single_word_destinations = ['tokyo', 'beijing', 'shanghai', 'paris', 'london', 
                                          'bangkok', 'singapore', 'seoul', 'osaka', 'kyoto', 'taipei',
                                          'sydney', 'melbourne', 'dubai', 'istanbul', 'rome',
                                          'barcelona', 'amsterdam', 'vienna', 'prague', 'berlin', 'moscow']
                
                # 首先检查多词目的地（2-3个连续词的组合）
                query_lower = query.lower()
                for multi_dest in multi_word_destinations:
                    if multi_dest in query_lower:
                        destination = multi_dest.title()
                        break
                
                # 如果没找到多词目的地，再检查单词目的地
                if not destination:
                    for word in query_words:
                        word_lower = word.lower().strip('.,!?')
                        if word_lower in single_word_destinations:
                            destination = word_lower.title()
                            break
                
                # 如果还是没找到，使用查询的前几个词作为目的地
                if not destination and query_words:
                    destination = query_words[0].title()
            
            # 保存到知识库（如果有目的地信息）
            if destination:
                try:
                    # 准备保存的文本
                    rag_texts = [summary]
                    
                    # 添加参考链接的摘要信息
                    references = []
                    for ref in enriched_results:
                        ref_info = {
                            'title': ref.get('title', ''),
                            'link': ref.get('link', ''),
                            'snippet': ref.get('snippet', '')
                        }
                        references.append(ref_info)
                        # 如果有摘要，也添加到文本中
                        if ref.get('snippet'):
                            ref_text = f"Title: {ref['title']}\nSummary: {ref['snippet']}"
                            rag_texts.append(ref_text)
                    
                    # 保存知识库
                    save_knowledge_base(
                        destination=destination,
                        texts=rag_texts,
                        references=references,
                        search_query=query
                    )
                    logger.info(f"搜索内容已保存到知识库，目的地: {destination}")
                    print(f"搜索内容已保存到知识库，目的地: {destination}")
                except Exception as save_error:
                    logger.warning(f"保存搜索内容到知识库失败: {str(save_error)}")
                    print(f"保存搜索内容到知识库失败: {str(save_error)}")
                    # 不影响返回结果，继续执行
            else:
                logger.info("未检测到目的地信息，跳过知识库保存")
            
            # 返回总结和参考链接
            return jsonify({
                'success': True,
                'summary': summary,
                'references': [{'title': r.get('title', ''), 'link': r.get('link', ''), 'snippet': r.get('snippet', '')} for r in enriched_results],
                'using_api': True,
                'destination': destination if destination else None  # 返回检测到的目的地
            })
        except Exception as api_error:
            import traceback
            api_error_detail = traceback.format_exc()
            logger.error(f"AI总结出错详情: {api_error_detail}")
            print(f"AI总结出错详情: {api_error_detail}")
            
            # 提供更详细的错误信息
            error_str = str(api_error)
            error_msg = "AI summary is temporarily unavailable. "
            
            if '401' in error_str or 'Unauthorized' in error_str:
                error_msg += "API key is invalid or expired. Please check your DeepSeek API key configuration."
            elif '429' in error_str:
                error_msg += "API rate limit exceeded. Please try again later."
            elif 'timeout' in error_str.lower():
                error_msg += "Request timeout. Please try again."
            else:
                error_msg += f"Error: {error_str[:200]}"
            
            # 如果AI总结失败，至少返回搜索结果和错误信息
            return jsonify({
                'success': True,
                'summary': error_msg + '\n\nHere are the search results:',
                'references': [{'title': r.get('title', ''), 'link': r.get('link', '')} for r in enriched_results],
                'using_api': True,
                'summary_error': True
            })
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"搜索错误详情: {error_detail}")
        print(f"搜索错误详情: {error_detail}")
        
        # 提供更友好的错误信息
        error_message = str(e)
        if 'timeout' in error_message.lower():
            error_message = 'Request timeout. The search and content extraction process may take longer than expected. Please try again.'
        elif 'Connection' in error_message:
            error_message = 'Connection error. Please check your network connection.'
        else:
            error_message = f'Search error: {error_message[:200]}'
        
        return jsonify({
            'success': False,
            'error': error_message
        }), 500

@app.route('/api/generate-plan', methods=['POST'])
def generate_plan():
    """生成旅游计划 (核心修改部分)"""
    try:
        # 记录请求信息，用于调试不同IP的问题
        client_ip = request.remote_addr
        origin = request.headers.get('Origin', 'N/A')
        logger.info(f"收到生成计划请求 - IP: {client_ip}, Origin: {origin}, Content-Type: {request.content_type}")
        
        # 检查请求是否为JSON格式
        if not request.is_json:
            logger.warning(f"请求不是JSON格式 - IP: {client_ip}, Content-Type: {request.content_type}")
            return jsonify({
                'success': False,
                'error': '请求格式错误，需要JSON格式'
            }), 400
        
        # 安全获取JSON数据
        try:
            data = request.get_json(force=True) if request.json is None else request.json
        except Exception as e:
            logger.error(f"解析JSON数据失败 - IP: {client_ip}, 错误: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'解析请求数据失败: {str(e)}'
            }), 400
        
        if not data:
            logger.warning(f"请求数据为空 - IP: {client_ip}")
            return jsonify({
                'success': False,
                'error': '请求数据为空'
            }), 400
        
        days = data.get('days')
        destination = data.get('destination')
        budget = data.get('budget', 'Moderate')
        preferences = data.get('preferences', '')
        llm_mode = data.get('llm_mode', LLM_MODE).lower()
        
        # 搜索上下文 (可能为空)
        search_context = data.get('search_context', '') 
        
        if not days or not destination:
            return jsonify({'success': False, 'error': 'Missing info'}), 400

        logger.info(f"生成请求: {destination}, {days}天, 模式: {llm_mode}")
        logger.info(f"搜索上下文长度: {len(search_context) if search_context else 0}")

        # ==========================================
        # 关键修复：初始化所有变量，避免 UnboundLocalError
        # Python看到变量在函数内被赋值就会认为它是局部变量
        # 必须在函数开始就初始化，否则会报 UnboundLocalError
        # ==========================================
        plan = ""
        prompt = ""  # 必须在函数开始就初始化！
        used_rag = False

        # ==========================================
        # 本地模式 (Local / Ollama)
        # ==========================================
        if llm_mode == 'local':
            logger.info("进入本地模式处理流程")
            used_rag = False
            
            try:
                # 1. 准备知识库 (VectorStore)
                vectorstore = None
                
                # 尝试加载历史知识库
                hist_vs = load_destination_knowledge_base(destination)
                
                if search_context and len(search_context.strip()) > 10:
                    # Case A: 有新的搜索结果
                    logger.info("使用新的搜索上下文...")
                    new_docs = create_documents_from_text(search_context, {'source': 'current_search'})
                    
                    # 保存新知识到磁盘
                    save_knowledge_base(destination, [search_context], data.get('references', []))
                    
                    if hist_vs:
                        # 简单合并：向内存中的历史库添加新文档
                        hist_vs.add_documents(new_docs)
                        vectorstore = hist_vs
                        logger.info("已合并历史库与新搜索内容")
                    else:
                        vectorstore = create_vectorstore_from_documents(new_docs)
                        logger.info("使用新搜索内容创建知识库")
                        
                elif hist_vs:
                    # Case B: 无新搜索，但有历史记录
                    logger.info("未提供新搜索结果，使用历史知识库...")
                    vectorstore = hist_vs
                    
                # 2. 决策：使用 RAG 还是 普通生成
                if vectorstore:
                    # ---> 执行手动 RAG <---
                    used_rag = True
                    logger.info("知识库就绪，执行手动 RAG...")
                    
                    llm = create_ollama_llm()
                    
                    prompt_template = """You are a travel planner. Create a {days}-day travel plan for {destination}.
                    
Reference Information (from search/history):
{context}

User Requirements:
- Budget: {budget}
- Preferences: {preferences}
- Question: {question}

Format:
Day 1: Morning / Afternoon / Evening / Stay
... (for all days)

Important:
- Use the reference information for accurate spots/hotels.
- If information is missing, use your own knowledge.
- Write in English.
"""
                    query_text = f"Create a detailed {days}-day itinerary for {destination}."
                    
                    # 调用手动推理
                    result = manual_rag_inference(
                        vectorstore=vectorstore,
                        llm=llm,
                        query=query_text,
                        custom_prompt=prompt_template
                            .replace('{days}', str(days))
                            .replace('{destination}', destination)
                            .replace('{budget}', budget)
                            .replace('{preferences}', preferences)
                    )
                    plan = result['answer']
                    
                else:
                    # Case C: 既无搜索也无历史 ---> 普通生成 (不强制搜索)
                    used_rag = False
                    logger.info("无搜索上下文且无历史库，使用 Ollama 原生知识直接生成...")
                    
                    prompt = f"""Create a {days}-day travel plan for {destination}.
Budget: {budget}
Preferences: {preferences}

Format:
Day 1: ...
Day 2: ...

Write in English. Be specific and practical."""
                    
                    # 调用普通的无状态 LLM 接口
                    plan = call_local_llm(prompt, "You are a travel planner.")

                # 在计划文本前添加RAG使用状态提示（使用HTML格式，前端会直接渲染）
                if used_rag:
                    rag_status = "✓ RAG Enabled (Retrieval-Augmented Generation)"
                else:
                    rag_status = "○ RAG Disabled (Using LLM Native Knowledge)"
                plan_with_status = f"<div style='color: #666; font-size: 0.85em; margin-bottom: 10px; padding: 5px; background-color: #f5f5f5; border-left: 3px solid #1459CF;'>{rag_status}</div>\n\n{plan}"

                return jsonify({
                    'success': True, 
                    'plan': plan_with_status, 
                    'references': data.get('references', []),
                    'source': 'rag' if used_rag else 'llm_native'
                })

            except Exception as e:
                logger.error(f"本地生成出错: {e}")
                # 降级策略：如果 RAG 挂了，尝试最后一次裸调
                logger.info("尝试降级到普通 LLM 生成...")
                fallback_prompt = f"Create a {days}-day trip to {destination}."
                fallback_plan = call_local_llm(fallback_prompt)
                # 降级模式未使用RAG
                rag_status = "○ RAG Disabled (Fallback Mode - Using LLM Native Knowledge)"
                fallback_plan_with_status = f"<div style='color: #666; font-size: 0.85em; margin-bottom: 10px; padding: 5px; background-color: #f5f5f5; border-left: 3px solid #ff9800;'>{rag_status}</div>\n\n{fallback_plan}"
                return jsonify({'success': True, 'plan': fallback_plan_with_status, 'note': 'Generated using fallback mode due to error.'})

        # ==========================================
        # 云端模式 (Cloud / DeepSeek)
        # ==========================================
        else:
            logger.info("进入云端模式处理流程")
            # 构建提示词 (确保 prompt 被赋值)
            prompt = f"""Create a detailed {days}-day travel plan for {destination}.
Budget: {budget}
Preferences: {preferences}

Please provide a comprehensive itinerary including:
- Day-by-day schedule with morning, afternoon, and evening activities
- Recommended attractions, restaurants, and accommodations
- Practical tips and transportation information
- Estimated costs where applicable

Format the response in a clear, structured way with day headings and bullet points.
Write in English."""
            
            # 安全检查：确保 prompt 已设置
            if not prompt:
                prompt = f"Create a {days}-day travel plan for {destination}. Budget: {budget}. Preferences: {preferences}."
                logger.warning("prompt 未正确设置，使用备用提示词")
            
            # 调用云端DeepSeek API
            logger.info("正在调用DeepSeek API...")
            logger.info(f"API密钥: {client.api_key[:10]}...")  # 只显示前10个字符
            logger.debug(f"使用提示词长度: {len(prompt)} 字符")
            print("正在调用DeepSeek API...")  # 调试日志
            print(f"API密钥: {client.api_key[:10]}...")  # 只显示前10个字符
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional travel planner, skilled at creating detailed and practical travel plans. Your responses should be well-structured, accurate, and provide reasonable suggestions. Always respond in English."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=4000,  # Increased to support longer itineraries (20+ days)
                timeout=120  # Increased timeout for longer responses
            )
            
            if not response or not response.choices:
                raise Exception("API返回数据格式错误")
            
            plan = response.choices[0].message.content
            
            # Check if response was truncated due to token limit
            finish_reason = response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else None
            if finish_reason == 'length':
                logger.warning(f"API response was truncated due to token limit. Requested {days} days.")
                print(f"警告: API响应因token限制被截断。请求了{days}天。")
                plan += f"\n\n[Note: The response was truncated due to token limit. The itinerary may be incomplete. For longer trips ({days} days), please consider splitting into multiple requests or reducing the detail level.]"
            
            logger.info("API调用成功，返回计划")
            print("API调用成功，返回计划")  # 调试日志
        
        return jsonify({
            'success': True,
            'plan': plan,
            'references': []  # 生成计划不使用搜索，所以没有参考链接
        })
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"生成计划错误详情: {error_detail}")
        print(f"生成计划错误详情: {error_detail}")  # 调试日志
        
        # 返回友好的错误信息
        error_str = str(e)
        if '401' in error_str or 'Unauthorized' in error_str:
            return jsonify({
                'success': False,
                'error': 'API密钥无效或已过期，请检查您的DeepSeek API密钥配置'
            }), 401
        elif '429' in error_str:
            return jsonify({
                'success': False,
                'error': 'API调用频率过高，请稍后再试'
            }), 429
        else:
            return jsonify({
                'success': False,
                'error': f'生成计划时出错: {error_str}'
            }), 500
    

@app.route('/api/export-ics', methods=['POST'])
def export_ics():
    """导出行程为.ics日历文件"""
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': '请求格式错误，需要JSON格式'
            }), 400
        
        data = request.json
        plan_text = data.get('plan', '')
        destination = data.get('destination', 'Travel Plan')
        start_date = data.get('start_date', None)  # 格式: 'YYYY-MM-DD'
        
        if not plan_text:
            return jsonify({
                'success': False,
                'error': '行程内容为空'
            }), 400
        
        # 解析并生成ICS文件
        calendar = parse_plan_to_ics(plan_text, destination, start_date)
        
        # 生成ICS文件内容
        ics_content = str(calendar)
        
        # 创建文件对象
        ics_file = BytesIO()
        ics_file.write(ics_content.encode('utf-8'))
        ics_file.seek(0)
        
        # 生成文件名
        filename = f"travel_plan_{destination.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.ics"
        
        logger.info(f"成功生成ICS文件: {filename}")
        return send_file(
            ics_file,
            mimetype='text/calendar',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"导出ICS文件错误: {error_detail}")
        return jsonify({
            'success': False,
            'error': f'导出ICS文件时出错：{str(e)}'
        }), 500

@app.route('/manual-kb')
def manual_kb():
    """返回手动添加攻略页面"""
    return render_template('manual_kb.html')

@app.route('/api/save-manual-kb', methods=['POST'])
def save_manual_kb():
    """保存用户手动输入的攻略到知识库"""
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': '请求格式错误，需要JSON格式'
            }), 400
        
        data = request.json
        destination = data.get('destination', '').strip()
        content = data.get('content', '').strip()
        
        if not destination:
            return jsonify({
                'success': False,
                'error': '目的地不能为空'
            }), 400
        
        if not content:
            return jsonify({
                'success': False,
                'error': '攻略内容不能为空'
            }), 400
        
        # 保存到知识库，标记为手动输入
        kb_id = save_knowledge_base(
            destination=destination,
            texts=[content],
            references=[],
            search_query='',
            is_manual=True  # 标记为手动输入
        )
        
        if kb_id:
            logger.info(f"手动攻略已保存: 目的地={destination}, kb_id={kb_id}")
            return jsonify({
                'success': True,
                'message': f'攻略已成功保存到知识库',
                'kb_id': kb_id
            })
        else:
            return jsonify({
                'success': False,
                'error': '保存失败，请重试'
            }), 500
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"保存手动攻略错误: {error_detail}")
        return jsonify({
            'success': False,
            'error': f'保存失败: {str(e)}'
        }), 500

@app.route('/api/list-manual-kb', methods=['GET'])
def list_manual_kb():
    """列出所有手动输入的攻略"""
    try:
        import os
        import json
        from datetime import datetime
        
        KNOWLEDGE_BASE_DIR = os.getenv('KNOWLEDGE_BASE_DIR', 'knowledge_base')
        guides = []
        
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            return jsonify({
                'success': True,
                'guides': []
            })
        
        # 遍历所有知识库目录
        for kb_id in os.listdir(KNOWLEDGE_BASE_DIR):
            kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_id)
            if not os.path.isdir(kb_path):
                continue
            
            texts_file = os.path.join(kb_path, 'texts.json')
            if not os.path.exists(texts_file):
                continue
            
            try:
                with open(texts_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 检查是否为手动输入（通过search_query为空且references为空来判断）
                    is_manual = (not data.get('search_query', '') and 
                                (not data.get('references') or len(data.get('references', [])) == 0))
                    
                    if is_manual:
                        # 合并所有文本内容
                        texts = data.get('texts', [])
                        content = '\n\n'.join(texts) if texts else ''
                        
                        # 获取创建时间
                        created_at = data.get('created_at', '')
                        if created_at:
                            try:
                                # 尝试格式化时间
                                if isinstance(created_at, (int, float)):
                                    created_at = datetime.fromtimestamp(float(created_at)).strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    created_at = str(created_at)
                            except:
                                created_at = str(created_at)
                        else:
                            # 使用文件修改时间
                            try:
                                mtime = os.path.getmtime(texts_file)
                                created_at = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                created_at = '未知'
                        
                        guides.append({
                            'kb_id': kb_id,
                            'destination': data.get('destination', '未知'),
                            'content': content,
                            'created_at': created_at
                        })
            except Exception as e:
                logger.warning(f"读取知识库 {kb_id} 失败: {e}")
                continue
        
        # 按创建时间倒序排列（最新的在前）
        guides.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'guides': guides
        })
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"列出手动攻略错误: {error_detail}")
        return jsonify({
            'success': False,
            'error': f'获取攻略列表失败: {str(e)}'
        }), 500

@app.route('/api/delete-manual-kb', methods=['POST'])
def delete_manual_kb():
    """删除指定的手动攻略"""
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': '请求格式错误，需要JSON格式'
            }), 400
        
        data = request.json
        kb_id = data.get('kb_id', '').strip()
        
        if not kb_id:
            return jsonify({
                'success': False,
                'error': '知识库ID不能为空'
            }), 400
        
        import os
        import shutil
        
        KNOWLEDGE_BASE_DIR = os.getenv('KNOWLEDGE_BASE_DIR', 'knowledge_base')
        kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_id)
        
        if not os.path.exists(kb_path):
            return jsonify({
                'success': False,
                'error': '知识库不存在'
            }), 404
        
        # 删除整个目录
        shutil.rmtree(kb_path)
        logger.info(f"已删除手动攻略: kb_id={kb_id}")
        
        return jsonify({
            'success': True,
            'message': '攻略已删除'
        })
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"删除手动攻略错误: {error_detail}")
        return jsonify({
            'success': False,
            'error': f'删除失败: {str(e)}'
        }), 500

if __name__ == '__main__':
    # 禁用自动重载，避免因系统文件变化（如 torch、Python 标准库）导致服务器不断重启
    # debug=True 仍然保留，用于显示详细错误信息
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)

