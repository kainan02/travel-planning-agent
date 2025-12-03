# rag_utils.py
"""
RAG (检索增强生成) 工具模块 - 稳定版 (Manual RAG)
功能：
1. 向量存储管理 (FAISS)
2. 文档处理与切分
3. 手动 RAG 推理流程 (避免 LangChain Chain 的并发崩溃问题)
4. 知识库的文件系统读写
"""
import os
import logging
import json
import hashlib
import warnings
import shutil
from typing import List, Optional, Dict, Any

# ==========================================
# 1. 环境配置与警告抑制
# ==========================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 强力抑制 FAISS 和 Multiprocessing 相关的警告
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')
warnings.filterwarnings('ignore', message='.*resource_tracker.*')
warnings.filterwarnings('ignore', message='.*leaked.*semaphore.*')

# 导入 LangChain 组件
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    # 尝试导入新版 Ollama，失败则回退旧版
    try:
        from langchain_ollama import OllamaLLM as Ollama
    except ImportError:
        from langchain_community.llms import Ollama
except ImportError as e:
    raise ImportError(f"缺少必要的库，请检查安装: {e}")

# 日志配置
logger = logging.getLogger(__name__)

# 全局配置
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
KNOWLEDGE_BASE_DIR = os.getenv('KNOWLEDGE_BASE_DIR', 'knowledge_base')

# 确保知识库目录存在
if not os.path.exists(KNOWLEDGE_BASE_DIR):
    os.makedirs(KNOWLEDGE_BASE_DIR)

# 全局缓存
_embeddings_instance = None


# ==========================================
# 2. 基础组件工厂
# ==========================================

def get_embeddings():
    """获取嵌入模型单例 (避免重复加载)"""
    global _embeddings_instance
    if _embeddings_instance is None:
        model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-MiniLM-L6-v2')
        logger.info(f"正在加载嵌入模型: {model_name}")
        try:
            _embeddings_instance = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'} # 强制使用 CPU 以避免某些 GPU 冲突
            )
            logger.info("嵌入模型加载完成")
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise e
    return _embeddings_instance

def create_documents_from_text(text: str, metadata: Optional[dict] = None) -> List[Document]:
    """将长文本切分为文档块"""
    if not text or not text.strip():
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    texts = splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(texts):
        meta = metadata.copy() if metadata else {}
        meta['chunk_index'] = i
        docs.append(Document(page_content=chunk, metadata=meta))
    
    return docs

def create_vectorstore_from_documents(documents: List[Document]) -> FAISS:
    """从文档创建内存中的 FAISS 向量库"""
    if not documents:
        raise ValueError("文档列表为空，无法创建向量库")
    try:
        return FAISS.from_documents(documents, get_embeddings())
    except Exception as e:
        logger.error(f"创建向量库失败: {e}")
        raise e

def create_ollama_llm(base_url=None, model=None):
    """创建 Ollama LLM 实例"""
    base_url = base_url or OLLAMA_BASE_URL
    model = model or OLLAMA_MODEL
    
    try:
        # 优先尝试新版 API 参数
        return Ollama(
            base_url=base_url,
            model=model,
            temperature=0.5,
            num_predict=3000,
            # timeout=300.0 # 如果库版本支持 timeout 可取消注释
        )
    except TypeError:
        # 回退到旧版参数
        logger.info("使用旧版 Ollama 参数初始化")
        return Ollama(base_url=base_url, model=model, temperature=0.5)

# ==========================================
# 3. 核心：手动 RAG 推理 (Manual RAG)
# ==========================================

def manual_rag_inference(vectorstore, llm, query: str, custom_prompt: str = None) -> Dict[str, str]:
    """
    手动执行 RAG 流程：检索 -> 拼接 -> 生成。
    替代 LangChain 的 Chain 对象，防止 Flask 环境下的并发崩溃。
    
    Args:
        vectorstore: 已初始化的 FAISS 实例
        llm: 已初始化的 Ollama 实例
        query: 用户问题
        custom_prompt: 包含 {context} and {question} 的提示词模板
        
    Returns:
        {'answer': '生成的文本'}
    """
    try:
        logger.info(f"执行手动 RAG 推理... Query: {query[:50]}...")
        
        # 1. 检索 (Retrieve)
        # k=4: 检索最相关的4个片段
        docs = vectorstore.similarity_search(query, k=4)
        
        if not docs:
            logger.warning("未检索到相关文档")
            context_text = "No specific context provided."
        else:
            logger.info(f"检索到 {len(docs)} 个相关文档片段")
            # 拼接上下文，包含源信息
            context_parts = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'unknown')
                content = doc.page_content.replace('\n', ' ')
                context_parts.append(f"[Source {i+1} - {source}]: {content}")
            context_text = "\n\n".join(context_parts)
            
        # 2. 构建 Prompt (Construct)
        if custom_prompt:
            # 简单的字符串替换，比 PromptTemplate 更不容易出错
            final_prompt = custom_prompt.replace("{context}", context_text).replace("{question}", query)
            # 处理可能的其他变量名（如 app.py 里的 prompt 可能用的是 task）
            final_prompt = final_prompt.replace("{task}", query)
        else:
            # 默认 Prompt
            final_prompt = f"""Use the following context to answer the question.
            
Context Information:
{context_text}

Question: {query}

Answer:"""

        # 3. 生成 (Generate)
        logger.info("正在调用 LLM (同步阻塞模式)...")
        # 直接 invoke 是同步的，绝对安全
        response = llm.invoke(final_prompt)
        logger.info("LLM 生成完成")
        
        return {"answer": response}

    except Exception as e:
        logger.error(f"手动 RAG 推理发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # 向上抛出，由 app.py 决定是否降级
        raise e

def call_local_llm(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """
    简单的本地 LLM 调用 (非 RAG)
    用于 app.py 的降级策略 (fallback)
    """
    try:
        llm = create_ollama_llm()
        # 简单的拼接 system prompt
        full_prompt = f"System: {system_prompt}\nUser: {prompt}"
        return llm.invoke(full_prompt)
    except Exception as e:
        logger.error(f"LLM 直接调用失败: {e}")
        return f"Error generating response: {str(e)}"

# ==========================================
# 4. 知识库文件管理
# ==========================================

def load_destination_knowledge_base(destination: str) -> Optional[FAISS]:
    """
    扫描 knowledge_base 目录，加载所有匹配 destination 的数据，
    并合并为一个临时的内存 FAISS 向量库。
    """
    target_dest = destination.lower().strip()
    all_docs = []
    found_kbs = 0
    
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        return None
    
    # 遍历所有子目录
    for kb_id in os.listdir(KNOWLEDGE_BASE_DIR):
        kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_id)
        if not os.path.isdir(kb_path):
            continue
            
        meta_file = os.path.join(kb_path, 'metadata.json')
        texts_file = os.path.join(kb_path, 'texts.json')
        
        # 检查文件完整性
        if os.path.exists(meta_file) and os.path.exists(texts_file):
            try:
                # 1. 检查元数据是否匹配目的地
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    if meta.get('destination', '').lower().strip() != target_dest:
                        continue
                
                # 2. 加载文本并转为文档
                with open(texts_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    texts = data.get('texts', [])
                    for text in texts:
                        if text and text.strip():
                            docs = create_documents_from_text(text, {'source': 'history', 'kb_id': kb_id})
                            all_docs.extend(docs)
                found_kbs += 1
                
            except Exception as e:
                logger.warning(f"读取知识库 {kb_id} 出错: {e}")
                continue
    
    if not all_docs:
        logger.info(f"未找到目的地 '{destination}' 的历史知识库")
        return None
    
    logger.info(f"成功加载 {found_kbs} 个历史知识库，共 {len(all_docs)} 个文档块")
    return create_vectorstore_from_documents(all_docs)

def save_knowledge_base(destination: str, texts: List[str], references: List[dict] = None, search_query: str = "", is_manual: bool = False):
    """
    将文本保存到文件系统。
    注意：这里只保存 JSON 文本，不保存 FAISS 索引文件（因为我们每次都在内存重建，避免版本冲突）。
    
    Args:
        destination: 目的地名称
        texts: 文本列表
        references: 参考链接列表（可选）
        search_query: 搜索查询（可选，用于区分手动输入和搜索输入）
        is_manual: 是否为手动输入（True表示用户手动输入的攻略）
    """
    try:
        import time
        # 生成唯一 ID
        content_hash = hashlib.md5(str(texts).encode('utf-8')).hexdigest()[:8]
        # 如果是手动输入，添加时间戳确保唯一性
        if is_manual:
            timestamp = str(int(time.time()))
            kb_id = f"{destination.replace(' ', '_')}_manual_{timestamp}_{content_hash}"
        else:
            kb_id = f"{destination.replace(' ', '_')}_{content_hash}"
        kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_id)
        
        if not os.path.exists(kb_path):
            os.makedirs(kb_path)
        
        # 获取创建时间
        current_time = time.time()
            
        # 保存文本内容
        with open(os.path.join(kb_path, 'texts.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'destination': destination,
                'search_query': search_query if not is_manual else '',  # 手动输入时search_query为空
                'texts': texts,
                'references': references or [],
                'created_at': str(current_time),
                'is_manual': is_manual  # 添加手动输入标记
            }, f, ensure_ascii=False, indent=2)
            
        # 保存元数据
        with open(os.path.join(kb_path, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'destination': destination,
                'doc_count': len(texts),
                'is_manual': is_manual
            }, f, ensure_ascii=False, indent=2)
            
        logger.info(f"知识库已保存: {kb_path} (手动输入: {is_manual})")
        return kb_id
        
    except Exception as e:
        logger.error(f"保存知识库失败: {e}")
        # 保存失败不应该阻断主流程
        return None