# 旅游计划生成器

一个智能旅游计划生成软件，可以根据您的输入（天数、地点、预算等）自动生成详细的旅游安排。

## 功能特点

- 📅 根据旅游天数自动规划行程
- 🌍 支持多个目的地
- 💰 考虑预算限制
- 🎯 个性化推荐景点和活动
- 📝 详细的每日行程安排
- 🔍 **谷歌搜索集成** - 自动搜索目的地最新信息，生成更准确的计划
- 🔎 **独立搜索功能** - 可以手动搜索景点、餐厅、酒店等旅游信息
- 🤖 **本地 LLM 支持** - 支持使用 Ollama 本地部署的模型
- 🧠 **RAG 增强生成** - 使用本地 LLM 时，自动将搜索信息整合到旅行计划中（基于 LangChain 的 RAG 技术）

## 安装步骤

1. 安装Python依赖：
```bash
pip install -r requirements.txt
```

2. 配置API密钥：
   - 创建 `.env` 文件
   - 在 `.env` 文件中填入你的 API 密钥：
     ```
     DEEPSEEK_API_KEY=你的DeepSeek API密钥
     GOOGLE_API_KEY=你的Google API密钥（可选，用于搜索功能）
     GOOGLE_SEARCH_ENGINE_ID=你的Google搜索引擎ID（可选，用于搜索功能）
     LLM_MODE=cloud  # 或 'local' 使用本地 Ollama
     OLLAMA_BASE_URL=http://localhost:11434  # 本地 LLM 地址（仅 local 模式需要）
     OLLAMA_MODEL=llama3.2  # 本地 LLM 模型名称（仅 local 模式需要）
     ```
   
   **注意**：
   - 如果不配置Google API密钥，应用仍可正常使用
   - 搜索功能会降级为简化模式（只提供搜索链接，不显示详细结果）
   - 配置Google API后可以获得完整的搜索结果和自动搜索功能
   - Google Custom Search API 需要同时配置API密钥和搜索引擎ID
   - 使用本地 LLM 模式时，需要先安装并运行 Ollama，并下载相应的模型

3. 运行应用：
```bash
python app.py
```

4. 在浏览器中打开 `http://localhost:5000`

## 使用方法

### 基本使用

1. 输入旅游天数
2. 输入目的地（可以是一个或多个城市）
3. 可选：输入预算、兴趣偏好等
4. 选择 LLM 模式（云端或本地）
5. 点击"生成计划"按钮
6. 等待AI生成详细的旅游计划

### 使用 RAG 增强功能（本地 LLM 模式）

1. 首先使用搜索功能搜索目的地相关信息（例如："东京 旅游攻略"）
2. 查看搜索结果和 AI 总结
3. 选择"本地 LLM"模式
4. 填写旅行计划表单并生成计划
5. 系统会自动将搜索信息整合到计划生成中，生成更准确和详细的旅行计划

**RAG 功能说明**：
- 仅在本地 LLM 模式下启用
- 需要先进行搜索，搜索信息会自动保存
- 使用 LangChain 的 RAG 技术，将搜索信息作为知识库
- 生成的计划会基于搜索信息，更加准确和实用
- **知识库持久化**：搜索信息会自动保存到 `knowledge_base/` 目录
- **智能重用**：下次为同一目的地生成计划时，会自动加载已保存的知识库，无需重新搜索
- 知识库按目的地和搜索查询进行索引，相同的目的地可以共享知识库

## 注意事项

- 需要有效的 DeepSeek API 密钥（云端模式）
- 谷歌搜索功能使用 Google Custom Search API（可选）
- 首次使用可能需要一些时间来生成计划
- 建议提供尽可能详细的信息以获得更好的计划
- DeepSeek API 使用 OpenAI 兼容的接口，价格更实惠
- 使用本地 LLM 模式时：
  - 需要先安装 [Ollama](https://ollama.ai/)
  - 运行 `ollama pull llama3.2` 下载模型（或其他模型）
  - 确保 Ollama 服务正在运行
  - 可以使用 `python test_ollama.py` 测试 Ollama 连接
- RAG 功能依赖以下库：
  - LangChain 及相关组件
  - FAISS（向量存储）
  - sentence-transformers（嵌入模型）
  - 首次使用时会自动下载嵌入模型，可能需要一些时间
- **知识库存储**：
  - 搜索信息会自动保存到 `knowledge_base/` 目录
  - 每个目的地和搜索查询组合会创建一个独立的知识库
  - 知识库包含向量存储、文本内容和元数据
  - 可以通过删除 `knowledge_base/` 目录来清除所有保存的知识库
  - 知识库文件已添加到 `.gitignore`，不会被提交到版本控制

## 获取 Google Custom Search API 密钥（可选）

如果您想使用谷歌搜索功能：

1. 访问 [Google Cloud Console](https://console.cloud.google.com/)
2. 创建新项目或选择现有项目
3. 启用 "Custom Search API"
4. 创建 API 密钥
5. 访问 [Google Custom Search](https://programmablesearchengine.google.com/) 创建搜索引擎
6. 获取搜索引擎 ID (CX)
7. 将 API 密钥和搜索引擎 ID 添加到 `.env` 文件中

