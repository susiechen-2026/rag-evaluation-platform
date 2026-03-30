import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI  # 或其他LLM客户端

# 1. 加载向量库
embedding = HuggingFaceEmbeddings(
    model_name="moka-ai/m3e-base",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
db = FAISS.load_local(
    "faiss_db_m3e",
    embedding,
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever(search_kwargs={"k": 3})  # 返回top-3

# 2. 提示词模板（你的测试经验在这里很有用）
prompt_template = """
你是一位专业的金融理财顾问。请严格基于以下资料回答问题。
如果资料中没有相关信息，请明确说明"根据现有资料无法回答"。

资料：
{context}

问题：{question}

回答："""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 3. 接入LLM（这里以OpenAI格式为例，可替换为DeepSeek/Qwen等）
# llm = OpenAI(
#     base_url="https://api.deepseek.com/v1",  # 或使用API
#     api_key="dummy",
#     temperature=0.1  # 降低幻觉
# )

llm = ChatOpenAI(
    model="deepseek-r1",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)


# 4. 构建RAG问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# 5. 测试
query = "这款理财产品的年化收益率是多少？"
result = qa_chain.invoke(query)
print(f"回答：{result}")