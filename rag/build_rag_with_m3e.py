# 查看 rag/rag_with_m3e.py 文件#M3E-base（embedding模型）可以用langchain直接集成
#注意：在导入任何相关库之前，添加以下两行代码来设置环境变量：
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 构建向量数据库
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import  DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 加载文档（以你的XX文档为例）
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if os.path.exists(data_path):
    loader = DirectoryLoader(data_path,
                             glob="**/*.pdf",
                             loader_cls=PyPDFLoader)
    documents = loader.load()

    # 2. 文档分块（chunk_size根据业务调整）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)

# 3. 加载 M3E-base 模型
# 设置 Hugging Face 镜像源，从https://hf-mirror.com/moka-ai/m3e-base下载
    # 然后正常加载模型
    model_name = "moka-ai/m3e-base"  # 会自动从HF下载，也可先下载到本地，然后指定本地指定路径
    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # 有GPU可改cuda
        encode_kwargs={"normalize_embeddings": True}  # 归一化提升检索效果
    )

    # 4. 构建 FAISS 向量库
    db = FAISS.from_documents(texts, embedding)

    # 5. 保存到本地
    db.save_local("faiss_db_m3e")
    print(f"✅ M3E 向量库构建完成，保存至 faiss_db_m3e/")