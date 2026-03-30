from typing import List,Tuple
import os
import chromadb
from chromadb import Settings
from deepeval.tracing import observe, update_current_span
from openai.types import EmbeddingModel
from openai import OpenAI

class VectorStore:
    """基于ChromaDB的向量数据库"""

    def __init__(self, embedding_model: EmbeddingModel, collection_name: str = "docs"):
        self.embedding_model = embedding_model
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))

        # 删除已存在的集合（如果存在）
        try:
            self.client.delete_collection(collection_name)
        except:
            pass

        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        self.documents = []  # 存储原始文档
        self.ids = []  # 存储文档ID

    def add_documents(self, documents: List[str]):
        """添加文档到向量库"""
        self.documents.extend(documents)

        # 生成ID
        start_id = len(self.ids)
        doc_ids = [f"doc_{start_id + i}" for i in range(len(documents))]
        self.ids.extend(doc_ids)

        # 计算向量
        embeddings = self.embedding_model.encode(documents)

        # 存入ChromaDB
        self.collection.add(
            ids=doc_ids,
            embeddings=embeddings.tolist(),
            documents=documents
        )

    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        """检索最相似的k个文档"""
        # 计算查询向量
        query_embedding = self.embedding_model.encode([query])[0]

        # 检索
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )

        return results['documents'][0] if results['documents'] else []


class RAGSystem:
    """支持可切换Embedding模型的RAG系统"""

    def __init__(self, vector_store: VectorStore, llm_model: str = "gpt-4o-mini"):
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.embedding_name = vector_store.embedding_model.get_name()

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """检索相关文档（带追踪）"""
        return self.vector_store.similarity_search(query, k=top_k)

    def generate(self, query: str, context: List[str]) -> str:
        """生成答案（带追踪）"""
        context_str = "\n\n".join(context)
        prompt = f"""基于以下信息回答问题。如果信息不足以回答问题，请说明。

信息：
{context_str}

问题：{query}

回答："""
        client = OpenAI(
            base_url="http://localhost:11434/v1/",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            # base_url= os.getenv("DEEPSEEK_BASE_URL")
        )
        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        return response.choices[0].message.content

    @observe(type="agent", name="RAG问答系统")
    def answer(self, query: str) -> Tuple[str, List[str]]:
        """完整问答流程"""
        # 检索
        context = self.retrieve(query)

        # 生成
        answer = self.generate(query, context)

        # 更新追踪信息
        update_current_span(
            input=query,
            output=answer,
            metadata={
                "embedding_model": self.embedding_name,
                "llm_model": self.llm_model,
                "num_docs_retrieved": len(context)
            }
        )

        return answer, context