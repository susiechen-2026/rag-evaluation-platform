from deepeval.tracing import observe, update_current_span
from deepeval.metrics import ContextualRelevancyMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import numpy as np


# 模拟向量数据库和embedding函数
import os
from pathlib import Path
import pickle
import hashlib

class VectorStore:
    def __init__(self):
        # 使用绝对路径指向rag/data目录
        data_path = Path(__file__).parent / "data"
        if data_path.exists():
            self.documents = SimpleDirectoryReader(str(data_path)).load_data()
        else:
            # 如果rag/data目录不存在，尝试使用项目根目录下的data
            project_data_path = Path(__file__).parent.parent / "data"
            if project_data_path.exists():
                self.documents = SimpleDirectoryReader(str(project_data_path)).load_data()
            else:
                # 如果都没有，则使用相对路径
                self.documents = SimpleDirectoryReader("data").load_data()
        
        # 初始化TF-IDF向量化器和文档矩阵
        self.vectorizer = None
        self.doc_vectors = None
        self.doc_texts = []
        self.original_docs = []
        
        # 准备文档列表
        for doc in self.documents:
            if hasattr(doc, 'text'):
                doc_text = doc.text
            elif hasattr(doc, 'page_content'):
                doc_text = doc.page_content
            else:
                doc_text = str(doc)
            
            self.doc_texts.append(doc_text)
            self.original_docs.append(doc_text)
    
    def _prepare_vectorizer(self):
        """准备TF-IDF向量化器，仅在首次需要时初始化"""
        if self.vectorizer is None and self.doc_texts:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # 优化TF-IDF向量化器配置
            self.vectorizer = TfidfVectorizer(
                stop_words=None,  # 保留所有词
                lowercase=True,  # 转为小写
                ngram_range=(1, 2),  # 考虑单字和双字
                min_df=1,  # 最小文档频率
                max_df=0.9,  # 最大文档频率
                use_idf=True,  # 使用IDF
                smooth_idf=True,  # 平滑IDF
                sublinear_tf=True  # 子线性TF缩放
            )
            
            # 将所有文档向量化
            self.doc_vectors = self.vectorizer.fit_transform(self.doc_texts)
    
    def similarity_search(self, query: str, k: int = 4) -> list[str]:
        # 使用更先进的相似度算法
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # 确保向量化器已准备好
        self._prepare_vectorizer()
        
        if not self.doc_texts or self.doc_vectors is None:
            return []
        
        # 向量化查询
        query_vector = self.vectorizer.transform([query])
        
        # 计算查询与所有文档的余弦相似度
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # 按相似度排序
        doc_indices = np.argsort(similarities)[::-1]  # 降序排列
        
        # 返回最相关的k个文档
        top_k_indices = doc_indices[:k]
        top_k_similarities = similarities[top_k_indices]
        
        # 只返回相似度高于较低阈值的文档
        threshold = 0.01  # 降低阈值以允许更多文档
        filtered_docs = []
        filtered_similarities = []
        
        for idx, sim in zip(top_k_indices, top_k_similarities):
            if sim > threshold:
                filtered_docs.append(self.original_docs[idx])
                filtered_similarities.append(sim)
        
        # 如果没有超过阈值的文档，返回前k个中最相似的
        if not filtered_docs:
            filtered_docs = [self.original_docs[idx] for idx in top_k_indices[:k]]
        
        return filtered_docs[:k]

    # def similarity_search(self, query: str, k: int = 4) -> list[str]:
    #     # 简化的相似度搜索（实际项目中会用embedding）
    #     # 这里简单返回包含查询词的相关文档
    #     relevant_docs = [doc for doc in self.documents if any(word in doc for word in query.split())]
    #     return relevant_docs[:k]


# 1. 检索器组件 - 添加@observe装饰器在这里
@observe(
    type="retriever",  # 指定组件类型
    name="文档检索器",  # 自定义名称
    metrics=[ContextualRelevancyMetric(threshold=0.3)]  # 降低阈值以适应实际情况
)
def retrieve(query: str) -> list[str]:
    """检索相关文档"""
    # 执行检索逻辑
    vector_store = VectorStore()
    # 增加检索数量以提高找到相关文档的概率
    docs = vector_store.similarity_search(query, k=5)

    # 记录当前span的输入输出，用于评估
    update_current_span(
        input=query,
        retrieval_context=docs,  # 检索到的上下文
        metadata={
            "top_k": 5,
            "retrieval_method": "tfidf_cosine_similarity",
            "query_length": len(query),
            "num_docs_retrieved": len(docs)
        }
    )

    return docs


# 2. 生成器组件
@observe(
    type="llm",
    name="答案生成器",model="gpt-4o",
    metrics=[AnswerRelevancyMetric(threshold=0.5)]  # 适当调整阈值
)
def generate_answer(query: str, context: list[str]) -> str:
    """基于检索到的上下文生成答案"""
    context_str = "\n".join(context)
    prompt = f"""你是一个专业的问答助手，任务是基于提供的上下文回答用户问题。

# 上下文信息
{context_str}

# 用户问题
{query}

# 回答要求
1. 仔细阅读并分析上下文中的所有信息
2. 严格基于上下文内容回答问题，不要使用上下文之外的信息
3. 提取上下文中与问题直接相关的要点
4. 用自然、连贯的语言组织回答
5. 确保回答涵盖问题的所有方面
6. 如果上下文中没有足够信息，请明确指出

# 回答
"""
    client = OpenAI(
        # api_key=os.getenv("DASHSCOPE_API_KEY"),
        # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        base_url="http://localhost:11434/v1/",
        api_key= os.getenv("DEEPSEEK_API_KEY"),
        # base_url= os.getenv("DEEPSEEK_BASE_URL")
    )
    response = client.chat.completions.create(
        # model="qwen-plus",#gpt-4o
        model="codegeex4",# deepseek-reasoner
        messages=[{"role": "user", "content": prompt}],
        extra_body={"thinking": {"type": "enabled"}},
        temperature=0.1  # 降低温度以获得更一致的输出
    )

    answer = response.choices[0].message.content

    # 记录生成器的输入输出
    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=context
        ),
        metadata={
            "model": "codegeex4",
            "temperature": 0,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    )

    return answer


# 3. 主RAG流程（最外层trace）
@observe(
    type="agent",
    name="RAG助手"
)
def rag_chatbot(query: str) -> str:
    """完整的RAG问答流程"""
    # 步骤1：检索相关文档
    context = retrieve(query)

    # 步骤2：生成答案
    answer = generate_answer(query, context)

    # 步骤3：更新整个trace的信息
    update_current_span(
        input=query,
        output=answer,
        metadata={
            "num_docs_retrieved": len(context),
            "query_length": len(query)
        }
    )

    return answer


# 4. 运行测试
if __name__ == "__main__":
    # 测试几个问题
    test_queries = [
        "大语言模型LLM是什么，有什么特点？",
        "自然语言处理（NLP）基准测试是什么？如NLP基准测试通常指一组标准化的数据集、任务和评估指标",
        "什么是RAG即检索增强生成？"
    ]

    for query in test_queries:
        print(f"\n问题: {query}")
        answer = rag_chatbot(query)
        print(f"答案: {answer}")
        print("-" * 50)


