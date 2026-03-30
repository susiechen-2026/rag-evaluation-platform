import os
from typing import List
import numpy as np

class EmbeddingModel:
    """统一的Embedding模型接口"""

    def encode(self, texts: List[str]) -> np.ndarray:
        """将文本列表转换为向量"""
        raise NotImplementedError

    def get_name(self) -> str:
        """返回模型名称"""
        raise NotImplementedError


# ========== text2vec 实现 ==========
from text2vec import SentenceModel as Text2VecModel


class Text2VecEmbedding(EmbeddingModel):
    def __init__(self):
        # 使用 text2vec 的轻量级模型
        self.model = Text2VecModel()

    def encode(self, texts: List[str]) -> np.ndarray:
        # text2vec 返回的是列表，转换为numpy数组
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts)
        return np.array(embeddings)

    def get_name(self) -> str:
        return "text2vec"


# ========== M3E-base 实现 ==========
class M3EEmbedding(EmbeddingModel):
    def __init__(self):
        # 使用 sentence-transformers 加载 M3E
        self.model = SentenceTransformer('moka-ai/m3e-base')

    def encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts)

    def get_name(self) -> str:
        return "m3e-base"


# ========== 可选的对比基线：OpenAI ==========
class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, api_key: str = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        # OpenAI embedding API 限制每次最多2048个文本
        embeddings = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i + 100]
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def get_name(self) -> str:
        return "openai"