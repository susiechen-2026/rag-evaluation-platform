
import requests
import json
import hashlib
from typing import List, Tuple, Dict, Any


class RAGSystemClient:
    """封装对项目1 RAG系统的调用"""

    def __init__(self, api_base_url="http://localhost:8000/api/v1"):
        self.api_base_url = api_base_url
        self.cache = {}  # 缓存查询结果

    def query(self, question: str) -> Tuple[str, List[str]]:
        """
        调用RAG系统，返回答案和检索到的上下文

        Returns:
            answer: 生成的答案
            contexts: 检索到的文档片段列表
        """
        # 生成缓存键
        cache_key = hashlib.md5(question.encode()).hexdigest()
        
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            response = requests.post(
                f"{self.api_base_url}/generate",
                json={"query": question},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                # 根据项目1的返回格式调整
                answer = data.get('test_scene', {}).get('scene_name', '')
                # 这里需要根据实际返回格式提取检索到的上下文
                contexts = data.get('retrieval_contexts', [])
                # 缓存结果
                self.cache[cache_key] = (answer, contexts)
                return answer, contexts
            else:
                print(f"API调用失败: {response.status_code}")
                return "", []

        except Exception as e:
            print(f"RAG系统调用异常: {e}")
            return "", []

    def batch_query(self, questions: List[str]) -> List[Tuple[str, List[str]]]:
        """批量查询"""
        results = []
        for q in questions:
            print(f"处理问题: {q[:30]}...")
            result = self.query(q)
            results.append(result)
        return results


# 如果RAG系统是本地函数（项目1的简化版）
class LocalRAGClient:
    """直接调用本地RAG函数（如果你把项目1集成进来了）"""

    def __init__(self, rag_function):
        """
        rag_function: 接收问题，返回 (答案, 上下文列表)
        """
        self.rag_function = rag_function
        self.cache = {}  # 缓存查询结果

    def query(self, question: str) -> Tuple[str, List[str]]:
        """统一处理不同格式的返回值"""
        # 生成缓存键
        cache_key = hashlib.md5(question.encode()).hexdigest()
        
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        raw_result = self.rag_function(question)
        print(f"调试: result = {raw_result}")
        
        # 标准化输出
        normalized_result = self._normalize_output(raw_result)
        # 缓存结果
        self.cache[cache_key] = normalized_result
        
        return normalized_result

    def _normalize_output(self, result) -> Tuple[str, List[str]]:
        """将各种格式标准化为 (answer, contexts)"""

        # 处理不同框架的输出
        if isinstance(result, tuple):
            if len(result) == 2:
                return result[0], result[1]
            elif len(result) == 3:
                # 假设格式: (answer, contexts, score)
                return result[0], result[1]
            else:
                return str(result[0]), list(result[1:])

        elif isinstance(result, dict):
            # 尝试常见键名
            answer = result.get('answer') or result.get('result') or result.get('output') or str(result)
            contexts = result.get('contexts') or result.get('source_documents') or result.get('documents') or []
            return answer, contexts

        elif isinstance(result, str):
            return result, []

        elif hasattr(result, '__dict__'):
            # 处理对象
            attrs = result.__dict__
            answer = attrs.get('answer') or attrs.get('response') or str(result)
            contexts = attrs.get('contexts') or attrs.get('source_nodes') or []
            return answer, contexts

        else:
            return str(result), []


    def batch_query(self, questions:List[str]):
        return [self.rag_function(q) for q in questions]