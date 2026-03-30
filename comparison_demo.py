# comparison_demo.py
"""
对比不同配置的RAG系统：
- 配置A: M3E-base + top_k=3
- 配置B: M3E-base + top_k=5
- 配置C: BGE-large-zh + top_k=3
"""
from deepeval.models import OllamaModel

from data.rag_test_data import create_manual_dataset
from metrics_config import RAGEvaluationMetrics
from rag_client import LocalRAGClient
from comparison_engine import RAGComparisonEngine
from config_manager import config_manager
from evaluation_engine import RAGEvaluationEngine

# 假设你有不同配置的RAG函数
def rag_config_a(question):
    # 实现配置A的RAG调用
    pass

def rag_config_b(question):
    # 实现配置B的RAG调用
    pass

def rag_config_c(question):
    # 实现配置C的RAG调用
    pass

configs = [
    {
        'name': 'M3E-base + top_k=3',
        'rag_client': LocalRAGClient(rag_config_a),
        'params': {'embedding': 'm3e-base', 'top_k': 3}
    },
    {
        'name': 'M3E-base + top_k=5',
        'rag_client': LocalRAGClient(rag_config_b),
        'params': {'embedding': 'm3e-base', 'top_k': 5}
    },
    {
        'name': 'BGE-large + top_k=3',
        'rag_client': LocalRAGClient(rag_config_c),
        'params': {'embedding': 'bge-large-zh', 'top_k': 3}
    }
]
# 2. 配置评测指标
judge_model = OllamaModel(model="deepseek-r1:8b",
                                  base_url="http://localhost:11434/",
                                  temperature=0)

evaluation_config = config_manager.get_evaluation_config()
metrics_config = RAGEvaluationMetrics(
            threshold=evaluation_config.get("threshold", 0.7),
            model=judge_model
        )


comparison = RAGComparisonEngine()
dataset = create_manual_dataset()
results = comparison.compare_configurations(
    configs=configs,
    dataset=dataset,
    metrics_config=metrics_config
)