#3.1 创建测试知识库
# 创建一个包含理财/测试知识的测试文档集
import os
from typing import List

from deepeval.evaluate import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

from rag.embeddingModel import Text2VecEmbedding, M3EEmbedding
from rag.rag_embedding_model import RAGSystem, VectorStore

test_documents = [
    "DeepEval是一个开源的LLM评估框架，支持RAG三元组指标，包括AnswerRelevancy、Faithfulness和ContextualRelevancy。",
    "RAG系统包含两个核心组件：检索器负责从知识库中找出相关文档，生成器负责基于这些文档生成答案。",
    "向量数据库用于存储和检索Embedding向量，常用的向量数据库包括Chroma、Weaviate和Qdrant。",
    "Embedding模型将文本转换为高维向量，使得语义相近的文本在向量空间中的距离更近。",
    "理财产品的测试需要关注资金安全、交易正确性和系统性能。",
    "性能测试中常用的指标包括TPS、响应时间和资源利用率。",
    "微服务架构下的接口测试需要关注服务间调用链和幂等性。",
    "测试数据构造工具可以显著提升测试效率，将数据准备时间从小时级缩短到分钟级。",
    "BI报表测试需要通过SQL和ETL验证数据一致性，确保报表数据准确反映业务情况。",
    "大语言模型可以用于自动化测试用例生成和缺陷预测。"
]

# 创建测试问题集（覆盖不同场景）
test_questions = [
    {"input": "DeepEval支持哪些评估指标？", "expected_keywords": ["RAG三元组", "AnswerRelevancy", "Faithfulness", "ContextualRelevancy"]},
    {"input": "RAG系统包含哪些组件？", "expected_keywords": ["检索器", "生成器"]},
    {"input": "什么是向量数据库？", "expected_keywords": ["存储", "检索", "向量", "Embedding"]},
    {"input": "理财产品的测试需要关注什么？", "expected_keywords": ["资金安全", "交易正确性", "性能"]},
    {"input": "如何提升测试数据准备效率？", "expected_keywords": ["工具", "数据构造", "自动化"]},
    {"input": "BI报表测试如何进行数据验证？", "expected_keywords": ["SQL", "ETL", "一致性", "业务逻辑"]},
    {"input": "微服务接口测试的难点是什么？", "expected_keywords": ["调用链", "幂等性", "服务间"]},
    {"input": "性能测试有哪些关键指标？", "expected_keywords": ["TPS", "响应时间", "资源利用率"]}
]

# 3.2 创建评测函数

def evaluate_rag_system(
        rag_system: RAGSystem,
        test_questions: List[dict],
        metrics: List) -> dict:
    """评测RAG系统，返回各项指标的平均得分"""

    test_cases = []

    for q in test_questions:
        query = q["input"]
        answer, context = rag_system.answer(query)

        # 创建测试用例
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=context,
            expected_output=q.get("expected_output", None)
        )
        test_cases.append(test_case)

    # 运行评测
    results = evaluate(
        test_cases=test_cases,
        metrics=metrics
    )

    # 计算平均分
    scores = {}
    for metric in metrics:
        metric_name = metric.__class__.__name__
        scores[metric_name] = sum([m.score for m in results]) / len(results) if results else 0

    return scores, results


def compare_embedding_models():
    """对比text2vec和M3E的效果"""

    # 定义要评测的模型
    embedding_models = [
        Text2VecEmbedding(),
        M3EEmbedding()
    ]

    # 定义评测指标
    metrics = [
        AnswerRelevancyMetric(threshold=0.6),
        FaithfulnessMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.6)
    ]

    results_summary = {}

    for emb_model in embedding_models:
        print(f"\n{'=' * 60}")
        print(f"评测 Embedding 模型: {emb_model.get_name()}")
        print(f"{'=' * 60}")

        # 1. 构建向量库
        print("构建向量库...")
        vector_store = VectorStore(embedding_model=emb_model)
        vector_store.add_documents(test_documents)
        print(f"已添加 {len(test_documents)} 个文档")

        # 2. 创建RAG系统
        rag_system = RAGSystem(vector_store=vector_store)

        # 3. 运行评测
        print("运行评测...")
        scores, results = evaluate_rag_system(
            rag_system=rag_system,
            test_questions=test_questions,
            metrics=metrics
        )

        results_summary[emb_model.get_name()] = {
            "scores": scores,
            "details": results
        }

        # 4. 打印结果
        print(f"\n{emb_model.get_name()} 评测结果:")
        for metric_name, score in scores.items():
            print(f"  {metric_name}: {score:.3f}")

    return results_summary


if __name__ == "__main__":
    # 设置OpenAI API Key（用于DeepEval的评测模型）
    os.environ["OPENAI_API_KEY"] = "your-api-key"

    # 运行对比评测
    results = compare_embedding_models()

    # 输出对比报告
    print("\n" + "=" * 60)
    print("对比评测报告")
    print("=" * 60)

    for model_name, data in results.items():
        print(f"\n{model_name}:")
        for metric, score in data["scores"].items():
            print(f"  {metric}: {score:.3f}")