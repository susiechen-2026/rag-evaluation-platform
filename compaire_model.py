from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from rag import rag_with_m3e

# 准备测试数据集
test_queries = [
    "这款理财产品的起投金额是多少？",
    "货币基金支持当天赎回吗？",
    # ... 更多测试问题
]

# 分别测试两个RAG系统
for model_name, rag_func in [("M3E", rag_with_m3e), ("BGE", rag_with_bge)]:
    test_cases = []
    for query in test_queries:
        answer, context = rag_func(query)
        test_cases.append(LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=context
        ))

    results = evaluate(
        test_cases=test_cases,
        metrics=[
            ContextualRelevancyMetric(threshold=0.7),
            AnswerRelevancyMetric(threshold=0.7)
        ]
    )
    print(f"{model_name} 平均得分: {results.average_score()}")