from rag.rag_observe import rag_chatbot

# 测试RAG系统
test_question = "大语言模型LLM是什么，有什么特点？"
print(f"测试问题: {test_question}")
result = rag_chatbot(test_question)
print(f"返回结果: {result}")