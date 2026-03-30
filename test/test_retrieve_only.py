from rag.rag_observe import retrieve

# 测试检索功能
test_query = "大语言模型LLM是什么，有什么特点？"
contexts = retrieve(test_query)
print(f"检索到的上下文数量: {len(contexts)}")
for i, ctx in enumerate(contexts):
    print(f"上下文 {i+1}: {ctx[:200]}...")  # 只显示前200个字符