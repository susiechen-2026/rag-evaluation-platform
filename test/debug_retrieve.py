from rag.rag_observe import VectorStore

# 测试检索功能
vector_store = VectorStore()
print(f"加载的文档数量: {len(vector_store.documents) if hasattr(vector_store, 'documents') else 'N/A'}")

test_query = "大语言模型LLM是什么，有什么特点？"
results = vector_store.similarity_search(test_query, k=4)
print(f"检索结果数量: {len(results)}")
print(f"检索结果: {results}")