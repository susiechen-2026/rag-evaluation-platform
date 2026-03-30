# data/enhanced_test_generator.py
# 问题：测试集单一，无法反映真实场景
class EnhancedTestGenerator:
    def generate_stratified_dataset(self, base_docs):
        """生成分层测试数据集"""
        return {
            "level_1_simple_fact": [
                {"query": "文档A的作者是谁？", "expected_chunks": ["doc_a_author"]}
            ],
            "level_2_multi_hop": [
                {"query": "比较文档A和文档B的主要观点", "expected_chunks": ["doc_a_summary", "doc_b_summary"]}
            ],
            "level_3_synthesis": [
                {"query": "基于所有文档，给出综合建议", "expected_chunks": ["doc_a", "doc_b", "doc_c"]}
            ],
            "level_4_adversarial": [
                {"query": "文档中没提到的内容是什么？", "expected_chunks": []}
            ]
        }

    def generate_difficulty_metrics(self):
        """为每个测试用例生成难度评分"""
        return {
            "complexity": 1-5,      # 查询复杂度
            "ambiguity": 1-5,       # 歧义程度
            "required_chunks": 1-5, # 需要检索的块数
            "reasoning_depth": 1-5  # 推理深度
        }