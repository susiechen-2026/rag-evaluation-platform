from deepeval.dataset import EvaluationDataset, Golden


# 方式1：手工构建测试集（推荐用于小规模验证）
def create_manual_dataset():
    """创建手工标注的测试数据集"""
    goldens = [
        Golden(
            input="大语言模型是多少，有什么特点？",
            expected_output="",
            metadata={"category": "基础查询", "difficulty": "简单"}
        ),
        Golden(
            input="LLMS都分别使用的什么激活函数",
            expected_output="",
            metadata={"category": "基础查询", "difficulty": "中等"}
        ),
        Golden(
            input="transformers的原理是什么，具体解决什么问题？",
            expected_output="",
            metadata={"category": "基础查询", "difficulty": "中等"}
        ),
        Golden(
            input="基金提前赎回会收取多少手续费？",
            expected_output="持有不足1年收取0.5%赎回费",
            metadata={"category": "费率规则", "difficulty": "简单"}
        ),
        Golden(
            input="货币基金和定期理财有什么区别？",
            expected_output="货币基金流动性高，风险低；定期理财收益较高但有锁定期",
            metadata={"category": "产品对比", "difficulty": "复杂"}
        )
    ]

    dataset = EvaluationDataset(goldens=goldens)
    return dataset


# 方式2：从JSON文件加载
def load_dataset_from_json(file_path):
    """从JSON文件加载测试集"""
    import json

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    goldens = []
    for item in data:
        golden = Golden(
            input=item['question'],
            expected_output=item.get('expected_output'),
            metadata=item.get('metadata', {})
        )
        goldens.append(golden)

    return EvaluationDataset(goldens=goldens)


# 方式3：合成数据生成（如果还没有测试集）
def generate_synthetic_dataset(documents, num_questions=20):
    """基于知识文档自动生成测试问题"""
    from deepeval.synthesizer import Synthesizer
    import tempfile
    import os

    # 处理输入
    text_contents = []

    if isinstance(documents, list):
        for doc in documents:
            if os.path.isfile(doc):  # 如果是文件路径
                # 尝试读取文件
                try:
                    if doc.endswith('.pdf'):
                        # 用pypdf读取PDF
                        from pypdf import PdfReader
                        reader = PdfReader(doc, strict=False)
                        text = "\n".join([page.extract_text() for page in reader.pages])
                    else:
                        # 文本文件直接读取
                        with open(doc, 'r', encoding='utf-8') as f:
                            text = f.read()
                    text_contents.append(text)
                except Exception as e:
                    print(f"⚠️ 跳过文件 {doc}: {e}")
                    continue
            else:
                # 已经是文本内容
                text_contents.append(doc)
    else:
        text_contents = [documents]

    if not text_contents:
        print("❌ 没有有效的文档内容")
        return []
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False, encoding='utf-8') as f:
        if isinstance(documents, list):
            f.write("\n".join(documents))
        else:
            f.write(documents)
        temp_path = f.name

    try:
        synthesizer = Synthesizer()
        goldens = synthesizer.generate_goldens_from_docs(
            document_paths=[temp_path],
            max_goldens_per_context=2,  # 控制每个上下文生成几个问题
            include_expected_output=True,  # 这个参数很重要
            # 注意：num_goldens 可能不是直接参数，总数由 max_goldens_per_context * max_contexts_per_document 决定
        )

        # 如果数量不够，可以提示或处理
        if len(goldens) < num_questions:
            print(f"⚠️ 实际生成 {len(goldens)} 个问题，少于请求的 {num_questions}")

        return goldens

    finally:
        # 清理临时文件
        os.unlink(temp_path)

if __name__=='__main__':
    print(generate_synthetic_dataset('rag/data/',5))