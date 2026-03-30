# 根据评测结果给出优化建议
# optimizer.py
import json
import os

class RAGOptimizer:
    """根据评测结果提供优化建议"""

    def __init__(self):
        self.suggestions = {
            'answer_relevancy': {
                'low': [
                    "🔧 答案相关性低，建议优化提示词模板",
                    "💡 在prompt中增加few-shot示例，让模型更理解期望的输出格式",
                    "🎯 检查用户问题是否清晰，是否需要拆解为多个子问题",
                    "📋 确保prompt中明确要求模型基于上下文回答问题"
                ],
                'medium': [
                    "📈 答案相关性中等，可以尝试调整temperature参数（当前值可能过高）",
                    "🔄 考虑增加系统提示词，明确输出规范",
                    "💬 优化问题理解部分，确保模型正确理解用户意图"
                ]
            },
            'faithfulness': {
                'low': [
                    "⚠️ 忠实度低（幻觉严重），建议更换更强的LLM模型",
                    "🔍 检查检索到的上下文是否足够支持答案生成",
                    "📋 在prompt中强调'如果上下文没有相关信息，请说明不知道'",
                    "📚 增加上下文相关性过滤，确保只有相关信息被用于生成"
                ],
                'medium': [
                    "⚙️ 忠实度中等，可以尝试降低temperature",
                    "📚 考虑增加检索数量（top_k），提供更丰富的上下文",
                    "🔧 优化prompt，强调事实一致性的重要性"
                ]
            },
            'contextual_relevancy': {
                'low': [
                    "🔍 检索相关性低，建议优化文档分块策略",
                    "🎯 检查chunk_size是否合适（当前可能过大或过小）",
                    "🔄 考虑更换embedding模型（如从M3E换到BGE）",
                    "📋 优化文档预处理，确保文档质量"
                ],
                'medium': [
                    "📊 检索相关性中等，可以尝试调整检索参数",
                    "⚡ 考虑增加重排序器（reranker）提升精度",
                    "🔧 优化查询扩展，提高检索准确性"
                ]
            },
            'contextual_precision': {
                'low': [
                    "📉 上下文精确率低，相关文档排名靠后",
                    "🔄 建议引入重排序器（reranker）",
                    "🎯 检查检索算法是否需要调整",
                    "⚡ 优化检索参数，如相似度阈值"
                ]
            },
            'contextual_recall': {
                'low': [
                    "📉 上下文召回率低，可能漏掉相关文档",
                    "📚 建议增加检索数量（增大top_k）",
                    "🔄 考虑混合检索（BM25+向量）",
                    "🎯 优化embedding模型，提高语义理解能力"
                ]
            },
            'latency': {
                'low': [
                    "⏱️ 延迟过高，建议优化RAG系统性能",
                    "⚡ 考虑使用更轻量级的模型",
                    "📚 优化检索速度，如使用更高效的向量数据库",
                    "🔧 实现缓存机制，避免重复计算"
                ]
            },
            'token_efficiency': {
                'low': [
                    "📊 Token效率低，建议优化输出长度",
                    "🔧 在prompt中明确限制输出长度",
                    "💡 优化答案生成策略，提高信息密度",
                    "📋 考虑使用更简洁的语言模型"
                ]
            },
            'cost': {
                'low': [
                    "💰 成本过高，建议优化token使用",
                    "⚡ 考虑使用更经济的模型",
                    "🔧 优化prompt，减少不必要的token消耗",
                    "📊 实现token使用监控和优化"
                ]
            }
        }

    def analyze_and_suggest(self, summary_data):
        """分析评测结果并给出优化建议"""

        suggestions = []
        metrics = summary_data.get('metric_averages', {})

        for metric, score in metrics.items():
            if metric in self.suggestions:
                if score < 0.5:
                    level = 'low'
                elif score < 0.7:
                    level = 'medium'
                else:
                    continue  # 高分不给出建议

                suggestions.append({
                    'metric': metric,
                    'score': score,
                    'level': '⚠️ 严重' if level == 'low' else '📌 中等',
                    'suggestions': self.suggestions[metric].get(level, [])
                })

        return suggestions

    def generate_report(self, summary_path):
        """生成完整的优化建议报告"""
        import json
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        suggestions = self.analyze_and_suggest(summary)

        # 生成Markdown报告
        report = []
        report.append("# RAG系统评测优化报告\n")
        report.append(f"评测时间: {summary.get('timestamp', '未知')}\n")
        report.append(f"总测试问题数: {summary.get('total_questions', 0)}\n")

        report.append("\n## 一、指标概览\n")
        report.append("| 指标 | 得分 | 通过率 | 状态 |")
        report.append("|------|------|--------|------|")

        for metric, score in summary.get('metric_averages', {}).items():
            pass_rate = summary.get('pass_rate', {}).get(metric, 0)
            status = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
            report.append(f"| {metric} | {score:.3f} | {pass_rate:.1%} | {status} |")

        report.append("\n## 二、优化建议\n")

        for s in suggestions:
            report.append(f"\n### {s['level']} {s['metric']} (得分: {s['score']:.3f})")
            for tip in s['suggestions']:
                report.append(f"- {tip}")

        # 添加具体失败案例分析
        report.append("\n## 三、失败案例分析\n")
        report.append("通过分析评测结果，以下是可能导致性能问题的常见原因：\n")
        report.append("1. **检索问题**：检索到的文档与查询不相关，导致生成的答案质量差")
        report.append("2. **生成问题**：模型可能过度生成或生成与上下文无关的内容")
        report.append("3. **prompt问题**：提示词设计不当，没有明确指导模型如何基于上下文生成答案")
        report.append("4. **性能问题**：系统响应时间过长，影响用户体验")

        report.append("\n## 四、下一步行动\n")
        report.append("1. **优先处理严重问题**：针对得分<0.5的指标立即优化")
        report.append("2. **单变量实验**：每次只改一个参数，重新评测验证效果")
        report.append("3. **建立基线**：保存每次优化的评测结果，跟踪进展")
        report.append("4. **全面优化**：综合考虑检索、生成、prompt和性能四个方面")
        report.append("5. **持续监控**：定期进行评测，确保系统性能稳定")

        return "\n".join(report)

    def generate_detailed_report(self, summary_path, results_path):
        """生成更详细的优化建议报告，包括具体失败案例分析"""
        # 读取汇总数据
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        # 读取详细结果数据
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # 分析失败案例
        failed_cases = []
        for item in results:
            for metric_name, metric_info in item['metric_scores'].items():
                if metric_info['score'] < 0.7:
                    failed_cases.append({
                        'question': item['input'],
                        'metric': metric_name,
                        'score': metric_info['score'],
                        'reason': metric_info.get('reason', '无原因')
                    })

        # 生成详细报告
        report = []
        report.append("# RAG系统评测详细优化报告\n")
        report.append(f"评测时间: {summary.get('timestamp', '未知')}\n")
        report.append(f"总测试问题数: {summary.get('total_questions', 0)}\n")
        report.append(f"失败案例数: {len(failed_cases)}\n")

        # 指标概览
        report.append("\n## 一、指标概览\n")
        report.append("| 指标 | 得分 | 通过率 | 状态 |")
        report.append("|------|------|--------|------|")

        for metric, score in summary.get('metric_averages', {}).items():
            pass_rate = summary.get('pass_rate', {}).get(metric, 0)
            status = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
            report.append(f"| {metric} | {score:.3f} | {pass_rate:.1%} | {status} |")

        # 优化建议
        suggestions = self.analyze_and_suggest(summary)
        report.append("\n## 二、优化建议\n")

        for s in suggestions:
            report.append(f"\n### {s['level']} {s['metric']} (得分: {s['score']:.3f})")
            for tip in s['suggestions']:
                report.append(f"- {tip}")

        # 失败案例分析
        if failed_cases:
            report.append("\n## 三、失败案例分析\n")
            report.append("### 3.1 失败案例概览\n")
            report.append(f"共发现 {len(failed_cases)} 个失败案例（得分<0.7）\n")

            # 按指标分组
            cases_by_metric = {}
            for case in failed_cases:
                if case['metric'] not in cases_by_metric:
                    cases_by_metric[case['metric']] = []
                cases_by_metric[case['metric']].append(case)

            for metric, cases in cases_by_metric.items():
                report.append(f"\n### 3.2 {metric} 失败案例（共 {len(cases)} 个）\n")
                for i, case in enumerate(cases[:5]):  # 只展示前5个案例
                    report.append(f"\n**案例 {i+1}**\n")
                    report.append(f"- 问题: {case['question'][:100]}...\n")
                    report.append(f"- 得分: {case['score']:.3f}\n")
                    report.append(f"- 原因: {case['reason'][:200]}...\n")
                if len(cases) > 5:
                    report.append(f"\n... 还有 {len(cases) - 5} 个案例未展示\n")

        report.append("\n## 四、优化行动计划\n")
        report.append("### 4.1 短期行动（1-2周）\n")
        report.append("1. **修复严重问题**：优先处理得分<0.5的指标")
        report.append("2. **优化prompt**：改进提示词设计，提高答案质量")
        report.append("3. **调整检索参数**：优化top_k、chunk_size等参数")

        report.append("\n### 4.2 中期行动（2-4周）\n")
        report.append("1. **评估模型**：考虑更换更适合的LLM模型")
        report.append("2. **优化检索系统**：引入重排序器或混合检索")
        report.append("3. **性能优化**：减少响应时间，提高系统效率")

        report.append("\n### 4.3 长期行动（1-3个月）\n")
        report.append("1. **建立自动化评测流程**：定期进行系统评测")
        report.append("2. **持续监控**：实时监控系统性能和质量")
        report.append("3. **迭代优化**：基于用户反馈持续改进系统")

        return "\n".join(report)