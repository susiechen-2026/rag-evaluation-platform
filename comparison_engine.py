# comparison_engine.py
import datetime
import json
from evaluation_engine import RAGEvaluationEngine


class RAGComparisonEngine:
    """对比多个RAG配置的评测结果"""

    def __init__(self, output_dir="./comparisons"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)

    def compare_configurations(self, configs, dataset, metrics_config):
        """
        对比多个RAG配置

        configs: [
            {'name': '配置A', 'rag_client': clientA, 'params': {...}},
            {'name': '配置B', 'rag_client': clientB, 'params': {...}}
        ]
        """
        results = {}

        for config in configs:
            print(f"\n🔍 评测配置: {config['name']}")

            engine = RAGEvaluationEngine(
                rag_client=config['rag_client'],
                metrics_config=metrics_config
            )

            eval_result = engine.run_evaluation(dataset)
            results[config['name']] = {
                'result': eval_result,
                'params': config.get('params', {})
            }

        # 生成对比报告
        self._generate_comparison_report(results)

        return results

    def _generate_comparison_report(self, results):
        """生成对比报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 整理对比数据
        comparison = {
            'timestamp': timestamp,
            'configs': []
        }

        for config_name, data in results.items():
            # 从保存的文件中读取汇总
            # 这里简化处理，实际需要从data中提取
            config_data = {
                'name': config_name,
                'params': data['params']
            }
            comparison['configs'].append(config_data)

        # 保存对比报告
        report_path = f"{self.output_dir}/comparison_{timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

        print(f"✅ 对比报告已保存: {report_path}")