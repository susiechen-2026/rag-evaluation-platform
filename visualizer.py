# visualizer.py
#结果可视化
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class ResultVisualizer:
    """评测结果可视化"""

    def __init__(self, output_dir="./visualizations"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)

    def load_results(self, json_path):
        """加载评测结果"""
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def plot_metric_distribution(self, results_path):
        """绘制指标分布图"""
        data = self.load_results(results_path)

        # 整理数据
        metrics_data = []
        for item in data:
            for metric_name, metric_info in item['metric_scores'].items():
                metrics_data.append({
                    'metric': metric_name,
                    'score': round(metric_info['score'],3),
                    'question': item['input'][:30]
                })
        df = pd.DataFrame(metrics_data)

        # 绘制箱线图
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='metric', y='score', data=df)
        plt.title('Distribution of evaluation metric scores')
        plt.xlabel('metric')
        plt.ylabel('score')
        plt.xticks(rotation=45)
        plt.tight_layout()

        save_path = f"{self.output_dir}/metric_distribution.png"
        plt.savefig(save_path)
        plt.show()
        print(f"✅ 分布图已保存: {save_path}")

    def plot_radar_chart(self, summary_path):
        """绘制雷达图（多维度对比）"""
        import json
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        metrics = list(summary['metric_averages'].keys())
        scores = list(summary['metric_averages'].values())

        # 雷达图需要闭合
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        scores += scores[:1]  # 闭合
        angles += angles[:1]
        metrics += metrics[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, scores, 'o-', linewidth=2)
        ax.fill(angles, scores, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics[:-1])
        ax.set_ylim(0, 1)
        ax.set_title('RAG System Multi-Dimensional Evaluation Radar Chart')

        save_path = f"{self.output_dir}/radar_chart.png"
        plt.savefig(save_path)
        plt.show()

    def plot_failed_cases(self, results_path, threshold=0.7):
        """列出低分案例供分析"""
        data = self.load_results(results_path)

        failed_cases = []
        for item in data:
            for metric_name, metric_info in item['metric_scores'].items():
                if metric_info['score'] < threshold:
                    failed_cases.append({
                        'question': item['input'],
                        'metric': metric_name,
                        'score': metric_info['score'],
                        'reason': metric_info.get('reason', '无原因')
                    })

        if failed_cases:
            df = pd.DataFrame(failed_cases)
            print(f"\n⚠️ 发现 {len(failed_cases)} 个低分案例（阈值 < {threshold}）")
            print(df.to_string(index=False))

            # 保存到文件
            csv_path = f"{self.output_dir}/failed_cases.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✅ 低分案例已保存: {csv_path}")
        else:
            print(f"✅ 所有指标均高于阈值 {threshold}")