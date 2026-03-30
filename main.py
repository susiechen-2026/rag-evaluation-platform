# main.py
import os
import logging
from deepeval.models import GPTModel,OllamaModel

from data.rag_test_data import load_dataset_from_json
from metrics_config import RAGEvaluationMetrics
from rag_client import LocalRAGClient
from evaluation_engine import RAGEvaluationEngine
from visualizer import ResultVisualizer
from optimizer import RAGOptimizer
from config_manager import config_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    try:
        logger.info("开始RAG系统评测流程")
        
        # 1. 初始化评测模型（使用本地Ollama）
        judge_model = OllamaModel(model="deepseek-r1:8b",
                                  base_url="http://localhost:11434/",
                                  temperature=0)

        # 2.使用付费模型
        #
        # api_key = config_manager.get_api_key()
        # if not api_key or api_key == "YOUR_API_KEY_HERE":
        #     raise ValueError("请在config.json中设置API密钥或设置环境变量 DASHSCOPE_API_KEY")
        # logger.info("成功获取API密钥")

         # 获取模型配置
        # model_config = config_manager.get_model_config()
        # judge_model = GPTModel(
        #    # "name": "qwen-plus",
        #    # "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        #     model=model_config.get("name", "qwen-plus"),  # 或 qwen2.5, llama3等
        #     base_url=model_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        #     api_key=api_key,
        #     temperature=model_config.get("temperature", 0)
        # )
        logger.info("成功初始化评测模型")

        # 2. 配置评测指标
        evaluation_config = config_manager.get_evaluation_config()
        metrics_config = RAGEvaluationMetrics(
            threshold=evaluation_config.get("threshold", 0.7),
            model=judge_model
        )
        logger.info("成功配置评测指标")

        # 3. 连接RAG系统（项目1）
        # rag_client = RAGSystemClient(
        #     api_base_url="http://localhost:8000/api/v1"  # 你的项目1地址
        # )
        rag_config = config_manager.get_rag_config()
        if rag_config.get("type") == "local":
            local_function = rag_config.get("local_function", "rag.rag_observe.rag_chatbot")
            # 动态导入本地RAG函数
            module_name, function_name = local_function.rsplit('.', 1)
            module = __import__(module_name, fromlist=[function_name])
            rag_chatbot = getattr(module, function_name)
            rag_client = LocalRAGClient(rag_chatbot)
        else:
            # 这里可以添加其他类型的RAG系统连接
            from rag.rag_observe import rag_chatbot
            rag_client = LocalRAGClient(rag_chatbot)
        logger.info("成功连接RAG系统")

        # 4. 准备测试数据集
        logger.info("加载测试数据集...")
        # dataset = create_manual_dataset()  # 或从文件加载
        # 根据文档生成问题
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data/dataset.json")
        dataset = load_dataset_from_json(dataset_path)
        logger.info(f"共 {len(dataset.goldens)} 个测试问题")

        # 5. 创建评测引擎
        output_config = config_manager.get_output_config()
        engine = RAGEvaluationEngine(
            rag_client=rag_client,
            metrics_config=metrics_config,
            output_dir=output_config.get("eval_results_dir", "./eval_results")
        )
        logger.info("成功创建评测引擎")

        # 6. 执行评测
        logger.info("开始RAG系统评测...")
        results = engine.run_evaluation(
            dataset=dataset,
            batch_size=evaluation_config.get("batch_size", 5),
            max_workers=evaluation_config.get("max_workers", 4)  # 设置最大并行工作线程数
        )
        logger.info("评测执行完成")

        # 7. 可视化分析
        logger.info("生成可视化报告...")
        visualizer = ResultVisualizer(output_dir=output_config.get("visualizations_dir", "./visualizations"))

        # 找到最新的结果文件
        import glob
        try:
            latest_json = max(glob.glob("./eval_results/eval_results_*.json"), key=os.path.getctime)
            latest_summary = max(glob.glob("./eval_results/summary_*.json"), key=os.path.getctime)
            logger.info(f"找到最新结果文件: {latest_json} 和 {latest_summary}")
        except ValueError:
            logger.error("未找到评测结果文件")
            raise

        visualizer.plot_metric_distribution(latest_json)
        visualizer.plot_radar_chart(latest_summary)
        visualizer.plot_failed_cases(latest_json, threshold=0.7)
        logger.info("可视化报告生成完成")

        # 8. 生成优化建议
        logger.info("生成优化建议...")
        optimizer = RAGOptimizer()
        # 生成详细报告，包括失败案例分析
        suggestions = optimizer.generate_detailed_report(latest_summary, latest_json)

        # 保存建议报告
        report_path = "./eval_results/optimization_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(suggestions)

        logger.info(f"优化报告已保存: {report_path}")
        print("\n" + suggestions)

        logger.info("RAG评测完成！")
        print("\n✨ RAG评测完成！")
    except Exception as e:
        logger.error(f"评测过程中出现错误: {str(e)}", exc_info=True)
        print(f"\n❌ 评测过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()