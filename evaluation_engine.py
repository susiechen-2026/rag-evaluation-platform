# evaluation_engine.py
import json
import time
import logging
import concurrent.futures
from datetime import datetime

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

# 配置日志
logger = logging.getLogger(__name__)


class RAGEvaluationEngine:
    """RAG系统评测引擎"""

    def __init__(self, rag_client, metrics_config, output_dir="./eval_results"):
        """
        rag_client: RAG系统客户端
        metrics_config: 评测指标配置
        output_dir: 结果输出目录
        """
        self.rag_client = rag_client
        self.metrics_config = metrics_config
        self.output_dir = output_dir

        import os
        os.makedirs(output_dir, exist_ok=True)

    def run_evaluation(self, dataset, metrics=None, batch_size=5, max_workers=None):
        """
        执行评测

        Args:
            dataset: EvaluationDataset对象
            metrics: 要评测的指标列表（默认使用全部）
            batch_size: 批量处理大小
            max_workers: 最大并行工作线程数

        Returns:
            评测结果
        """
        try:
            if metrics is None:
                # 获取所有指标，但排除性能指标（性能指标在评测过程中单独计算）
                all_metrics = self.metrics_config.get_all_metrics(include_performance=False)
                metrics = list(all_metrics.values())

            test_cases = []
            raw_results = []

            # 定义处理单个测试用例的函数
            def process_test_case(i, golden):
                logger.info(f"[{i + 1}/{len(dataset.goldens)}] 评测问题: {golden.input[:50]}...")
                print(f"\n[{i + 1}/{len(dataset.goldens)}] 评测问题: {golden.input[:50]}...")

                # 记录开始时间
                start_time = time.time()

                # 调用RAG系统
                try:
                    actual_output, retrieval_context = self.rag_client.query(golden.input)
                    # 确保actual_output不为空
                    if not actual_output or actual_output.strip() == "":
                        actual_output = "未获取到相关信息"
                        logger.warning(f"RAG系统返回空结果: {golden.input[:50]}...")
                except Exception as e:
                    logger.error(f"调用RAG系统失败: {str(e)}")
                    # 处理失败情况
                    actual_output = "系统错误，请稍后重试"
                    retrieval_context = []

                # 记录结束时间
                end_time = time.time()

                # 计算延迟（毫秒）
                latency = (end_time - start_time) * 1000

                # 计算token数（简单估算，实际应该使用tokenizer）
                input_tokens = len(golden.input.split())
                output_tokens = len(actual_output.split())
                total_tokens = input_tokens + output_tokens
                token_efficiency = output_tokens / total_tokens if total_tokens > 0 else 0

                # 计算成本（简单估算）
                input_price_per_1k = 0.001
                output_price_per_1k = 0.002
                cost = (input_tokens / 1000) * input_price_per_1k + (output_tokens / 1000) * output_price_per_1k

                # 创建测试用例
                test_case = LLMTestCase(
                    input=golden.input,
                    actual_output=actual_output,
                    expected_output=golden.expected_output,  # 可能为None
                    retrieval_context=retrieval_context,
                    context=retrieval_context  # 添加context参数，用于Hallucination metric
                )

                # 保存原始数据
                raw_result = {
                    'input': golden.input,
                    'expected_output': golden.expected_output,
                    'actual_output': actual_output,
                    'retrieval_context': retrieval_context,
                    'performance': {
                        'latency': latency,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': total_tokens,
                        'token_efficiency': token_efficiency,
                        'cost': cost
                    }
                }

                return test_case, raw_result

            # 使用线程池并行处理
            logger.info(f"开始并行处理评测任务，最大工作线程数: {max_workers}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_index = {executor.submit(process_test_case, i, golden): i for i, golden in enumerate(dataset.goldens)}

                # 收集结果
                results = []
                for future in concurrent.futures.as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        test_case, raw_result = future.result()
                        results.append((i, test_case, raw_result))
                    except Exception as e:
                        logger.error(f"处理测试用例 {i} 失败: {str(e)}")

                # 按原始顺序排序
                results.sort(key=lambda x: x[0])
                for i, test_case, raw_result in results:
                    test_cases.append(test_case)
                    raw_results.append(raw_result)

            # 执行评测
            logger.info("开始执行评测...")
            print("\n🚀 开始执行评测...")
            evaluation_results = evaluate(
                test_cases=test_cases,
                metrics=metrics
            )

            # 保存结果
            self._save_results(evaluation_results, test_cases, raw_results)

            return evaluation_results
        except Exception as e:
            logger.error(f"执行评测失败: {str(e)}", exc_info=True)
            raise

    def _save_results(self, eval_results, test_cases, raw_results):
        """保存评测结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. 保存JSON格式
            json_path = f"{self.output_dir}/eval_results_{timestamp}.json"

            # 整理可序列化的结果
            serializable_results = []
            
            # 确保 eval_results 是列表
            if not isinstance(eval_results, list):
                eval_results = [eval_results]
            
            # 确保我们有足够的eval_results与raw_results匹配
            # 如果evaluate只返回一个结果对象，我们需要为每个测试用例提取相应的指标
            if len(eval_results) == 1 and len(raw_results) > 1:
                # 如果只有一个eval_result对象，但有多个raw_result，说明该对象包含了所有测试用例的结果
                eval_result = eval_results[0]
                
                # 从eval_result.test_results获取所有测试结果
                test_results = eval_result.test_results if hasattr(eval_result, 'test_results') else []
                
                # 将test_results与raw_results配对
                for i, raw in enumerate(raw_results):
                    if i < len(test_results):
                        test_result = test_results[i]
                        # 获取每个指标的分数
                        metric_scores = {}
                        
                        # 从test_result中提取指标数据
                        if hasattr(test_result, 'metrics_data'):
                            for metric in test_result.metrics_data:
                                metric_name = getattr(metric, 'name', 'unknown')
                                metric_scores[metric_name] = {
                                    'score': getattr(metric, 'score', 0),
                                    'reason': getattr(metric, 'reason', ''),
                                    'success': getattr(metric, 'success', False)
                                }
                        elif hasattr(test_result, '__dict__'):
                            # 如果metrics_data不存在，尝试从其他属性获取
                            for attr_name in dir(test_result):
                                if not attr_name.startswith('_') and hasattr(getattr(test_result, attr_name), 'score'):
                                    metric = getattr(test_result, attr_name)
                                    metric_name = attr_name
                                    metric_scores[metric_name] = {
                                        'score': getattr(metric, 'score', 0),
                                        'reason': getattr(metric, 'reason', ''),
                                        'success': getattr(metric, 'success', False)
                                    }
                        
                        # 添加性能指标
                        if 'performance' in raw:
                            performance = raw['performance']
                            
                            # 归一化延迟指标（值越低越好）
                            latency = performance['latency']
                            max_latency = 10000  # 10秒作为最大延迟
                            normalized_latency = max(0, 1 - (latency / max_latency))
                            
                            # 归一化成本指标（值越低越好）
                            cost = performance['cost']
                            max_cost = 0.1  # $0.1作为最大成本
                            normalized_cost = max(0, 1 - (cost / max_cost))
                            
                            metric_scores['latency'] = {
                                'score': normalized_latency,
                                'reason': f'响应时间: {latency:.2f}ms',
                                'success': latency <= 3000  # 3秒阈值
                            }
                            metric_scores['token_efficiency'] = {
                                'score': performance['token_efficiency'],
                                'reason': f'Token效率: {performance["token_efficiency"]:.3f}',
                                'success': performance['token_efficiency'] >= 0.3  # 0.3阈值
                            }
                            metric_scores['cost'] = {
                                'score': normalized_cost,
                                'reason': f'成本: ${cost:.4f}',
                                'success': cost <= 0.02  # $0.02阈值
                            }

                        result_item = {
                            'input': raw['input'],
                            'expected_output': raw['expected_output'],
                            'actual_output': raw['actual_output'],
                            'metric_scores': metric_scores,
                            'performance': raw.get('performance', {})
                        }
                        serializable_results.append(result_item)
                    else:
                        # 如果没有对应的test_result，则创建空的metric_scores
                        result_item = {
                            'input': raw['input'],
                            'expected_output': raw['expected_output'],
                            'actual_output': raw['actual_output'],
                            'metric_scores': {},
                            'performance': raw.get('performance', {})
                        }
                        serializable_results.append(result_item)
            else:
                # 如果eval_results和raw_results数量相等，则一一对应
                for eval_result, raw in zip(eval_results, raw_results):
                    # 获取每个指标的分数
                    metric_scores = {}
                    try:
                        for test_result in eval_result.test_results:
                            if hasattr(test_result, 'metrics_data'):
                                for metric in test_result.metrics_data:
                                    metric_name = getattr(metric, 'name')
                                    metric_scores[metric_name] = {
                                        'score': getattr(metric, 'score', 0),
                                        'reason': getattr(metric, 'reason', ''),
                                        'success': getattr(metric, 'success', False)
                                    }
                    except Exception as e:
                        logger.error(f"处理评测结果失败: {str(e)}")
                        # 继续处理，不中断整个流程

                    # 添加性能指标
                    if 'performance' in raw:
                        performance = raw['performance']
                        
                        # 归一化延迟指标（值越低越好）
                        latency = performance['latency']
                        max_latency = 10000  # 10秒作为最大延迟
                        normalized_latency = max(0, 1 - (latency / max_latency))
                        
                        # 归一化成本指标（值越低越好）
                        cost = performance['cost']
                        max_cost = 0.1  # $0.1作为最大成本
                        normalized_cost = max(0, 1 - (cost / max_cost))
                        
                        metric_scores['latency'] = {
                            'score': normalized_latency,
                            'reason': f'响应时间: {latency:.2f}ms',
                            'success': latency <= 3000  # 3秒阈值
                        }
                        metric_scores['token_efficiency'] = {
                            'score': performance['token_efficiency'],
                            'reason': f'Token效率: {performance["token_efficiency"]:.3f}',
                            'success': performance['token_efficiency'] >= 0.3  # 0.3阈值
                        }
                        metric_scores['cost'] = {
                            'score': normalized_cost,
                            'reason': f'成本: ${cost:.4f}',
                            'success': cost <= 0.02  # $0.02阈值
                        }

                    result_item = {
                        'input': raw['input'],
                        'expected_output': raw['expected_output'],
                        'actual_output': raw['actual_output'],
                        'metric_scores': metric_scores,
                        'performance': raw.get('performance', {})
                    }
                    serializable_results.append(result_item)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)

            logger.info(f"JSON结果已保存: {json_path}")
            print(f"✅ JSON结果已保存: {json_path}")

            # 2. 生成汇总报告
            self._generate_summary_report(serializable_results, timestamp)
        except Exception as e:
            logger.error(f"保存评测结果失败: {str(e)}", exc_info=True)
            raise

    def _generate_summary_report(self, results, timestamp):
        """生成汇总报告"""
        try:
            # 统计各指标的平均分
            metric_aggregates = {}

            for item in results:
                for metric_name, metric_data in item['metric_scores'].items():
                    if metric_name not in metric_aggregates:
                        metric_aggregates[metric_name] = []
                    metric_aggregates[metric_name].append(metric_data['score'])

            # 计算平均值
            summary = {
                'total_questions': len(results),
                'timestamp': timestamp,
                'metric_averages': {},
                'pass_rate': {}
            }

            for metric_name, scores in metric_aggregates.items():
                avg_score = sum(scores) / len(scores)
                pass_count = sum(1 for s in scores if s >= 0.7)  # 默认阈值0.7
                pass_rate = pass_count / len(scores)

                summary['metric_averages'][metric_name] = round(avg_score, 3)
                summary['pass_rate'][metric_name] = round(pass_rate, 3)

            # 保存汇总报告
            summary_path = f"{self.output_dir}/summary_{timestamp}.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            logger.info(f"汇总报告已保存: {summary_path}")
            print(f"✅ 汇总报告已保存: {summary_path}")

            # 打印简表
            logger.info("评测结果摘要")
            print("\n📊 评测结果摘要")
            print("=" * 50)
            print(f"总问题数: {summary['total_questions']}")
            print("\n指标平均分:")
            for metric, score in summary['metric_averages'].items():
                logger.info(f"  {metric}: {score:.3f}")
                print(f"  {metric}: {score:.3f}")
            print("\n通过率 (>=0.7):")
            for metric, rate in summary['pass_rate'].items():
                logger.info(f"  {metric}: {rate:.1%}")
                print(f"  {metric}: {rate:.1%}")
        except Exception as e:
            logger.error(f"生成汇总报告失败: {str(e)}", exc_info=True)
            raise