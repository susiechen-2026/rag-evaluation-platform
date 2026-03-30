# metrics_config.py
from deepeval.metrics import (
    # 检索器指标
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,

    # 生成器指标
    AnswerRelevancyMetric,
    FaithfulnessMetric,

    # 综合指标
    HallucinationMetric,
    SummarizationMetric,

    # 新增指标
    # CoherenceMetric,
    ToxicityMetric,
    BiasMetric
)

from typing import Dict, List, Any, Union
from dataclasses import dataclass
from enum import Enum
from deepeval.metrics import BaseMetric
# 新增：自定义性能指标类
class PerformanceMetric(BaseMetric):
    """自定义性能指标基类"""

    def __init__(self, name: str, threshold: float = None):
        self.name = name
        self.threshold = threshold
        self.value = None

    def measure(self, **kwargs) -> float:
        """测量指标值"""
        raise NotImplementedError

    def is_passing(self) -> bool:
        """检查是否通过阈值"""
        if self.threshold is None or self.value is None:
            return True
        return self.value <= self.threshold if self.name == "latency" else self.value >= self.threshold


class LatencyMetric(PerformanceMetric):
    """延迟指标"""

    def __init__(self, max_latency_ms: float = 3000, name: str = "latency", threshold: float = None):
        super().__init__(name, threshold)
        self.max_latency_ms = max_latency_ms

    def measure(self, start_time: float, end_time: float) -> float:
        self.value = (end_time - start_time) * 1000  # 转换为毫秒
        return self.value

    @property
    def __name__(self):
        return "Latency"


class TokenEfficiencyMetric(PerformanceMetric):
    """Token效率指标（答案长度/总token数）"""

    def __init__(self, min_efficiency: float = 0.3, name: str = "token_efficiency", threshold: float = None):
        super().__init__(name, threshold)

    def measure(self, answer_tokens: int, total_tokens: int) -> float:
        if total_tokens == 0:
            self.value = 0
        else:
            self.value = answer_tokens / total_tokens
        return self.value

    @property
    def __name__(self):
        return "TokenEfficiency"


class CostMetric(PerformanceMetric):
    """成本指标"""

    def __init__(self, max_cost_per_query: float = 0.02, name: str = "cost", threshold: float = None):
        super().__init__(name, threshold)

    def measure(self, input_tokens: int, output_tokens: int,
                input_price_per_1k: float = 0.001,
                output_price_per_1k: float = 0.002) -> float:
        input_cost = (input_tokens / 1000) * input_price_per_1k
        output_cost = (output_tokens / 1000) * output_price_per_1k
        self.value = input_cost + output_cost
        return self.value

    @property
    def __name__(self):
        return "Cost"

# 新增：评估阶段枚举
class EvaluationStage(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    BENCHMARK = "benchmark"


# 新增：指标配置数据类
@dataclass
class MetricConfig:
    """指标配置数据类"""
    name: str
    metric: Any
    weight: float = 1.0
    enabled: bool = True
    description: str = ""

class RAGEvaluationMetrics:
    """RAG系统评测指标配置"""

    def __init__(self, threshold=0.7, model=None,
                 stage: EvaluationStage = EvaluationStage.TESTING):
        """
        threshold: 指标合格阈值（0-1）
        model: 评测用的LLM模型
        stage: 评估阶段，影响指标严格程度
        """
        self.threshold = threshold
        self.model = model
        self.stage = stage

        # 根据阶段调整阈值
        self._adjust_thresholds_by_stage()

        # 性能指标
        self.performance_metrics = {
            'latency': LatencyMetric(max_latency_ms=self._get_latency_threshold(), name="latency", threshold=self._get_latency_threshold()),
            'token_efficiency': TokenEfficiencyMetric(min_efficiency=0.3, name="token_efficiency", threshold=0.3),
            'cost': CostMetric(max_cost_per_query=0.02, name="cost", threshold=0.02)
        }

    def _adjust_thresholds_by_stage(self):
        """根据评估阶段调整阈值"""
        stage_multipliers = {
            EvaluationStage.DEVELOPMENT: 0.8,  # 开发阶段更宽松
            EvaluationStage.TESTING: 1.0,  # 测试阶段标准
            EvaluationStage.PRODUCTION: 1.2,  # 生产阶段更严格
            EvaluationStage.BENCHMARK: 1.5  # 基准测试最严格
        }
        multiplier = stage_multipliers.get(self.stage, 1.0)
        self.effective_threshold = min(1.0, self.threshold * multiplier)

    def _get_latency_threshold(self) -> float:
        """根据阶段获取延迟阈值"""
        latency_thresholds = {
            EvaluationStage.DEVELOPMENT: 5000,  # 5秒
            EvaluationStage.TESTING: 3000,  # 3秒
            EvaluationStage.PRODUCTION: 2000,  # 2秒
            EvaluationStage.BENCHMARK: 1000  # 1秒
        }
        return latency_thresholds.get(self.stage, 3000)

    def get_retriever_metrics(self):
        """获取检索器相关指标"""
        return {
            'contextual_relevancy': ContextualRelevancyMetric(
                threshold=0.3,  # 降低阈值以适应实际情况
                model=self.model,
                include_reason=True  # 包含评分理由
            ),
            'contextual_precision': ContextualPrecisionMetric(
                threshold=0.3,  # 降低阈值以适应实际情况
                model=self.model,
                include_reason= True
            ),
            'contextual_recall': ContextualRecallMetric(
                threshold=0.3,  # 降低阈值以适应实际情况
                model=self.model,
                include_reason= True
            )
        }

    def get_generator_metrics(self):
        """获取生成器相关指标 原有的2个指标 + 幻觉检测 + 连贯性"""
        return {
            'answer_relevancy': AnswerRelevancyMetric(
                threshold=self.effective_threshold,
                model=self.model,
                include_reason=True
            ),
            'faithfulness': FaithfulnessMetric(
                threshold=min(0.95,self.effective_threshold*1.1),# 忠实度要求更高
                model=self.model,
                include_reason=True
            ),
            'hallucination': HallucinationMetric(
                threshold=0.9,  # 幻觉检测固定高阈值
                model=self.model,
                include_reason=True
            )
            # 'coherence': CoherenceMetric(
            #     threshold=self.effective_threshold * 0.9,
            #     model=self.model
            # )
        }

    def get_quality_metrics(self) -> Dict[str, Any]:
        """获取质量相关指标（新增）"""
        return {
            'toxicity': ToxicityMetric(
                threshold=0.1,  # 毒性检测，越低越好
                model=self.model
            ),
            'bias': BiasMetric(
                threshold=0.2,  # 偏见检测，越低越好
                model=self.model
            )
        }
    def get_performance_metrics(self) -> Dict[str, PerformanceMetric]:
        """获取性能指标"""
        return self.performance_metrics

    def get_all_metrics(self,include_performance:bool=True):
        """获取所有指标"""
        metrics = {}

        # 质量指标
        metrics.update(self.get_retriever_metrics())
        metrics.update(self.get_generator_metrics())
        metrics.update(self.get_quality_metrics())

        # 性能指标
        if include_performance:
            metrics.update(self.get_performance_metrics())

        return metrics

    def get_metrics_by_category(self) -> Dict[str, Dict[str, Any]]:
        """按类别获取指标"""
        return {
            'retrieval': self.get_retriever_metrics(),
            'generation': self.get_generator_metrics(),
            'quality': self.get_quality_metrics(),
            'performance': self.get_performance_metrics()
        }

    def get_weighted_metrics(self) -> List[MetricConfig]:
        """获取带权重的指标配置"""
        weights = {
            'contextual_relevancy': 1.2,  # 检索相关性最重要
            'answer_relevancy': 1.1,  # 答案相关性次重要
            'faithfulness': 1.3,  # 事实一致性最重要
            'hallucination': 1.2,  # 幻觉检测很重要
            'latency': 0.8,  # 性能指标权重稍低
            'contextual_precision': 0.9,
            'contextual_recall': 0.9,
            'coherence': 0.7,
            'toxicity': 1.0,
            'bias': 1.0,
            'token_efficiency': 0.6,
            'cost': 0.5
        }

        all_metrics = self.get_all_metrics()
        weighted_configs = []

        for name, metric in all_metrics.items():
            weight = weights.get(name, 1.0)
            description = self._get_metric_description(name)

            config = MetricConfig(
                name=name,
                metric=metric,
                weight=weight,
                enabled=True,
                description=description
            )
            weighted_configs.append(config)

        return weighted_configs

    def _get_metric_description(self, metric_name: str) -> str:
        """获取指标描述"""
        descriptions = {
            'contextual_relevancy': '检索到的上下文与查询的相关程度',
            'contextual_precision': '检索到的相关上下文占所有检索上下文的比例',
            'contextual_recall': '检索到的相关上下文占所有相关上下文的比例',
            'answer_relevancy': '生成的答案与查询的相关程度',
            'faithfulness': '生成的答案与检索上下文的忠实程度',
            'hallucination': '生成答案中包含幻觉（虚构信息）的程度',
            'coherence': '生成答案的逻辑连贯性和流畅性',
            'toxicity': '生成答案中包含有害或冒犯性内容的程度',
            'bias': '生成答案中存在的偏见程度',
            'latency': '从查询到响应的总延迟时间（毫秒）',
            'token_efficiency': '答案token数占总token数的比例',
            'cost': '单次查询的估算成本（美元）'
        }
        return descriptions.get(metric_name, '')

    def get_metrics_for_stage(self, stage: Union[str, EvaluationStage]) -> Dict[str, Any]:
        """获取特定阶段的指标配置"""
        if isinstance(stage, str):
            stage = EvaluationStage(stage)

        # 临时创建该阶段的指标配置
        temp_config = RAGEvaluationMetrics(
            threshold=self.threshold,
            model=self.model,
            stage=stage
        )
        return temp_config.get_all_metrics()

    def calculate_overall_score(self, metric_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """计算总体评分 - 新增功能"""
        weighted_configs = self.get_weighted_metrics()

        total_weight = 0
        weighted_sum = 0
        category_scores = {}

        for config in weighted_configs:
            if not config.enabled:
                continue

            metric_name = config.name
            if metric_name not in metric_results:
                continue

            result = metric_results[metric_name]
            score = result.get('score', 0)

            # 对于性能指标（latency, cost），分数需要转换
            if metric_name in ['latency', 'cost']:
                metric_obj = self.performance_metrics.get(metric_name)
                if metric_obj and metric_obj.threshold:
                    # 归一化到0-1，越低越好
                    normalized_score = max(0, 1 - (score / metric_obj.threshold))
                    score = normalized_score

            weighted_sum += score * config.weight
            total_weight += config.weight

            # 按类别统计
            category = self._get_metric_category(metric_name)
            if category not in category_scores:
                category_scores[category] = {'total': 0, 'weight': 0}
            category_scores[category]['total'] += score * config.weight
            category_scores[category]['weight'] += config.weight

        # 计算总体分数
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0

        # 计算类别分数
        category_final_scores = {}
        for category, data in category_scores.items():
            if data['weight'] > 0:
                category_final_scores[category] = data['total'] / data['weight']

        return {
            'overall_score': overall_score,
            'category_scores': category_final_scores,
            'weighted_configs': [c.__dict__ for c in weighted_configs if c.enabled]
        }

    def _get_metric_category(self, metric_name: str) -> str:
        """获取指标所属类别"""
        category_map = {
            'contextual_relevancy': 'retrieval',
            'contextual_precision': 'retrieval',
            'contextual_recall': 'retrieval',
            'answer_relevancy': 'generation',
            'faithfulness': 'generation',
            'hallucination': 'generation',
            'coherence': 'generation',
            'toxicity': 'quality',
            'bias': 'quality',
            'latency': 'performance',
            'token_efficiency': 'performance',
            'cost': 'performance'
        }
        return category_map.get(metric_name, 'other')


# 使用示例
# if __name__ == "__main__":
#     # 基本用法（保持向后兼容）
#     metrics = RAGEvaluationMetrics(threshold=0.7, model="gpt-4")
#
#     # 获取所有指标
#     all_metrics = metrics.get_all_metrics()
#     print(f"总指标数: {len(all_metrics)}")
#
#     # 按类别获取
#     by_category = metrics.get_metrics_by_category()
#     print(f"检索指标: {len(by_category['retrieval'])}个")
#     print(f"生成指标: {len(by_category['generation'])}个")
#
#     # 获取带权重的配置
#     weighted = metrics.get_weighted_metrics()
#     print(f"加权指标配置: {len(weighted)}个")