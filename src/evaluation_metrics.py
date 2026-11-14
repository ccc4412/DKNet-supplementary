# -*- coding: utf-8 -*-
"""
评估指标计算模块
实现P@K、R@K、F1@K、Diversity和Timeliness等评估指标
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional, Union
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from loguru import logger
import math
from datetime import datetime, timedelta

class EvaluationMetrics:
    """评估指标计算器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化评估指标计算器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 获取评估配置，提供默认值
        eval_config = self.config.get('experiment', {}).get('evaluation', {})
        self.k_values = eval_config.get('k_values', [1, 3, 5, 10])
        self.diversity_metrics = eval_config.get('diversity_metrics', ['intra_list', 'coverage', 'entropy'])
        self.temporal_metrics = eval_config.get('temporal_metrics', ['timeliness', 'novelty', 'consistency'])
        self.diversity_threshold = eval_config.get('diversity_threshold', 0.7)
        
        logger.info(f"评估指标计算器初始化完成，K值: {self.k_values}")
    
    def precision_at_k(self, predicted: List[str], ground_truth: List[str], k: int) -> float:
        """计算P@K"""
        if not predicted or k <= 0:
            return 0.0
        
        predicted_k = predicted[:k]
        ground_truth_set = set(ground_truth)
        
        relevant_count = sum(1 for item in predicted_k if item in ground_truth_set)
        precision = relevant_count / len(predicted_k)
        
        return precision
    
    def recall_at_k(self, predicted: List[str], ground_truth: List[str], k: int) -> float:
        """计算R@K"""
        if not ground_truth or k <= 0:
            return 0.0
        
        predicted_k = predicted[:k]
        predicted_set = set(predicted_k)
        ground_truth_set = set(ground_truth)
        
        relevant_count = len(predicted_set & ground_truth_set)
        recall = relevant_count / len(ground_truth_set)
        
        return recall
    
    def f1_at_k(self, predicted: List[str], ground_truth: List[str], k: int) -> float:
        """计算F1@K"""
        precision = self.precision_at_k(predicted, ground_truth, k)
        recall = self.recall_at_k(predicted, ground_truth, k)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def mean_average_precision(self, predicted: List[str], ground_truth: List[str]) -> float:
        """计算MAP (Mean Average Precision)"""
        if not predicted or not ground_truth:
            return 0.0
        
        ground_truth_set = set(ground_truth)
        relevant_count = 0
        precision_sum = 0.0
        
        for i, item in enumerate(predicted):
            if item in ground_truth_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        if relevant_count == 0:
            return 0.0
        
        map_score = precision_sum / len(ground_truth_set)
        return map_score
    
    def ndcg_at_k(self, predicted: List[str], ground_truth: List[str], k: int,
                  relevance_scores: Optional[Dict[str, float]] = None) -> float:
        """计算NDCG@K"""
        if not predicted or k <= 0:
            return 0.0
        
        predicted_k = predicted[:k]
        
        # 如果没有提供相关性分数，使用二元相关性
        if relevance_scores is None:
            ground_truth_set = set(ground_truth)
            relevance_scores = {item: 1.0 if item in ground_truth_set else 0.0 for item in predicted_k}
        
        # 计算DCG
        dcg = 0.0
        for i, item in enumerate(predicted_k):
            relevance = relevance_scores.get(item, 0.0)
            dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # 计算IDCG (理想DCG)
        ideal_relevances = sorted([relevance_scores.get(item, 0.0) for item in ground_truth], reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances):
            idcg += relevance / math.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return ndcg
    
    def diversity_score(self, keywords: List[str], 
                       embeddings: Dict[str, np.ndarray],
                       metric: str = 'intra_list') -> float:
        """计算多样性分数"""
        if len(keywords) < 2:
            return 1.0
        
        if metric == 'intra_list':
            return self._intra_list_diversity(keywords, embeddings)
        elif metric == 'coverage':
            return self._coverage_diversity(keywords, embeddings)
        elif metric == 'entropy':
            return self._entropy_diversity(keywords)
        else:
            return self._intra_list_diversity(keywords, embeddings)
    
    def _intra_list_diversity(self, keywords: List[str], 
                             embeddings: Dict[str, np.ndarray]) -> float:
        """计算列表内多样性"""
        similarities = []
        
        for i, word1 in enumerate(keywords):
            for j, word2 in enumerate(keywords):
                if i < j and word1 in embeddings and word2 in embeddings:
                    sim = cosine_similarity(
                        embeddings[word1].reshape(1, -1),
                        embeddings[word2].reshape(1, -1)
                    )[0][0]
                    similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        # 多样性 = 1 - 平均相似度
        diversity = 1.0 - np.mean(similarities)
        return max(0.0, diversity)
    
    def _coverage_diversity(self, keywords: List[str], 
                           embeddings: Dict[str, np.ndarray]) -> float:
        """计算覆盖多样性"""
        # 这里需要一个参考词汇集合，暂时使用关键词本身
        return self._intra_list_diversity(keywords, embeddings)
    
    def _entropy_diversity(self, keywords: List[str]) -> float:
        """计算熵多样性"""
        if not keywords:
            return 0.0
        
        # 基于词长度的简单熵计算
        length_counts = Counter([len(word) for word in keywords])
        total = len(keywords)
        
        entropy = 0.0
        for count in length_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        # 归一化
        max_entropy = math.log2(len(length_counts)) if length_counts else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def timeliness_score(self, keywords: List[str], 
                        timestamps: List[datetime],
                        current_time: Optional[datetime] = None,
                        decay_factor: float = 0.1) -> float:
        """计算时效性分数"""
        if not keywords or not timestamps:
            return 0.0
        
        if current_time is None:
            current_time = datetime.now()
        
        timeliness_scores = []
        
        for i, keyword in enumerate(keywords):
            if i < len(timestamps):
                time_diff = (current_time - timestamps[i]).total_seconds() / 3600  # 小时
                timeliness = math.exp(-decay_factor * time_diff)
                timeliness_scores.append(timeliness)
        
        if not timeliness_scores:
            return 0.0
        
        return np.mean(timeliness_scores)
    
    def novelty_score(self, predicted: List[str], 
                     historical_keywords: List[List[str]],
                     time_weights: Optional[List[float]] = None) -> float:
        """计算新颖性分数"""
        if not predicted or not historical_keywords:
            return 1.0
        
        # 构建历史关键词集合
        historical_set = set()
        if time_weights is None:
            time_weights = [1.0] * len(historical_keywords)
        
        for i, hist_keywords in enumerate(historical_keywords):
            weight = time_weights[i] if i < len(time_weights) else 1.0
            for keyword in hist_keywords:
                historical_set.add((keyword, weight))
        
        # 计算新颖性
        novelty_scores = []
        for keyword in predicted:
            # 检查是否在历史中出现过
            max_weight = 0.0
            for hist_keyword, weight in historical_set:
                if keyword == hist_keyword:
                    max_weight = max(max_weight, weight)
            
            # 新颖性 = 1 - 历史权重
            novelty = 1.0 - max_weight
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores) if novelty_scores else 1.0
    
    def coherence_score(self, keywords: List[str], 
                       embeddings: Dict[str, np.ndarray],
                       method: str = 'centroid') -> float:
        """计算一致性分数"""
        if len(keywords) < 2:
            return 1.0
        
        valid_embeddings = [embeddings[word] for word in keywords if word in embeddings]
        if len(valid_embeddings) < 2:
            return 0.0
        
        if method == 'centroid':
            # 基于质心的一致性
            centroid = np.mean(valid_embeddings, axis=0)
            similarities = []
            for embedding in valid_embeddings:
                sim = cosine_similarity(
                    embedding.reshape(1, -1),
                    centroid.reshape(1, -1)
                )[0][0]
                similarities.append(sim)
            return np.mean(similarities)
        
        elif method == 'pairwise':
            # 基于成对相似度的一致性
            similarities = []
            for i, emb1 in enumerate(valid_embeddings):
                for j, emb2 in enumerate(valid_embeddings):
                    if i < j:
                        sim = cosine_similarity(
                            emb1.reshape(1, -1),
                            emb2.reshape(1, -1)
                        )[0][0]
                        similarities.append(sim)
            return np.mean(similarities) if similarities else 0.0
        
        else:
            return self.coherence_score(keywords, embeddings, 'centroid')
    
    def evaluate_single_result(self, predicted: List[str], 
                              ground_truth: List[str],
                              embeddings: Optional[Dict[str, np.ndarray]] = None,
                              timestamps: Optional[List[datetime]] = None,
                              historical_keywords: Optional[List[List[str]]] = None) -> Dict[str, float]:
        """评估单个结果"""
        results = {}
        
        # 基础指标
        for k in self.k_values:
            results[f'P@{k}'] = self.precision_at_k(predicted, ground_truth, k)
            results[f'R@{k}'] = self.recall_at_k(predicted, ground_truth, k)
            results[f'F1@{k}'] = self.f1_at_k(predicted, ground_truth, k)
            results[f'NDCG@{k}'] = self.ndcg_at_k(predicted, ground_truth, k)
        
        # MAP
        results['MAP'] = self.mean_average_precision(predicted, ground_truth)
        
        # 多样性指标
        if embeddings:
            results['Diversity'] = self.diversity_score(predicted, embeddings)
            results['Coherence'] = self.coherence_score(predicted, embeddings)
        
        # 时效性指标
        if timestamps:
            results['Timeliness'] = self.timeliness_score(predicted, timestamps)
        
        # 新颖性指标
        if historical_keywords:
            results['Novelty'] = self.novelty_score(predicted, historical_keywords)
        
        return results
    
    def evaluate_batch_results(self, batch_predicted: List[List[str]],
                              batch_ground_truth: List[List[str]],
                              batch_embeddings: Optional[List[Dict[str, np.ndarray]]] = None,
                              batch_timestamps: Optional[List[List[datetime]]] = None,
                              batch_historical: Optional[List[List[List[str]]]] = None) -> Dict[str, float]:
        """批量评估结果"""
        all_results = []
        
        for i, (predicted, ground_truth) in enumerate(zip(batch_predicted, batch_ground_truth)):
            embeddings = batch_embeddings[i] if batch_embeddings else None
            timestamps = batch_timestamps[i] if batch_timestamps else None
            historical = batch_historical[i] if batch_historical else None
            
            result = self.evaluate_single_result(
                predicted, ground_truth, embeddings, timestamps, historical
            )
            all_results.append(result)
        
        # 计算平均值
        avg_results = {}
        if all_results:
            for metric in all_results[0].keys():
                values = [result[metric] for result in all_results if metric in result]
                avg_results[metric] = np.mean(values) if values else 0.0
                avg_results[f'{metric}_std'] = np.std(values) if values else 0.0
        
        logger.info(f"批量评估完成，处理了 {len(batch_predicted)} 个结果")
        
        return avg_results
    
    def statistical_significance_test(self, results1: List[float], 
                                    results2: List[float],
                                    test_type: str = 'paired_t') -> Dict[str, float]:
        """统计显著性检验"""
        from scipy import stats
        
        if len(results1) != len(results2) or len(results1) < 2:
            return {'p_value': 1.0, 'statistic': 0.0, 'significant': False}
        
        if test_type == 'paired_t':
            statistic, p_value = stats.ttest_rel(results1, results2)
        elif test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(results1, results2)
        elif test_type == 'mann_whitney':
            statistic, p_value = stats.mannwhitneyu(results1, results2)
        else:
            statistic, p_value = stats.ttest_rel(results1, results2)
        
        significant = p_value < 0.05
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': significant,
            'test_type': test_type
        }
    
    def correlation_analysis(self, metric1_values: List[float],
                           metric2_values: List[float],
                           method: str = 'pearson') -> Dict[str, float]:
        """相关性分析"""
        if len(metric1_values) != len(metric2_values) or len(metric1_values) < 2:
            return {'correlation': 0.0, 'p_value': 1.0}
        
        if method == 'pearson':
            correlation, p_value = pearsonr(metric1_values, metric2_values)
        elif method == 'spearman':
            correlation, p_value = spearmanr(metric1_values, metric2_values)
        else:
            correlation, p_value = pearsonr(metric1_values, metric2_values)
        
        return {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'method': method
        }
    
    def generate_evaluation_report(self, results: Dict[str, float],
                                 method_name: str = 'DKNet') -> str:
        """生成评估报告"""
        report = f"\n=== {method_name} 评估报告 ===\n"
        
        # 基础指标
        report += "\n基础指标:\n"
        for k in self.k_values:
            if f'P@{k}' in results:
                report += f"  P@{k}: {results[f'P@{k}']:.4f}\n"
            if f'R@{k}' in results:
                report += f"  R@{k}: {results[f'R@{k}']:.4f}\n"
            if f'F1@{k}' in results:
                report += f"  F1@{k}: {results[f'F1@{k}']:.4f}\n"
        
        # 高级指标
        report += "\n高级指标:\n"
        if 'MAP' in results:
            report += f"  MAP: {results['MAP']:.4f}\n"
        if 'Diversity' in results:
            report += f"  Diversity: {results['Diversity']:.4f}\n"
        if 'Coherence' in results:
            report += f"  Coherence: {results['Coherence']:.4f}\n"
        if 'Timeliness' in results:
            report += f"  Timeliness: {results['Timeliness']:.4f}\n"
        if 'Novelty' in results:
            report += f"  Novelty: {results['Novelty']:.4f}\n"
        
        return report
    
    def export_results_to_csv(self, results: Dict[str, Dict[str, float]], 
                              filepath: str) -> None:
        """导出结果到CSV文件"""
        df = pd.DataFrame(results).T
        df.to_csv(filepath, index=True)
        logger.info(f"评估结果已导出到: {filepath}")
    
    def compare_methods(self, method_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict]:
        """比较不同方法的性能"""
        comparison = {}
        
        # 获取所有指标
        all_metrics = set()
        for results in method_results.values():
            all_metrics.update(results.keys())
        
        for metric in all_metrics:
            metric_values = {}
            for method, results in method_results.items():
                if metric in results:
                    metric_values[method] = results[metric]
            
            if len(metric_values) > 1:
                # 找到最佳方法
                best_method = max(metric_values.keys(), key=lambda x: metric_values[x])
                best_value = metric_values[best_method]
                
                # 计算相对改进
                improvements = {}
                for method, value in metric_values.items():
                    if method != best_method and value > 0:
                        improvement = (best_value - value) / value * 100
                        improvements[method] = improvement
                
                comparison[metric] = {
                    'best_method': best_method,
                    'best_value': best_value,
                    'all_values': metric_values,
                    'improvements': improvements
                }
        
        return comparison


class MetricAggregator:
    """指标聚合器"""
    
    def __init__(self):
        self.results_history = []
    
    def add_results(self, results: Dict[str, float], metadata: Optional[Dict] = None):
        """添加评估结果"""
        entry = {
            'results': results,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        }
        self.results_history.append(entry)
    
    def get_aggregated_results(self, aggregation_method: str = 'mean') -> Dict[str, float]:
        """获取聚合结果"""
        if not self.results_history:
            return {}
        
        # 收集所有指标
        all_metrics = set()
        for entry in self.results_history:
            all_metrics.update(entry['results'].keys())
        
        aggregated = {}
        
        for metric in all_metrics:
            values = []
            for entry in self.results_history:
                if metric in entry['results']:
                    values.append(entry['results'][metric])
            
            if values:
                if aggregation_method == 'mean':
                    aggregated[metric] = np.mean(values)
                elif aggregation_method == 'median':
                    aggregated[metric] = np.median(values)
                elif aggregation_method == 'max':
                    aggregated[metric] = np.max(values)
                elif aggregation_method == 'min':
                    aggregated[metric] = np.min(values)
                else:
                    aggregated[metric] = np.mean(values)
                
                # 添加标准差
                aggregated[f'{metric}_std'] = np.std(values)
        
        return aggregated
    
    def get_trend_analysis(self, metric: str, window_size: int = 5) -> Dict:
        """获取趋势分析"""
        values = []
        timestamps = []
        
        for entry in self.results_history:
            if metric in entry['results']:
                values.append(entry['results'][metric])
                timestamps.append(entry['timestamp'])
        
        if len(values) < 2:
            return {'trend': 'insufficient_data'}
        
        # 计算移动平均
        if len(values) >= window_size:
            moving_avg = []
            for i in range(window_size - 1, len(values)):
                avg = np.mean(values[i - window_size + 1:i + 1])
                moving_avg.append(avg)
        else:
            moving_avg = values
        
        # 计算趋势
        if len(moving_avg) >= 2:
            trend_slope = (moving_avg[-1] - moving_avg[0]) / len(moving_avg)
            if trend_slope > 0.001:
                trend = 'improving'
            elif trend_slope < -0.001:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'trend_slope': trend_slope if 'trend_slope' in locals() else 0,
            'values': values,
            'moving_average': moving_avg,
            'timestamps': timestamps
        }
    
    def clear_history(self):
        """清空历史记录"""
        self.results_history.clear()
        logger.info("评估结果历史已清空")