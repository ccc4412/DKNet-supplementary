# -*- coding: utf-8 -*-
"""
自适应权重融合模块
实现概念漂移检测和权重动态调整机制

增强功能：
- Contextual Bandit (LinUCB) 实现
- Context Features 计算 (drift index, volume delta, precision proxy)
- Reward Function (coverage gain + precision@K 加权和)
- Thompson Sampling 作为替代方案
- Exploration Budget (K_exp) 支持
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Set, Union
from collections import deque, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy import stats
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class AdaptiveFusionAlgorithm:
    """自适应权重融合算法"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 自适应权重参数
        self.alpha = config['model']['adaptive']['alpha_init']  # 初始权重
        self.beta = config['model']['adaptive']['beta']  # 调节参数
        self.bias = config['model']['adaptive']['bias']  # 偏置项
        self.drift_threshold = config['model']['adaptive']['drift_threshold']  # 漂移阈值
        
        # 历史数据缓存
        self.history_window_size = 100  # 历史窗口大小
        self.kg_scores_history = deque(maxlen=self.history_window_size)
        self.llm_scores_history = deque(maxlen=self.history_window_size)
        self.fusion_scores_history = deque(maxlen=self.history_window_size)
        
        # 概念漂移检测
        self.drift_detector = ConceptDriftDetector(config)
        
        # 权重历史
        self.alpha_history = deque(maxlen=self.history_window_size)
        self.alpha_history.append(self.alpha)
        
        logger.info(f"自适应融合算法初始化完成，初始权重: {self.alpha}")
    
    def calculate_adaptive_weight(self, kg_scores: np.ndarray, 
                                llm_scores: np.ndarray,
                                current_context: Dict = None) -> float:
        """计算自适应权重"""
        # 检测概念漂移
        drift_score = self.drift_detector.detect_drift(
            kg_scores, llm_scores, current_context
        )
        
        # 计算性能差异
        performance_diff = self._calculate_performance_difference(
            kg_scores, llm_scores
        )
        
        # 更新权重
        new_alpha = self._update_weight(drift_score, performance_diff)
        
        # 记录历史
        self.alpha_history.append(new_alpha)
        self.kg_scores_history.append(kg_scores.copy())
        self.llm_scores_history.append(llm_scores.copy())
        
        self.alpha = new_alpha
        
        logger.debug(f"权重更新: {new_alpha:.4f}, 漂移分数: {drift_score:.4f}")
        
        return new_alpha
    
    def _calculate_performance_difference(self, kg_scores: np.ndarray, 
                                        llm_scores: np.ndarray) -> float:
        """计算KG和LLM性能差异"""
        if len(kg_scores) == 0 or len(llm_scores) == 0:
            return 0.0
        
        # 归一化分数
        kg_norm = kg_scores / (np.max(kg_scores) + 1e-8)
        llm_norm = llm_scores / (np.max(llm_scores) + 1e-8)
        
        # 计算平均性能
        kg_performance = np.mean(kg_norm)
        llm_performance = np.mean(llm_norm)
        
        # 性能差异
        performance_diff = llm_performance - kg_performance
        
        return performance_diff
    
    def _update_weight(self, drift_score: float, performance_diff: float) -> float:
        """更新权重"""
        # 基于漂移分数的权重调整
        drift_adjustment = self.beta * drift_score
        
        # 基于性能差异的权重调整
        performance_adjustment = 0.1 * performance_diff
        
        # 计算新权重
        new_alpha = self.alpha + drift_adjustment + performance_adjustment + self.bias
        
        # 权重约束在[0, 1]范围内
        new_alpha = np.clip(new_alpha, 0.0, 1.0)
        
        return new_alpha
    
    def adaptive_fusion_score(self, kg_score: float, llm_score: float, 
                            alpha: Optional[float] = None) -> float:
        """自适应融合分数计算"""
        if alpha is None:
            alpha = self.alpha
        
        # 融合公式: S_fusion = α * S_KG + (1-α) * S_LLM
        fusion_score = alpha * kg_score + (1 - alpha) * llm_score
        
        return fusion_score
    
    def batch_adaptive_fusion(self, kg_scores: np.ndarray, 
                            llm_scores: np.ndarray,
                            update_weights: bool = True) -> Tuple[np.ndarray, float]:
        """批量自适应融合"""
        if len(kg_scores) != len(llm_scores):
            raise ValueError("KG分数和LLM分数长度不匹配")
        
        # 更新权重（如果需要）
        if update_weights:
            current_alpha = self.calculate_adaptive_weight(kg_scores, llm_scores)
        else:
            current_alpha = self.alpha
        
        # 批量计算融合分数
        fusion_scores = current_alpha * kg_scores + (1 - current_alpha) * llm_scores
        
        # 记录融合分数历史
        self.fusion_scores_history.append(fusion_scores.copy())
        
        return fusion_scores, current_alpha
    
    def get_weight_statistics(self) -> Dict:
        """获取权重统计信息"""
        if not self.alpha_history:
            return {}
        
        alpha_values = list(self.alpha_history)
        
        stats = {
            'current_alpha': self.alpha,
            'mean_alpha': np.mean(alpha_values),
            'std_alpha': np.std(alpha_values),
            'min_alpha': np.min(alpha_values),
            'max_alpha': np.max(alpha_values),
            'alpha_trend': self._calculate_trend(alpha_values)
        }
        
        return stats
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 3:
            return 'insufficient_data'
        
        # 使用线性回归计算趋势
        x = np.arange(len(values))
        slope, _, _, p_value, _ = stats.linregress(x, values)
        
        if p_value > 0.05:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def reset_weights(self):
        """重置权重"""
        self.alpha = self.config['model']['adaptive']['alpha_init']
        self.alpha_history.clear()
        self.alpha_history.append(self.alpha)
        logger.info("权重已重置")


class ConceptDriftDetector:
    """概念漂移检测器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.window_size = 50  # 检测窗口大小
        self.reference_window = deque(maxlen=self.window_size)
        self.current_window = deque(maxlen=self.window_size)
        
        # 漂移检测方法
        self.detection_methods = {
            'statistical': self._statistical_drift_test,
            'distribution': self._distribution_drift_test,
            'clustering': self._clustering_drift_test
        }
        
        logger.info("概念漂移检测器初始化完成")
    
    def detect_drift(self, kg_scores: np.ndarray, llm_scores: np.ndarray,
                    context: Dict = None) -> float:
        """检测概念漂移"""
        # 合并分数作为特征
        combined_scores = np.column_stack([kg_scores, llm_scores])
        
        # 更新窗口
        for score_pair in combined_scores:
            if len(self.reference_window) < self.window_size:
                self.reference_window.append(score_pair)
            else:
                self.current_window.append(score_pair)
        
        # 如果当前窗口未满，返回0
        if len(self.current_window) < self.window_size // 2:
            return 0.0
        
        # 执行多种漂移检测
        drift_scores = []
        
        for method_name, method_func in self.detection_methods.items():
            try:
                drift_score = method_func()
                drift_scores.append(drift_score)
            except Exception as e:
                logger.warning(f"漂移检测方法 {method_name} 失败: {e}")
                drift_scores.append(0.0)
        
        # 综合漂移分数
        final_drift_score = np.mean(drift_scores)
        
        # 如果检测到显著漂移，更新参考窗口
        if final_drift_score > self.config['model']['adaptive']['drift_threshold']:
            self._update_reference_window()
        
        return final_drift_score
    
    def _statistical_drift_test(self) -> float:
        """统计检验漂移检测"""
        if len(self.reference_window) == 0 or len(self.current_window) == 0:
            return 0.0
        
        ref_data = np.array(list(self.reference_window))
        cur_data = np.array(list(self.current_window))
        
        # 对每个维度进行t检验
        drift_scores = []
        
        for dim in range(ref_data.shape[1]):
            try:
                # Welch's t-test
                t_stat, p_value = stats.ttest_ind(
                    ref_data[:, dim], cur_data[:, dim], equal_var=False
                )
                
                # 将p值转换为漂移分数
                drift_score = 1.0 - p_value
                drift_scores.append(drift_score)
                
            except Exception:
                drift_scores.append(0.0)
        
        return np.mean(drift_scores)
    
    def _distribution_drift_test(self) -> float:
        """分布漂移检测"""
        if len(self.reference_window) == 0 or len(self.current_window) == 0:
            return 0.0
        
        ref_data = np.array(list(self.reference_window))
        cur_data = np.array(list(self.current_window))
        
        # 使用KS检验
        drift_scores = []
        
        for dim in range(ref_data.shape[1]):
            try:
                ks_stat, p_value = stats.ks_2samp(
                    ref_data[:, dim], cur_data[:, dim]
                )
                
                # KS统计量作为漂移分数
                drift_scores.append(ks_stat)
                
            except Exception:
                drift_scores.append(0.0)
        
        return np.mean(drift_scores)
    
    def _clustering_drift_test(self) -> float:
        """聚类漂移检测"""
        if len(self.reference_window) == 0 or len(self.current_window) == 0:
            return 0.0
        
        ref_data = np.array(list(self.reference_window))
        cur_data = np.array(list(self.current_window))
        
        try:
            # 合并数据
            all_data = np.vstack([ref_data, cur_data])
            
            # K-means聚类
            n_clusters = min(3, len(all_data) // 10 + 1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(all_data)
            
            # 计算参考窗口和当前窗口的聚类分布
            ref_labels = labels[:len(ref_data)]
            cur_labels = labels[len(ref_data):]
            
            # 计算分布差异
            ref_dist = np.bincount(ref_labels, minlength=n_clusters) / len(ref_labels)
            cur_dist = np.bincount(cur_labels, minlength=n_clusters) / len(cur_labels)
            
            # 使用JS散度计算分布差异
            drift_score = self._js_divergence(ref_dist, cur_dist)
            
            return drift_score
            
        except Exception:
            return 0.0
    
    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算JS散度"""
        # 避免除零
        p = p + 1e-8
        q = q + 1e-8
        
        # 归一化
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # 计算JS散度
        m = 0.5 * (p + q)
        js_div = 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)
        
        return js_div
    
    def _update_reference_window(self):
        """更新参考窗口"""
        # 将当前窗口的数据移到参考窗口
        current_data = list(self.current_window)
        self.reference_window.clear()
        
        for data in current_data:
            self.reference_window.append(data)
        
        self.current_window.clear()
        
        logger.info("检测到概念漂移，已更新参考窗口")
    
    def get_drift_statistics(self) -> Dict:
        """获取漂移统计信息"""
        stats = {
            'reference_window_size': len(self.reference_window),
            'current_window_size': len(self.current_window),
            'detection_methods': list(self.detection_methods.keys())
        }
        
        return stats


class DriftScoreCalculator:
    """漂移分数计算器"""
    
    @staticmethod
    def calculate_drift_score(historical_performance: List[float],
                            current_performance: float,
                            window_size: int = 10) -> float:
        """计算漂移分数
        
        Args:
            historical_performance: 历史性能列表
            current_performance: 当前性能
            window_size: 窗口大小
            
        Returns:
            漂移分数 (0-1之间)
        """
        if len(historical_performance) < window_size:
            return 0.0
        
        # 取最近的窗口
        recent_performance = historical_performance[-window_size:]
        
        # 计算历史平均性能
        historical_mean = np.mean(recent_performance)
        historical_std = np.std(recent_performance)
        
        if historical_std == 0:
            return 0.0
        
        # 计算标准化差异
        normalized_diff = abs(current_performance - historical_mean) / historical_std
        
        # 转换为0-1范围的漂移分数
        drift_score = min(normalized_diff / 3.0, 1.0)  # 3个标准差作为最大值
        
        return drift_score


# ==================== Context Features 计算 ====================

class ContextFeatureExtractor:
    """上下文特征提取器
    
    从当前数据流状态中提取context features，供bandit算法使用。
    包含：
    - drift_index: 概念漂移指数（来自ConceptDriftDetector）
    - volume_delta: 数据量变化率
    - precision_proxy: 精度代理指标（基于历史融合分数分布）
    """
    
    def __init__(self, drift_detector: 'ConceptDriftDetector' = None):
        self.drift_detector = drift_detector
        self.prev_volume = 0
        self.score_history = deque(maxlen=100)
        self.feature_dim = 3  # drift_index, volume_delta, precision_proxy
    
    def extract_features(self,
                         kg_scores: np.ndarray,
                         llm_scores: np.ndarray,
                         drift_score: float = 0.0) -> np.ndarray:
        """提取上下文特征向量
        
        Args:
            kg_scores: KG模块分数
            llm_scores: LLM模块分数
            drift_score: 概念漂移分数（可从ConceptDriftDetector获取）
            
        Returns:
            特征向量 shape=(3,)
        """
        # Feature 1: Drift Index
        drift_index = np.clip(drift_score, 0.0, 1.0)
        
        # Feature 2: Volume Delta (数据量变化率)
        current_volume = len(kg_scores) + len(llm_scores)
        if self.prev_volume > 0:
            volume_delta = (current_volume - self.prev_volume) / self.prev_volume
        else:
            volume_delta = 0.0
        volume_delta = np.clip(volume_delta, -1.0, 1.0)
        self.prev_volume = current_volume
        
        # Feature 3: Precision Proxy
        # 基于融合分数的离散程度来估计精度
        if len(kg_scores) > 0 and len(llm_scores) > 0:
            fusion = 0.5 * kg_scores + 0.5 * llm_scores
            self.score_history.extend(fusion.tolist())
        
        if len(self.score_history) >= 5:
            recent_scores = list(self.score_history)[-20:]
            # 分数越高且越集中，precision proxy越高
            mean_score = np.mean(recent_scores)
            std_score = np.std(recent_scores) + 1e-8
            precision_proxy = np.clip(mean_score / (mean_score + std_score), 0.0, 1.0)
        else:
            precision_proxy = 0.5
        
        features = np.array([drift_index, volume_delta, precision_proxy], dtype=np.float64)
        return features


# ==================== Reward Function ====================

class RewardFunction:
    """奖励函数
    
    综合评估融合策略的效果：
    R = lambda_cov * coverage_gain + lambda_prec * precision@K
    
    - coverage_gain: 新覆盖的实体比例
    - precision@K: top-K结果中相关结果的比例
    """
    
    def __init__(self, lambda_cov: float = 0.4, lambda_prec: float = 0.6, K: int = 10):
        """
        Args:
            lambda_cov: coverage gain的权重
            lambda_prec: precision@K的权重
            K: precision@K中的K值
        """
        self.lambda_cov = lambda_cov
        self.lambda_prec = lambda_prec
        self.K = K
        self.prev_covered_entities: Set[str] = set()
    
    def compute_reward(self,
                       recommended: List[str],
                       relevant: Set[str],
                       all_candidates: Set[str] = None) -> float:
        """计算奖励值
        
        Args:
            recommended: 推荐的候选列表 [candidate_name, ...]
            relevant: 相关实体集合
            all_candidates: 所有候选实体集合（用于计算coverage）
            
        Returns:
            奖励值 R in [0, 1]
        """
        if not recommended:
            return 0.0
        
        recommended_set = set(recommended)
        
        # Coverage Gain: 新覆盖的实体比例
        new_entities = recommended_set - self.prev_covered_entities
        if all_candidates and len(all_candidates) > 0:
            coverage_gain = len(new_entities) / len(all_candidates)
        else:
            coverage_gain = len(new_entities) / max(len(recommended), 1)
        
        # Precision@K
        top_k = recommended[:self.K]
        if relevant:
            hits = sum(1 for item in top_k if item in relevant)
            precision_at_k = hits / min(len(top_k), self.K)
        else:
            precision_at_k = 0.5  # 无标签时的默认值
        
        # 加权和
        reward = self.lambda_cov * coverage_gain + self.lambda_prec * precision_at_k
        
        # 更新已覆盖实体
        self.prev_covered_entities.update(recommended_set)
        
        return float(np.clip(reward, 0.0, 1.0))
    
    def reset_coverage(self):
        """重置覆盖历史"""
        self.prev_covered_entities.clear()


# ==================== Contextual Bandit (LinUCB) ====================

class LinUCBBandit:
    """LinUCB Contextual Bandit
    
    使用线性上置信界（LinUCB）算法自适应选择KG和LLM的融合权重。
    
    每个arm代表一个离散的alpha值（融合权重），context features作为
    线性模型的输入，UCB公式平衡exploration和exploitation。
    
    Arms: alpha in {0.0, 0.1, 0.2, ..., 1.0}（11个离散权重值）
    """
    
    def __init__(self,
                 context_dim: int = 3,
                 alpha_values: List[float] = None,
                 exploration_budget: int = 100):
        """
        Args:
            context_dim: 上下文特征维度
            alpha_values: 离散的alpha候选值列表
            exploration_budget: 总探索预算 K_exp
        """
        if alpha_values is None:
            alpha_values = [round(i * 0.1, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0
        
        self.alpha_values = alpha_values
        self.n_arms = len(alpha_values)
        self.context_dim = context_dim
        
        # Exploration budget
        self.K_exp = exploration_budget
        self.total_rounds = 0
        
        # LinUCB参数：每个arm维护一个 A_a 和 b_a
        # A_a: d x d 的矩阵，初始化为单位矩阵
        # b_a: d 维向量，初始化为零向量
        self.A = [np.eye(context_dim) for _ in range(self.n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(self.n_arms)]
        
        # 统计信息
        self.arm_pulls = [0] * self.n_arms
        self.arm_rewards = [[] for _ in range(self.n_arms)]
        
        logger.info(f"LinUCB Bandit初始化完成: {self.n_arms} arms, "
                     f"context_dim={context_dim}, K_exp={exploration_budget}")
    
    def select_arm(self, context: np.ndarray) -> int:
        """选择arm（即选择alpha值）
        
        UCB公式: alpha_a = x^T * theta_a + alpha_ucb * sqrt(x^T * A_a^{-1} * x)
        
        其中 theta_a = A_a^{-1} * b_a
        
        Args:
            context: 上下文特征向量 shape=(d,)
            
        Returns:
            选择的arm索引
        """
        self.total_rounds += 1
        
        # 如果还在exploration budget内，进行exploration
        if self.total_rounds <= self.K_exp:
            # epsilon-greedy exploration
            unexplored = [i for i in range(self.n_arms) if self.arm_pulls[i] == 0]
            if unexplored:
                return np.random.choice(unexplored)
            return np.random.randint(0, self.n_arms)
        
        # UCB选择
        ucb_values = np.zeros(self.n_arms)
        alpha_ucb = 1.0  # exploration参数
        
        for a in range(self.n_arms):
            if self.arm_pulls[a] == 0:
                ucb_values[a] = float('inf')
                continue
            
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            
            # 预测值
            pred = context @ theta
            
            # 不确定性
            uncertainty = alpha_ucb * np.sqrt(context @ A_inv @ context)
            
            ucb_values[a] = pred + uncertainty
        
        return int(np.argmax(ucb_values))
    
    def update(self, arm: int, context: np.ndarray, reward: float):
        """更新arm参数
        
        Args:
            arm: 被选择的arm索引
            context: 上下文特征向量
            reward: 观察到的奖励值
        """
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.arm_pulls[arm] += 1
        self.arm_rewards[arm].append(reward)
    
    def get_alpha(self, arm: int) -> float:
        """获取arm对应的alpha值"""
        return self.alpha_values[arm]
    
    def get_best_alpha(self) -> float:
        """获取当前最优alpha（基于平均奖励）"""
        avg_rewards = []
        for a in range(self.n_arms):
            if self.arm_rewards[a]:
                avg_rewards.append(np.mean(self.arm_rewards[a]))
            else:
                avg_rewards.append(0.0)
        best_arm = int(np.argmax(avg_rewards))
        return self.alpha_values[best_arm]
    
    def get_statistics(self) -> Dict:
        """获取bandit统计信息"""
        stats = {
            'total_rounds': self.total_rounds,
            'exploration_remaining': max(0, self.K_exp - self.total_rounds),
            'n_arms': self.n_arms,
            'arm_pulls': dict(zip(
                [f"alpha={a:.1f}" for a in self.alpha_values],
                self.arm_pulls
            )),
            'arm_avg_rewards': {}
        }
        for a in range(self.n_arms):
            key = f"alpha={self.alpha_values[a]:.1f}"
            if self.arm_rewards[a]:
                stats['arm_avg_rewards'][key] = float(np.mean(self.arm_rewards[a]))
            else:
                stats['arm_avg_rewards'][key] = None
        
        return stats


# ==================== Thompson Sampling ====================

class ThompsonSamplingBandit:
    """Thompson Sampling Bandit
    
    使用Beta分布作为先验，通过采样进行arm选择。
    每个arm维护一个Beta(alpha_param, beta_param)分布。
    
    相比LinUCB，Thompson Sampling在稀疏奖励场景下表现更好，
    且不需要context features。
    """
    
    def __init__(self,
                 alpha_values: List[float] = None,
                 exploration_budget: int = 100):
        """
        Args:
            alpha_values: 离散的alpha候选值列表
            exploration_budget: 总探索预算 K_exp
        """
        if alpha_values is None:
            alpha_values = [round(i * 0.1, 1) for i in range(11)]
        
        self.alpha_values = alpha_values
        self.n_arms = len(alpha_values)
        
        # Exploration budget
        self.K_exp = exploration_budget
        self.total_rounds = 0
        
        # Beta分布参数: 每个arm的 (alpha_param, beta_param)
        self.alpha_params = np.ones(self.n_arms)  # 成功次数 + 1
        self.beta_params = np.ones(self.n_arms)   # 失败次数 + 1
        
        # 统计信息
        self.arm_pulls = np.zeros(self.n_arms)
        self.arm_total_rewards = np.zeros(self.n_arms)
        
        logger.info(f"Thompson Sampling Bandit初始化完成: {self.n_arms} arms, "
                     f"K_exp={exploration_budget}")
    
    def select_arm(self) -> int:
        """选择arm
        
        从每个arm的Beta后验中采样，选择采样值最大的arm。
        
        Returns:
            选择的arm索引
        """
        self.total_rounds += 1
        
        # Exploration阶段：随机选择未探索的arm
        if self.total_rounds <= self.K_exp:
            unexplored = np.where(self.arm_pulls == 0)[0]
            if len(unexplored) > 0:
                return int(np.random.choice(unexplored))
        
        # Thompson Sampling: 从Beta后验中采样
        samples = np.random.beta(self.alpha_params, self.beta_params)
        return int(np.argmax(samples))
    
    def update(self, arm: int, reward: float):
        """更新arm参数
        
        Args:
            arm: 被选择的arm索引
            reward: 观察到的奖励值 [0, 1]
        """
        # 将连续reward转换为伯努利反馈
        # 使用reward作为成功概率的观测
        success_prob = np.clip(reward, 0.0, 1.0)
        
        # 使用软更新（加权更新而非二值更新）
        self.alpha_params[arm] += success_prob
        self.beta_params[arm] += (1.0 - success_prob)
        
        self.arm_pulls[arm] += 1
        self.arm_total_rewards[arm] += reward
    
    def get_alpha(self, arm: int) -> float:
        """获取arm对应的alpha值"""
        return self.alpha_values[arm]
    
    def get_best_alpha(self) -> float:
        """获取当前最优alpha（基于后验均值）"""
        posterior_means = self.alpha_params / (self.alpha_params + self.beta_params)
        best_arm = int(np.argmax(posterior_means))
        return self.alpha_values[best_arm]
    
    def get_statistics(self) -> Dict:
        """获取bandit统计信息"""
        posterior_means = self.alpha_params / (self.alpha_params + self.beta_params)
        stats = {
            'total_rounds': int(self.total_rounds),
            'exploration_remaining': max(0, self.K_exp - int(self.total_rounds)),
            'n_arms': self.n_arms,
            'arm_pulls': dict(zip(
                [f"alpha={a:.1f}" for a in self.alpha_values],
                self.arm_pulls.astype(int).tolist()
            )),
            'posterior_means': dict(zip(
                [f"alpha={a:.1f}" for a in self.alpha_values],
                posterior_means.tolist()
            )),
            'best_alpha': self.get_best_alpha()
        }
        return stats


# ==================== Bandit-based Adaptive Fusion ====================

class BanditAdaptiveFusion:
    """基于Bandit的自适应融合
    
    将contextual bandit集成到自适应融合框架中。
    支持LinUCB和Thompson Sampling两种策略。
    """
    
    def __init__(self, config: Dict, strategy: str = 'linucb'):
        """
        Args:
            config: 配置字典
            strategy: bandit策略，'linucb' 或 'thompson'
        """
        self.config = config
        self.strategy = strategy
        
        # 初始化组件
        self.drift_detector = ConceptDriftDetector(config)
        self.context_extractor = ContextFeatureExtractor(self.drift_detector)
        self.reward_fn = RewardFunction(
            lambda_cov=config['model']['adaptive'].get('lambda_cov', 0.4),
            lambda_prec=config['model']['adaptive'].get('lambda_prec', 0.6),
            K=config['model']['adaptive'].get('reward_K', 10)
        )
        
        # Exploration budget
        self.K_exp = config['model']['adaptive'].get('exploration_budget', 100)
        
        # 初始化bandit
        if strategy == 'linucb':
            self.bandit = LinUCBBandit(
                context_dim=self.context_extractor.feature_dim,
                exploration_budget=self.K_exp
            )
        elif strategy == 'thompson':
            self.bandit = ThompsonSamplingBandit(
                exploration_budget=self.K_exp
            )
        else:
            raise ValueError(f"未知策略: {strategy}，支持 'linucb' 或 'thompson'")
        
        # 历史记录
        self.alpha_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        
        logger.info(f"Bandit自适应融合初始化完成，策略: {strategy}")
    
    def select_and_fuse(self,
                        kg_scores: np.ndarray,
                        llm_scores: np.ndarray,
                        relevant: Set[str] = None,
                        all_candidates: Set[str] = None) -> Tuple[np.ndarray, float, int]:
        """选择alpha并执行融合
        
        Args:
            kg_scores: KG分数
            llm_scores: LLM分数
            relevant: 相关实体集合（用于计算reward）
            all_candidates: 所有候选集合
            
        Returns:
            (fusion_scores, selected_alpha, arm_index)
        """
        # 获取漂移分数
        drift_score = self.drift_detector.detect_drift(kg_scores, llm_scores)
        
        # 提取context features
        context = self.context_extractor.extract_features(kg_scores, llm_scores, drift_score)
        
        # 选择arm
        if self.strategy == 'linucb':
            arm = self.bandit.select_arm(context)
        else:
            arm = self.bandit.select_arm()
        
        alpha = self.bandit.get_alpha(arm)
        
        # 执行融合
        fusion_scores = alpha * kg_scores + (1 - alpha) * llm_scores
        
        # 计算reward（如果有标签）
        reward = 0.0
        if relevant is not None:
            # 将融合结果排序作为推荐列表
            if len(fusion_scores) > 0:
                sorted_indices = np.argsort(-fusion_scores)
                # 使用索引映射回候选名称（简化处理）
                recommended = [f"item_{idx}" for idx in sorted_indices[:20]]
                reward = self.reward_fn.compute_reward(
                    recommended, relevant, all_candidates
                )
        
        # 更新bandit
        if self.strategy == 'linucb':
            self.bandit.update(arm, context, reward)
        else:
            self.bandit.update(arm, reward)
        
        # 记录历史
        self.alpha_history.append(alpha)
        self.reward_history.append(reward)
        
        return fusion_scores, alpha, arm
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'strategy': self.strategy,
            'current_alpha_history': list(self.alpha_history)[-10:],
            'avg_reward': float(np.mean(self.reward_history)) if self.reward_history else 0.0,
            'bandit_stats': self.bandit.get_statistics()
        }
        return stats