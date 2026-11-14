# -*- coding: utf-8 -*-
"""
MMR多样性优化模块
实现最大边际相关性算法，确保关键词集合的多样性和相关性平衡
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Set, Optional
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from loguru import logger
import heapq

class MMROptimizer:
    """MMR多样性优化器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.lambda_diversity = config['model']['mmr']['lambda_diversity']
        self.similarity_metric = config['model']['mmr']['similarity_metric']
        
        # 缓存相似度矩阵
        self.similarity_cache = {}
        
        logger.info(f"MMR优化器初始化完成，多样性权重: {self.lambda_diversity}")
    
    def mmr_diversify(self, candidates: List[Tuple[str, float]], 
                     embeddings: Dict[str, np.ndarray],
                     query_embedding: Optional[np.ndarray] = None,
                     top_k: int = 10,
                     lambda_diversity: Optional[float] = None) -> List[Tuple[str, float]]:
        """MMR多样性选择
        
        Args:
            candidates: 候选词及其基础分数列表 [(word, score), ...]
            embeddings: 候选词嵌入字典 {word: embedding}
            query_embedding: 查询/主题嵌入（可选）
            top_k: 返回的关键词数量
            lambda_diversity: 多样性权重（可选，覆盖默认值）
            
        Returns:
            选择的关键词及其MMR分数列表
        """
        if not candidates:
            return []
        
        if lambda_diversity is None:
            lambda_diversity = self.lambda_diversity
        
        # 初始化
        selected = []  # 已选择的关键词
        remaining = candidates.copy()  # 剩余候选词
        
        # 如果没有查询嵌入，使用候选词的平均嵌入作为查询
        if query_embedding is None:
            valid_embeddings = [embeddings[word] for word, _ in candidates if word in embeddings]
            if valid_embeddings:
                query_embedding = np.mean(valid_embeddings, axis=0)
            else:
                logger.warning("无法获取查询嵌入，使用零向量")
                query_embedding = np.zeros(768)  # 默认BERT维度
        
        # 预计算相似度矩阵
        similarity_matrix = self._compute_similarity_matrix(candidates, embeddings)
        
        # MMR选择循环
        for _ in range(min(top_k, len(candidates))):
            if not remaining:
                break
            
            best_candidate = None
            best_mmr_score = float('-inf')
            best_idx = -1
            
            # 计算每个剩余候选词的MMR分数
            for idx, (candidate, base_score) in enumerate(remaining):
                if candidate not in embeddings:
                    continue
                
                # 计算相关性分数
                relevance_score = self._calculate_relevance(
                    candidate, embeddings[candidate], query_embedding, base_score
                )
                
                # 计算多样性惩罚
                diversity_penalty = self._calculate_diversity_penalty(
                    candidate, selected, embeddings, similarity_matrix
                )
                
                # MMR分数
                mmr_score = (1 - lambda_diversity) * relevance_score - lambda_diversity * diversity_penalty
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = (candidate, mmr_score)
                    best_idx = idx
            
            # 添加最佳候选词
            if best_candidate is not None:
                selected.append(best_candidate)
                remaining.pop(best_idx)
        
        logger.debug(f"MMR选择完成，从 {len(candidates)} 个候选词中选择了 {len(selected)} 个")
        
        return selected
    
    def _compute_similarity_matrix(self, candidates: List[Tuple[str, float]], 
                                  embeddings: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
        """预计算相似度矩阵"""
        similarity_matrix = {}
        words = [word for word, _ in candidates if word in embeddings]
        
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if i != j:
                    pair = (word1, word2)
                    if pair not in similarity_matrix:
                        sim = self._calculate_similarity(
                            embeddings[word1], embeddings[word2]
                        )
                        similarity_matrix[pair] = sim
                        # 对称性
                        similarity_matrix[(word2, word1)] = sim
        
        return similarity_matrix
    
    def _calculate_relevance(self, candidate: str, candidate_embedding: np.ndarray,
                           query_embedding: np.ndarray, base_score: float) -> float:
        """计算相关性分数"""
        # 语义相关性
        semantic_relevance = self._calculate_similarity(candidate_embedding, query_embedding)
        
        # 结合基础分数
        relevance_score = 0.7 * semantic_relevance + 0.3 * base_score
        
        return relevance_score
    
    def _calculate_diversity_penalty(self, candidate: str, selected: List[Tuple[str, float]],
                                   embeddings: Dict[str, np.ndarray],
                                   similarity_matrix: Dict[Tuple[str, str], float]) -> float:
        """计算多样性惩罚"""
        if not selected or candidate not in embeddings:
            return 0.0
        
        max_similarity = 0.0
        
        for selected_word, _ in selected:
            if selected_word in embeddings:
                pair = (candidate, selected_word)
                if pair in similarity_matrix:
                    similarity = similarity_matrix[pair]
                else:
                    # 实时计算（备用）
                    similarity = self._calculate_similarity(
                        embeddings[candidate], embeddings[selected_word]
                    )
                
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个嵌入之间的相似度"""
        if self.similarity_metric == 'cosine':
            # 余弦相似度
            similarity = cosine_similarity(
                embedding1.reshape(1, -1), 
                embedding2.reshape(1, -1)
            )[0][0]
        elif self.similarity_metric == 'euclidean':
            # 欧几里得距离（转换为相似度）
            distance = np.linalg.norm(embedding1 - embedding2)
            similarity = 1.0 / (1.0 + distance)
        elif self.similarity_metric == 'dot_product':
            # 点积
            similarity = np.dot(embedding1, embedding2)
        else:
            # 默认使用余弦相似度
            similarity = cosine_similarity(
                embedding1.reshape(1, -1), 
                embedding2.reshape(1, -1)
            )[0][0]
        
        return float(similarity)
    
    def batch_mmr_diversify(self, batch_candidates: List[List[Tuple[str, float]]],
                           batch_embeddings: List[Dict[str, np.ndarray]],
                           batch_query_embeddings: Optional[List[np.ndarray]] = None,
                           top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """批量MMR多样性选择"""
        results = []
        
        for i, candidates in enumerate(batch_candidates):
            embeddings = batch_embeddings[i]
            query_embedding = batch_query_embeddings[i] if batch_query_embeddings else None
            
            selected = self.mmr_diversify(
                candidates, embeddings, query_embedding, top_k
            )
            results.append(selected)
        
        logger.info(f"批量MMR选择完成，处理了 {len(batch_candidates)} 个候选集")
        
        return results
    
    def adaptive_mmr(self, candidates: List[Tuple[str, float]],
                    embeddings: Dict[str, np.ndarray],
                    query_embedding: Optional[np.ndarray] = None,
                    top_k: int = 10,
                    diversity_strategy: str = 'balanced') -> List[Tuple[str, float]]:
        """自适应MMR选择
        
        Args:
            diversity_strategy: 多样性策略 ('high', 'balanced', 'low')
        """
        # 根据策略调整lambda值
        if diversity_strategy == 'high':
            lambda_diversity = 0.8
        elif diversity_strategy == 'balanced':
            lambda_diversity = 0.5
        elif diversity_strategy == 'low':
            lambda_diversity = 0.2
        else:
            lambda_diversity = self.lambda_diversity
        
        return self.mmr_diversify(
            candidates, embeddings, query_embedding, top_k, lambda_diversity
        )
    
    def incremental_mmr(self, candidates: List[Tuple[str, float]],
                       embeddings: Dict[str, np.ndarray],
                       existing_keywords: List[str],
                       query_embedding: Optional[np.ndarray] = None,
                       top_k: int = 10) -> List[Tuple[str, float]]:
        """增量MMR选择（考虑已有关键词）"""
        if not candidates:
            return []
        
        # 将已有关键词作为初始选择
        selected = [(word, 1.0) for word in existing_keywords if word in embeddings]
        remaining = candidates.copy()
        
        # 移除已存在的候选词
        remaining = [(word, score) for word, score in remaining if word not in existing_keywords]
        
        if query_embedding is None:
            valid_embeddings = [embeddings[word] for word, _ in remaining if word in embeddings]
            if valid_embeddings:
                query_embedding = np.mean(valid_embeddings, axis=0)
            else:
                return []
        
        # 预计算相似度矩阵
        all_words = existing_keywords + [word for word, _ in remaining]
        all_candidates = [(word, 1.0) for word in all_words]
        similarity_matrix = self._compute_similarity_matrix(all_candidates, embeddings)
        
        # 继续MMR选择
        for _ in range(min(top_k, len(remaining))):
            if not remaining:
                break
            
            best_candidate = None
            best_mmr_score = float('-inf')
            best_idx = -1
            
            for idx, (candidate, base_score) in enumerate(remaining):
                if candidate not in embeddings:
                    continue
                
                # 计算相关性分数
                relevance_score = self._calculate_relevance(
                    candidate, embeddings[candidate], query_embedding, base_score
                )
                
                # 计算多样性惩罚（包括已有关键词）
                diversity_penalty = self._calculate_diversity_penalty(
                    candidate, selected, embeddings, similarity_matrix
                )
                
                # MMR分数
                mmr_score = (1 - self.lambda_diversity) * relevance_score - self.lambda_diversity * diversity_penalty
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = (candidate, mmr_score)
                    best_idx = idx
            
            if best_candidate is not None:
                selected.append(best_candidate)
                remaining.pop(best_idx)
        
        # 只返回新选择的关键词
        new_selected = selected[len(existing_keywords):]
        
        logger.debug(f"增量MMR选择完成，在 {len(existing_keywords)} 个已有关键词基础上新增了 {len(new_selected)} 个")
        
        return new_selected
    
    def evaluate_diversity(self, keywords: List[str], 
                          embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
        """评估关键词集合的多样性"""
        if len(keywords) < 2:
            return {'avg_similarity': 0.0, 'min_similarity': 0.0, 'max_similarity': 0.0, 'diversity_score': 1.0}
        
        similarities = []
        
        for i, word1 in enumerate(keywords):
            for j, word2 in enumerate(keywords):
                if i < j and word1 in embeddings and word2 in embeddings:
                    sim = self._calculate_similarity(embeddings[word1], embeddings[word2])
                    similarities.append(sim)
        
        if not similarities:
            return {'avg_similarity': 0.0, 'min_similarity': 0.0, 'max_similarity': 0.0, 'diversity_score': 1.0}
        
        avg_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        
        # 多样性分数（1 - 平均相似度）
        diversity_score = 1.0 - avg_sim
        
        return {
            'avg_similarity': avg_sim,
            'min_similarity': min_sim,
            'max_similarity': max_sim,
            'diversity_score': diversity_score
        }
    
    def optimize_lambda(self, candidates: List[Tuple[str, float]],
                       embeddings: Dict[str, np.ndarray],
                       ground_truth: List[str],
                       query_embedding: Optional[np.ndarray] = None,
                       top_k: int = 10,
                       lambda_range: Tuple[float, float] = (0.1, 0.9),
                       num_steps: int = 9) -> Tuple[float, Dict[str, float]]:
        """优化lambda参数
        
        Returns:
            最优lambda值和对应的评估指标
        """
        best_lambda = self.lambda_diversity
        best_score = 0.0
        best_metrics = {}
        
        lambda_values = np.linspace(lambda_range[0], lambda_range[1], num_steps)
        
        for lambda_val in lambda_values:
            # 使用当前lambda值进行MMR选择
            selected = self.mmr_diversify(
                candidates, embeddings, query_embedding, top_k, lambda_val
            )
            
            selected_words = [word for word, _ in selected]
            
            # 计算精度
            precision = len(set(selected_words) & set(ground_truth)) / len(selected_words) if selected_words else 0
            
            # 计算多样性
            diversity_metrics = self.evaluate_diversity(selected_words, embeddings)
            diversity_score = diversity_metrics['diversity_score']
            
            # 综合分数（可调整权重）
            combined_score = 0.7 * precision + 0.3 * diversity_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_lambda = lambda_val
                best_metrics = {
                    'precision': precision,
                    'diversity_score': diversity_score,
                    'combined_score': combined_score
                }
        
        logger.info(f"Lambda优化完成，最优值: {best_lambda:.3f}，综合分数: {best_score:.3f}")
        
        return best_lambda, best_metrics
    
    def get_mmr_statistics(self, selected_keywords: List[Tuple[str, float]],
                          original_candidates: List[Tuple[str, float]],
                          embeddings: Dict[str, np.ndarray]) -> Dict:
        """获取MMR选择统计信息"""
        selected_words = [word for word, _ in selected_keywords]
        selected_scores = [score for _, score in selected_keywords]
        
        original_words = [word for word, _ in original_candidates]
        original_scores = [score for _, score in original_candidates]
        
        # 多样性评估
        diversity_metrics = self.evaluate_diversity(selected_words, embeddings)
        
        stats = {
            'num_selected': len(selected_keywords),
            'num_original': len(original_candidates),
            'selection_ratio': len(selected_keywords) / len(original_candidates) if original_candidates else 0,
            'avg_mmr_score': np.mean(selected_scores) if selected_scores else 0,
            'avg_original_score': np.mean(original_scores) if original_scores else 0,
            'score_improvement': (np.mean(selected_scores) - np.mean(original_scores)) if selected_scores and original_scores else 0,
            'diversity_metrics': diversity_metrics,
            'lambda_diversity': self.lambda_diversity
        }
        
        return stats


class DiversityMetrics:
    """多样性指标计算器"""
    
    @staticmethod
    def calculate_intra_list_diversity(keywords: List[str], 
                                     embeddings: Dict[str, np.ndarray],
                                     metric: str = 'cosine') -> float:
        """计算列表内多样性"""
        if len(keywords) < 2:
            return 1.0
        
        similarities = []
        
        for i, word1 in enumerate(keywords):
            for j, word2 in enumerate(keywords):
                if i < j and word1 in embeddings and word2 in embeddings:
                    if metric == 'cosine':
                        sim = cosine_similarity(
                            embeddings[word1].reshape(1, -1),
                            embeddings[word2].reshape(1, -1)
                        )[0][0]
                    else:
                        # 其他相似度度量可以在这里添加
                        sim = cosine_similarity(
                            embeddings[word1].reshape(1, -1),
                            embeddings[word2].reshape(1, -1)
                        )[0][0]
                    
                    similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        # 多样性 = 1 - 平均相似度
        diversity = 1.0 - np.mean(similarities)
        return max(0.0, diversity)  # 确保非负
    
    @staticmethod
    def calculate_coverage_diversity(keywords: List[str],
                                   all_candidates: List[str],
                                   embeddings: Dict[str, np.ndarray],
                                   threshold: float = 0.7) -> float:
        """计算覆盖多样性"""
        if not keywords or not all_candidates:
            return 0.0
        
        covered_candidates = set()
        
        for keyword in keywords:
            if keyword not in embeddings:
                continue
                
            for candidate in all_candidates:
                if candidate not in embeddings:
                    continue
                    
                similarity = cosine_similarity(
                    embeddings[keyword].reshape(1, -1),
                    embeddings[candidate].reshape(1, -1)
                )[0][0]
                
                if similarity >= threshold:
                    covered_candidates.add(candidate)
        
        coverage = len(covered_candidates) / len(all_candidates)
        return coverage
    
    @staticmethod
    def calculate_shannon_diversity(keywords: List[str],
                                  topic_assignments: Dict[str, int]) -> float:
        """计算Shannon多样性指数"""
        if not keywords:
            return 0.0
        
        # 统计每个主题的关键词数量
        topic_counts = defaultdict(int)
        total_keywords = 0
        
        for keyword in keywords:
            if keyword in topic_assignments:
                topic_counts[topic_assignments[keyword]] += 1
                total_keywords += 1
        
        if total_keywords == 0:
            return 0.0
        
        # 计算Shannon熵
        shannon_entropy = 0.0
        for count in topic_counts.values():
            if count > 0:
                proportion = count / total_keywords
                shannon_entropy -= proportion * np.log2(proportion)
        
        return shannon_entropy