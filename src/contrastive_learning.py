# -*- coding: utf-8 -*-
"""
对比学习模块
实现基于知识图谱的对比学习，用于学习高质量的节点嵌入。

核心功能：
- InfoNCE Loss 函数
- 正样本构造（时间共现、同事件对）
- 硬负样本采样（高语义相似但无边）
- 增量训练循环（每微批次几步SGD）
- 节点嵌入缓存和刷新
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict, deque
from datetime import datetime
from loguru import logger
import random


class InfoNCELoss(nn.Module):
    """InfoNCE (Information Noise Contrastive Estimation) Loss
    
    对比学习的标准损失函数，也称为NT-Xent (Normalized Temperature-scaled
    Cross-Entropy) Loss。
    
    公式:
        L = -1/(2N) * sum_{i=1}^{2N} log[ exp(sim(z_i, z_j) / tau) /
                                          sum_{k=1}^{2N} 1_{k!=i} * exp(sim(z_i, z_k) / tau) ]
    
    其中 sim(u, v) = u^T * v / (||u|| * ||v||)，tau为温度参数。
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: 温度参数tau，控制分布的锐度。
                        较小的tau使模型更关注难样本。
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self,
                anchor_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: torch.Tensor = None) -> torch.Tensor:
        """计算InfoNCE损失
        
        Args:
            anchor_embeddings: 锚点嵌入 (N, d)
            positive_embeddings: 正样本嵌入 (N, d)
            negative_embeddings: 负样本嵌入 (N, K, d) 或 None
                                若为None，使用batch内其他样本作为负样本
        
        Returns:
            标量损失值
        """
        N = anchor_embeddings.size(0)
        device = anchor_embeddings.device
        
        # L2归一化
        anchor = F.normalize(anchor_embeddings, dim=1)
        positive = F.normalize(positive_embeddings, dim=1)
        
        if negative_embeddings is not None:
            # 显式负样本模式
            negative_embeddings = negative_embeddings.view(N, -1, negative_embeddings.size(-1))
            negative = F.normalize(negative_embeddings, dim=2)
            
            # 计算正样本相似度: (N,)
            pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # (N,)
            
            # 计算负样本相似度: (N, K)
            neg_sim = torch.bmm(negative, anchor.unsqueeze(2)).squeeze(2) / self.temperature  # (N, K)
            
            # 拼接: (N, 1+K)
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            
            # 标签: 正样本在索引0
            labels = torch.zeros(N, dtype=torch.long, device=device)
            
            loss = F.cross_entropy(logits, labels)
            
        else:
            # Batch内负样本模式
            # 拼接anchor和positive: (2N, d)
            all_embeddings = torch.cat([anchor, positive], dim=0)  # (2N, d)
            
            # 计算相似度矩阵: (2N, 2N)
            sim_matrix = torch.mm(all_embeddings, all_embeddings.t()) / self.temperature
            
            # 排除自身
            mask = torch.eye(2 * N, dtype=torch.bool, device=device)
            sim_matrix = sim_matrix.masked_fill(mask, -1e9)
            
            # 标签: anchor i 的正样本是 i+N，positive i+N 的正样本是 i
            labels = torch.cat([
                torch.arange(N, 2 * N, device=device),  # anchor的正样本索引
                torch.arange(0, N, device=device)        # positive的正样本索引
            ])
            
            loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """计算两个嵌入的余弦相似度
        
        Args:
            emb1: (N, d) 或 (d,)
            emb2: (N, d) 或 (d,)
            
        Returns:
            相似度值
        """
        e1 = F.normalize(emb1, dim=-1)
        e2 = F.normalize(emb2, dim=-1)
        return torch.sum(e1 * e2, dim=-1)


class PositiveSampleConstructor:
    """正样本构造器
    
    基于知识图谱结构构造正样本对：
    1. 时间共现正样本：在时间窗口内共现的实体
    2. 同事件正样本：属于同一事件/文档的实体
    3. 邻居正样本：在KG中直接相连的实体
    """
    
    def __init__(self,
                 time_window_hours: float = 24.0,
                 co_occurrence_threshold: int = 2):
        """
        Args:
            time_window_hours: 时间窗口大小（小时）
            co_occurrence_threshold: 共现次数阈值
        """
        self.time_window_hours = time_window_hours
        self.co_occurrence_threshold = co_occurrence_threshold
        
        # 共现统计
        self.cooccurrence_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self.entity_timestamps: Dict[str, List[datetime]] = defaultdict(list)
        
        # 事件-实体映射
        self.event_entities: Dict[str, Set[str]] = defaultdict(set)
        
        logger.info(f"正样本构造器初始化完成，时间窗口: {time_window_hours}h")
    
    def add_cooccurrence(self, entity1: str, entity2: str):
        """记录实体共现"""
        if entity1 == entity2:
            return
        pair = tuple(sorted([entity1, entity2]))
        self.cooccurrence_count[pair] += 1
    
    def add_entity_timestamp(self, entity: str, timestamp: datetime):
        """记录实体出现时间"""
        self.entity_timestamps[entity].append(timestamp)
    
    def add_event_entities(self, event_id: str, entities: List[str]):
        """记录事件中的实体"""
        self.event_entities[event_id].update(entities)
    
    def get_temporal_positive_pairs(self) -> List[Tuple[str, str, float]]:
        """获取时间共现正样本对
        
        Returns:
            [(entity1, entity2, weight), ...] 权重基于共现频率
        """
        positive_pairs = []
        
        for (e1, e2), count in self.cooccurrence_count.items():
            if count >= self.co_occurrence_threshold:
                # 检查时间窗口
                times1 = self.entity_timestamps.get(e1, [])
                times2 = self.entity_timestamps.get(e2, [])
                
                if times1 and times2:
                    # 计算最近一次共现的时间差
                    latest1 = max(times1)
                    latest2 = max(times2)
                    delta_hours = abs((latest1 - latest2).total_seconds() / 3600)
                    
                    if delta_hours <= self.time_window_hours:
                        # 权重 = 共现次数 * 时间衰减
                        time_decay = np.exp(-delta_hours / self.time_window_hours)
                        weight = count * time_decay
                        positive_pairs.append((e1, e2, float(weight)))
        
        return positive_pairs
    
    def get_event_positive_pairs(self) -> List[Tuple[str, str, float]]:
        """获取同事件正样本对
        
        Returns:
            [(entity1, entity2, weight), ...]
        """
        positive_pairs = []
        
        for event_id, entities in self.event_entities.items():
            entity_list = list(entities)
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    positive_pairs.append((entity_list[i], entity_list[j], 1.0))
        
        return positive_pairs
    
    def get_all_positive_pairs(self) -> List[Tuple[str, str, float]]:
        """获取所有正样本对（合并时间共现和同事件）"""
        temporal_pairs = self.get_temporal_positive_pairs()
        event_pairs = self.get_event_positive_pairs()
        
        # 合并并去重
        pair_weights: Dict[Tuple[str, str], float] = defaultdict(float)
        for e1, e2, w in temporal_pairs:
            pair_weights[(e1, e2)] += w
        for e1, e2, w in event_pairs:
            pair_weights[(e1, e2)] += w
        
        return [(e1, e2, w) for (e1, e2), w in pair_weights.items()]


class HardNegativeSampler:
    """硬负样本采样器
    
    采样策略：选择语义相似度高但在知识图谱中没有边连接的实体对作为硬负样本。
    这迫使模型学习更细粒度的区分能力。
    """
    
    def __init__(self,
                 similarity_threshold: float = 0.7,
                 num_negatives_per_anchor: int = 5,
                 kg_edges: Set[Tuple[str, str]] = None):
        """
        Args:
            similarity_threshold: 语义相似度阈值，高于此值且无边的作为硬负样本
            num_negatives_per_anchor: 每个锚点的负样本数量
            kg_edges: 知识图谱中的边集合（无向）
        """
        self.similarity_threshold = similarity_threshold
        self.num_negatives = num_negatives_per_anchor
        self.kg_edges: Set[Tuple[str, str]] = kg_edges or set()
        
        # 语义相似度缓存
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
        logger.info(f"硬负样本采样器初始化完成，阈值: {similarity_threshold}")
    
    def update_kg_edges(self, edges: List[Tuple[str, str]]):
        """更新KG边集合"""
        for u, v in edges:
            self.kg_edges.add((u, v))
            self.kg_edges.add((v, u))  # 无向
    
    def has_edge(self, entity1: str, entity2: str) -> bool:
        """检查两个实体之间是否有边"""
        return (entity1, entity2) in self.kg_edges
    
    def sample_hard_negatives(self,
                               anchor: str,
                               candidate_entities: List[str],
                               embeddings: Dict[str, np.ndarray]) -> List[str]:
        """为锚点采样硬负样本
        
        策略：
        1. 计算锚点与所有候选实体的语义相似度
        2. 过滤掉有边连接的实体（这些是正样本或不应作为负样本）
        3. 在高相似度（>threshold）的无边实体中采样
        
        Args:
            anchor: 锚点实体
            candidate_entities: 候选实体列表
            embeddings: 实体嵌入字典
            
        Returns:
            硬负样本列表
        """
        if anchor not in embeddings:
            return []
        
        anchor_emb = embeddings[anchor]
        hard_negatives = []
        
        # 计算所有候选的相似度
        similarities = []
        for candidate in candidate_entities:
            if candidate == anchor:
                continue
            if self.has_edge(anchor, candidate):
                continue  # 有边的不能作为负样本
            
            if candidate not in embeddings:
                continue
            
            # 检查缓存
            cache_key = tuple(sorted([anchor, candidate]))
            if cache_key in self.similarity_cache:
                sim = self.similarity_cache[cache_key]
            else:
                cand_emb = embeddings[candidate]
                sim = float(np.dot(anchor_emb, cand_emb) /
                            (np.linalg.norm(anchor_emb) * np.linalg.norm(cand_emb) + 1e-8))
                self.similarity_cache[cache_key] = sim
            
            if sim >= self.similarity_threshold:
                similarities.append((candidate, sim))
        
        # 按相似度降序排序，选择top-K
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 采样：优先选择最难的（相似度最高的），但加入少量随机性
        if len(similarities) <= self.num_negatives:
            hard_negatives = [s[0] for s in similarities]
        else:
            # 80%选择最难的，20%随机选择
            n_top = max(1, int(self.num_negatives * 0.8))
            n_random = self.num_negatives - n_top
            
            top_negatives = [s[0] for s in similarities[:n_top]]
            
            remaining = [s[0] for s in similarities[n_top:]]
            random_negatives = random.sample(remaining, min(n_random, len(remaining)))
            
            hard_negatives = top_negatives + random_negatives
        
        return hard_negatives[:self.num_negatives]
    
    def batch_sample_negatives(self,
                               anchors: List[str],
                               all_entities: List[str],
                               embeddings: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """批量采样硬负样本
        
        Args:
            anchors: 锚点列表
            all_entities: 所有候选实体
            embeddings: 实体嵌入字典
            
        Returns:
            {anchor: [negative_entities]} 字典
        """
        result = {}
        for anchor in anchors:
            negatives = self.sample_hard_negatives(anchor, all_entities, embeddings)
            result[anchor] = negatives
        return result
    
    def clear_cache(self):
        """清除相似度缓存"""
        self.similarity_cache.clear()


class NodeEmbeddingCache:
    """节点嵌入缓存和刷新机制
    
    维护一个LRU缓存，存储节点的嵌入向量。
    支持增量更新和定期刷新。
    """
    
    def __init__(self,
                 embedding_dim: int = 768,
                 max_cache_size: int = 10000,
                 refresh_interval: int = 1000):
        """
        Args:
            embedding_dim: 嵌入维度
            max_cache_size: 最大缓存条目数
            refresh_interval: 刷新间隔（更新次数）
        """
        self.embedding_dim = embedding_dim
        self.max_cache_size = max_cache_size
        self.refresh_interval = refresh_interval
        
        # 嵌入存储
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # LRU访问记录
        self.access_order: deque = deque(maxlen=max_cache_size)
        
        # 更新计数
        self.update_count = 0
        
        # 待刷新的节点
        self.dirty_nodes: Set[str] = set()
        
        # 统计信息
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"节点嵌入缓存初始化完成，维度: {embedding_dim}，容量: {max_cache_size}")
    
    def get(self, node_id: str) -> Optional[np.ndarray]:
        """获取节点嵌入
        
        Args:
            node_id: 节点ID
            
        Returns:
            嵌入向量或None
        """
        if node_id in self.embeddings:
            self.cache_hits += 1
            # 更新LRU顺序
            if node_id in self.access_order:
                self.access_order.remove(node_id)
            self.access_order.append(node_id)
            return self.embeddings[node_id]
        
        self.cache_misses += 1
        return None
    
    def put(self, node_id: str, embedding: np.ndarray):
        """存入节点嵌入
        
        Args:
            node_id: 节点ID
            embedding: 嵌入向量
        """
        # 如果缓存已满，淘汰最久未使用的
        if len(self.embeddings) >= self.max_cache_size and node_id not in self.embeddings:
            self._evict_lru()
        
        self.embeddings[node_id] = embedding.copy()
        
        # 更新LRU顺序
        if node_id in self.access_order:
            self.access_order.remove(node_id)
        self.access_order.append(node_id)
    
    def batch_get(self, node_ids: List[str]) -> Dict[str, np.ndarray]:
        """批量获取节点嵌入
        
        Args:
            node_ids: 节点ID列表
            
        Returns:
            {node_id: embedding} 字典（仅包含缓存中存在的）
        """
        result = {}
        for nid in node_ids:
            emb = self.get(nid)
            if emb is not None:
                result[nid] = emb
        return result
    
    def mark_dirty(self, node_id: str):
        """标记节点为脏（需要刷新）"""
        self.dirty_nodes.add(node_id)
    
    def mark_dirty_batch(self, node_ids: List[str]):
        """批量标记脏节点"""
        self.dirty_nodes.update(node_ids)
    
    def refresh(self, embedding_fn=None):
        """刷新脏节点
        
        Args:
            embedding_fn: 用于重新计算嵌入的函数 (node_id) -> np.ndarray
                         若为None，仅清除脏节点的嵌入
        """
        if not self.dirty_nodes:
            return
        
        refreshed_count = 0
        for node_id in list(self.dirty_nodes):
            if embedding_fn is not None:
                try:
                    new_emb = embedding_fn(node_id)
                    if new_emb is not None:
                        self.put(node_id, new_emb)
                        refreshed_count += 1
                except Exception as e:
                    logger.warning(f"刷新节点 {node_id} 嵌入失败: {e}")
            else:
                # 无embedding_fn，直接清除
                if node_id in self.embeddings:
                    del self.embeddings[node_id]
                refreshed_count += 1
        
        self.dirty_nodes.clear()
        self.update_count += 1
        
        logger.debug(f"缓存刷新完成: {refreshed_count} 个节点，"
                      f"总更新次数: {self.update_count}")
    
    def should_refresh(self) -> bool:
        """检查是否需要定期刷新"""
        return self.update_count > 0 and self.update_count % self.refresh_interval == 0
    
    def _evict_lru(self):
        """淘汰最久未使用的节点"""
        if self.access_order:
            oldest = self.access_order.popleft()
            if oldest in self.embeddings:
                del self.embeddings[oldest]
    
    def get_statistics(self) -> Dict:
        """获取缓存统计信息"""
        total_access = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_access if total_access > 0 else 0.0
        
        return {
            'cache_size': len(self.embeddings),
            'max_cache_size': self.max_cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': float(hit_rate),
            'dirty_nodes': len(self.dirty_nodes),
            'update_count': self.update_count,
            'embedding_dim': self.embedding_dim
        }
    
    def clear(self):
        """清空缓存"""
        self.embeddings.clear()
        self.access_order.clear()
        self.dirty_nodes.clear()
        self.cache_hits = 0
        self.cache_misses = 0


class ContrastiveLearningTrainer:
    """对比学习增量训练器
    
    整合InfoNCE Loss、正样本构造、硬负样本采样和嵌入缓存，
    实现端到端的增量对比学习训练循环。
    """
    
    def __init__(self,
                 config: Dict,
                 embedding_dim: int = 768,
                 learning_rate: float = 1e-4,
                 temperature: float = 0.07):
        """
        Args:
            config: 配置字典
            embedding_dim: 嵌入维度
            learning_rate: 学习率
            temperature: InfoNCE温度参数
        """
        self.config = config
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        # 初始化组件
        self.loss_fn = InfoNCELoss(temperature=temperature)
        self.positive_constructor = PositiveSampleConstructor(
            time_window_hours=config.get('time_window_hours', 24.0),
            co_occurrence_threshold=config.get('co_occurrence_threshold', 2)
        )
        self.negative_sampler = HardNegativeSampler(
            similarity_threshold=config.get('neg_similarity_threshold', 0.7),
            num_negatives_per_anchor=config.get('num_negatives', 5)
        )
        self.embedding_cache = NodeEmbeddingCache(
            embedding_dim=embedding_dim,
            max_cache_size=config.get('cache_size', 10000),
            refresh_interval=config.get('refresh_interval', 1000)
        )
        
        # 增量训练参数
        self.sgd_steps_per_batch = config.get('sgd_steps_per_batch', 3)
        self.global_step = 0
        self.total_loss = 0.0
        self.loss_history: List[float] = []
        
        # 可选的编码器（如果使用可训练的encoder）
        self.encoder = None
        self.optimizer = None
        
        logger.info(f"对比学习训练器初始化完成，温度: {temperature}，"
                     f"学习率: {learning_rate}")
    
    def set_encoder(self, encoder: nn.Module):
        """设置可训练的编码器
        
        Args:
            encoder: PyTorch模块，输入节点ID（或特征），输出嵌入向量
        """
        self.encoder = encoder
        self.optimizer = torch.optim.Adam(
            encoder.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        logger.info("编码器和优化器已设置")
    
    def construct_training_batch(self,
                                  entities: List[str],
                                  embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """构造一个训练批次
        
        Args:
            entities: 参与训练的实体列表
            embeddings: 当前实体嵌入字典
            
        Returns:
            训练批次字典，包含anchors, positives, negatives
        """
        # 获取正样本对
        positive_pairs = self.positive_constructor.get_all_positive_pairs()
        
        if not positive_pairs:
            logger.warning("无正样本对，跳过批次构造")
            return {}
        
        # 过滤出在当前实体列表中的正样本对
        entity_set = set(entities)
        valid_pairs = [(e1, e2, w) for e1, e2, w in positive_pairs
                       if e1 in entity_set and e2 in entity_set]
        
        if not valid_pairs:
            return {}
        
        # 构造anchors和positives
        anchors = []
        positives = []
        weights = []
        
        for e1, e2, w in valid_pairs:
            anchors.append(e1)
            positives.append(e2)
            weights.append(w)
            
            # 双向添加
            anchors.append(e2)
            positives.append(e1)
            weights.append(w)
        
        # 采样硬负样本
        negatives_map = self.negative_sampler.batch_sample_negatives(
            list(set(anchors)), entities, embeddings
        )
        
        # 构造negative tensors
        neg_list = []
        for anchor in anchors:
            negs = negatives_map.get(anchor, [])
            if negs:
                neg_embeddings = np.array([embeddings[n] for n in negs if n in embeddings])
                if len(neg_embeddings) > 0:
                    neg_list.append(neg_embeddings)
                else:
                    neg_list.append(np.zeros((1, self.embedding_dim)))
            else:
                neg_list.append(np.zeros((1, self.embedding_dim)))
        
        # 转换为numpy数组
        anchor_embs = np.array([embeddings[a] for a in anchors if a in embeddings])
        positive_embs = np.array([embeddings[p] for p in positives if p in embeddings])
        
        if len(anchor_embs) == 0 or len(positive_embs) == 0:
            return {}
        
        # 对齐长度
        min_len = min(len(anchor_embs), len(positive_embs))
        anchor_embs = anchor_embs[:min_len]
        positive_embs = positive_embs[:min_len]
        neg_list = neg_list[:min_len]
        
        # Padding negatives到相同长度
        max_neg = max(n.shape[0] for n in neg_list)
        neg_array = np.zeros((len(neg_list), max_neg, self.embedding_dim))
        for i, n in enumerate(neg_list):
            neg_array[i, :n.shape[0], :] = n
        
        batch = {
            'anchors': torch.tensor(anchor_embs, dtype=torch.float32),
            'positives': torch.tensor(positive_embs, dtype=torch.float32),
            'negatives': torch.tensor(neg_array, dtype=torch.float32),
            'anchor_ids': anchors[:min_len],
            'positive_ids': positives[:min_len],
            'weights': np.array(weights[:min_len])
        }
        
        return batch
    
    def train_step(self, batch: Dict[str, Any]) -> float:
        """执行一步训练
        
        Args:
            batch: 训练批次（来自construct_training_batch）
            
        Returns:
            损失值
        """
        if not batch or 'anchors' not in batch:
            return 0.0
        
        anchors = batch['anchors']
        positives = batch['positives']
        negatives = batch['negatives']
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        anchors = anchors.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)
        
        # 如果有可训练编码器，通过编码器计算嵌入
        if self.encoder is not None:
            self.encoder.train()
            anchor_embs = self.encoder(anchors)
            positive_embs = self.encoder(positives)
            
            # 展平negatives并编码
            N, K, D = negatives.shape
            flat_negs = negatives.view(N * K, D)
            neg_embs = self.encoder(flat_negs).view(N, K, -1)
        else:
            anchor_embs = anchors
            positive_embs = positives
            neg_embs = negatives
        
        # 计算InfoNCE损失
        loss = self.loss_fn(anchor_embs, positive_embs, neg_embs)
        
        # 反向传播
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) if self.encoder else [],
                max_norm=1.0
            )
            
            self.optimizer.step()
        
        self.global_step += 1
        loss_value = loss.item()
        self.total_loss += loss_value
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def incremental_train(self,
                          entities: List[str],
                          embeddings: Dict[str, np.ndarray],
                          num_sgd_steps: int = None) -> Dict:
        """增量训练循环
        
        对每个微批次执行几步SGD更新。
        
        Args:
            entities: 当前批次的实体列表
            embeddings: 当前实体嵌入
            num_sgd_steps: SGD步数，None则使用self.sgd_steps_per_batch
            
        Returns:
            训练统计信息
        """
        if num_sgd_steps is None:
            num_sgd_steps = self.sgd_steps_per_batch
        
        # 构造训练批次
        batch = self.construct_training_batch(entities, embeddings)
        
        if not batch:
            return {'loss': 0.0, 'steps': 0}
        
        # 执行多步SGD
        step_losses = []
        for step in range(num_sgd_steps):
            loss = self.train_step(batch)
            step_losses.append(loss)
        
        # 更新嵌入缓存
        if self.encoder is not None:
            self.encoder.eval()
            with torch.no_grad():
                device = next(self.encoder.parameters()).device
                for nid in entities:
                    if nid in embeddings:
                        emb_tensor = torch.tensor(
                            embeddings[nid], dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        new_emb = self.encoder(emb_tensor).cpu().numpy().flatten()
                        self.embedding_cache.put(nid, new_emb)
        else:
            for nid in entities:
                if nid in embeddings:
                    self.embedding_cache.put(nid, embeddings[nid])
        
        # 检查是否需要刷新缓存
        if self.embedding_cache.should_refresh():
            self.embedding_cache.refresh()
        
        avg_loss = np.mean(step_losses) if step_losses else 0.0
        
        return {
            'loss': float(avg_loss),
            'steps': num_sgd_steps,
            'num_anchors': len(batch.get('anchor_ids', [])),
            'global_step': self.global_step
        }
    
    def update_positive_samples(self,
                                cooccurrences: List[Tuple[str, str]] = None,
                                event_entities: Dict[str, List[str]] = None,
                                entity_timestamps: Dict[str, List[datetime]] = None):
        """更新正样本信息
        
        Args:
            cooccurrences: 共现实体对列表
            event_entities: 事件-实体映射
            entity_timestamps: 实体时间戳映射
        """
        if cooccurrences:
            for e1, e2 in cooccurrences:
                self.positive_constructor.add_cooccurrence(e1, e2)
        
        if event_entities:
            for event_id, ents in event_entities.items():
                self.positive_constructor.add_event_entities(event_id, ents)
        
        if entity_timestamps:
            for entity, timestamps in entity_timestamps.items():
                for ts in timestamps:
                    self.positive_constructor.add_entity_timestamp(entity, ts)
    
    def update_kg_edges(self, edges: List[Tuple[str, str]]):
        """更新KG边信息（用于硬负样本采样）"""
        self.negative_sampler.update_kg_edges(edges)
    
    def get_statistics(self) -> Dict:
        """获取训练统计信息"""
        stats = {
            'global_step': self.global_step,
            'total_loss': float(self.total_loss),
            'avg_loss': float(np.mean(self.loss_history[-100:])) if self.loss_history else 0.0,
            'recent_losses': self.loss_history[-10:],
            'cache_stats': self.embedding_cache.get_statistics()
        }
        return stats
