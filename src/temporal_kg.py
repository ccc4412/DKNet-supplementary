# -*- coding: utf-8 -*-
"""
时间感知知识图谱模块
实现动态知识图谱构建、更新和查询功能

增强功能：
- Multi-layer KG 支持 (L1: co-occurrence, L2: temporal, L3: external)
- Temporal Decay 增量更新 (w_t = lambda^Delta_t * w_{t-1} + Delta_w)
- Edge Pruning (w < epsilon 时删除)
- Degree Cap (top-M neighbors)
- Incremental PageRank
- Approximate kNN Index 支持
"""

import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict, Counter
from loguru import logger
import json
import pickle

class TemporalKnowledgeGraph:
    """时间感知知识图谱
    
    增强版：支持多层知识图谱、时间衰减增量更新、边剪枝、
    度数上限、增量PageRank和近似kNN索引。
    """
    
    # 图层定义
    LAYER_CO_OCCURRENCE = 'L1_co_occurrence'
    LAYER_TEMPORAL = 'L2_temporal'
    LAYER_EXTERNAL = 'L3_external'
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Multi-layer KG: 每层独立图
        self.layer_graphs: Dict[str, nx.MultiDiGraph] = {
            self.LAYER_CO_OCCURRENCE: nx.MultiDiGraph(),
            self.LAYER_TEMPORAL: nx.MultiDiGraph(),
            self.LAYER_EXTERNAL: nx.MultiDiGraph(),
        }
        
        # 向后兼容：保留主图引用（合并所有层的视图）
        self.graph = self.layer_graphs[self.LAYER_CO_OCCURRENCE]
        
        self.time_windows = {}  # 时间窗口缓存
        self.entity_embeddings = {}  # 实体嵌入
        self.temporal_weights = {}  # 时间权重
        self.centrality_cache = {}  # 中心性缓存
        
        # 配置参数
        self.tau = config['model']['kg']['time_decay_factor']
        self.node_types = config['model']['kg']['node_types']
        self.edge_types = config['model']['kg']['edge_types']
        
        # Temporal decay 增量更新参数
        self.lambda_decay = config['model']['kg'].get('lambda_decay', 0.95)  # 衰减系数
        self.last_edge_weights: Dict[str, float] = {}  # 上一次的边权重缓存
        
        # Edge pruning 参数
        self.epsilon_prune = config['model']['kg'].get('epsilon_prune', 0.01)  # 剪枝阈值
        
        # Degree cap 参数
        self.degree_cap_M = config['model']['kg'].get('degree_cap_M', 50)  # top-M neighbors
        
        # Incremental PageRank 参数
        self.pagerank_scores: Dict[str, float] = {}
        self.pagerank_damping = config['model']['kg'].get('pagerank_damping', 0.85)
        self.pagerank_iterations = config['model']['kg'].get('pagerank_iterations', 30)
        self.pagerank_tolerance = config['model']['kg'].get('pagerank_tolerance', 1e-6)
        
        # Approximate kNN 索引
        self.knn_index = None  # 延迟初始化
        self.knn_dim = config['model']['kg'].get('knn_dim', 768)  # embedding维度
        self.knn_index_built = False
        
        logger.info("时间感知知识图谱初始化完成（增强版，支持多层KG）")
    
    def add_node(self, node_id: str, node_type: str, 
                 attributes: Dict, timestamp: datetime):
        """添加节点"""
        self.graph.add_node(node_id, 
                           node_type=node_type,
                           attributes=attributes,
                           created_at=timestamp,
                           last_updated=timestamp,
                           frequency=1)
        
        # 初始化时间权重
        self.temporal_weights[node_id] = 1.0
    
    def add_edge(self, source: str, target: str, edge_type: str,
                 weight: float, timestamp: datetime, attributes: Dict = None):
        """添加边"""
        if attributes is None:
            attributes = {}
            
        edge_id = f"{source}_{target}_{edge_type}_{timestamp.isoformat()}"
        
        self.graph.add_edge(source, target,
                           key=edge_id,
                           edge_type=edge_type,
                           weight=weight,
                           timestamp=timestamp,
                           attributes=attributes)
    
    def update_node_frequency(self, node_id: str, timestamp: datetime):
        """更新节点频率"""
        if node_id in self.graph.nodes:
            self.graph.nodes[node_id]['frequency'] += 1
            self.graph.nodes[node_id]['last_updated'] = timestamp
            
            # 更新时间权重
            self._update_temporal_weight(node_id, timestamp)
    
    def _update_temporal_weight(self, node_id: str, current_time: datetime):
        """更新时间权重"""
        if node_id not in self.graph.nodes:
            return
            
        last_updated = self.graph.nodes[node_id]['last_updated']
        delta_hours = (current_time - last_updated).total_seconds() / 3600
        
        # 时间衰减权重
        decay_weight = np.exp(-delta_hours / self.tau)
        self.temporal_weights[node_id] = decay_weight
    
    def build_from_documents(self, documents: List[Dict], 
                           data_processor, current_time: datetime):
        """从文档构建知识图谱"""
        logger.info(f"开始构建知识图谱，文档数量: {len(documents)}")
        
        entity_cooccurrence = defaultdict(int)
        entity_temporal = defaultdict(list)
        
        for doc in documents:
            text = doc.get('cleaned_text', doc.get('text', ''))
            timestamp = doc.get('created_at', current_time)
            
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            # 提取实体
            entities = data_processor.extract_entities(text)
            
            # 添加实体节点
            for entity in entities:
                if entity in self.graph.nodes:
                    self.update_node_frequency(entity, timestamp)
                else:
                    self.add_node(entity, 'entity', 
                                 {'text': entity, 'source_doc': doc.get('id', '')},
                                 timestamp)
                
                entity_temporal[entity].append(timestamp)
            
            # 构建共现关系
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    if entity1 != entity2:
                        pair = tuple(sorted([entity1, entity2]))
                        entity_cooccurrence[pair] += 1
        
        # 添加共现边
        for (entity1, entity2), count in entity_cooccurrence.items():
            if count >= 2:  # 至少共现2次
                weight = np.log(count + 1)  # 对数权重
                avg_time = current_time  # 简化处理
                
                self.add_edge(entity1, entity2, 'co_occur', 
                             weight, avg_time, {'count': count})
                self.add_edge(entity2, entity1, 'co_occur', 
                             weight, avg_time, {'count': count})
        
        # 添加时间关系
        self._build_temporal_edges(entity_temporal, current_time)
        
        logger.info(f"知识图谱构建完成 - 节点: {len(self.graph.nodes)}, 边: {len(self.graph.edges)}")
    
    def _build_temporal_edges(self, entity_temporal: Dict, current_time: datetime):
        """构建时间关系边"""
        for entity, timestamps in entity_temporal.items():
            if len(timestamps) > 1:
                # 按时间排序
                timestamps.sort()
                
                # 添加时间序列边
                for i in range(len(timestamps) - 1):
                    time_diff = (timestamps[i+1] - timestamps[i]).total_seconds() / 3600
                    if time_diff <= 24:  # 24小时内认为有时间关联
                        weight = 1.0 / (1.0 + time_diff)
                        self.add_edge(f"{entity}_{i}", f"{entity}_{i+1}", 
                                     'temporal', weight, timestamps[i+1])
    
    def get_node_centrality(self, centrality_type: str = 'pagerank') -> Dict[str, float]:
        """计算节点中心性"""
        if centrality_type in self.centrality_cache:
            return self.centrality_cache[centrality_type]
        
        # 创建无向图用于中心性计算
        undirected_graph = self.graph.to_undirected()
        
        if centrality_type == 'pagerank':
            centrality = nx.pagerank(undirected_graph, weight='weight')
        elif centrality_type == 'betweenness':
            centrality = nx.betweenness_centrality(undirected_graph, weight='weight')
        elif centrality_type == 'closeness':
            centrality = nx.closeness_centrality(undirected_graph, distance='weight')
        elif centrality_type == 'degree':
            centrality = dict(undirected_graph.degree(weight='weight'))
            # 归一化
            max_degree = max(centrality.values()) if centrality else 1
            centrality = {k: v/max_degree for k, v in centrality.items()}
        else:
            centrality = nx.pagerank(undirected_graph, weight='weight')
        
        self.centrality_cache[centrality_type] = centrality
        return centrality
    
    def get_temporal_score(self, node_id: str, current_time: datetime) -> float:
        """获取节点的时间分数"""
        if node_id not in self.graph.nodes:
            return 0.0
        
        # 更新时间权重
        self._update_temporal_weight(node_id, current_time)
        
        # 结合频率和时间衰减
        frequency = self.graph.nodes[node_id]['frequency']
        temporal_weight = self.temporal_weights.get(node_id, 0.0)
        
        return frequency * temporal_weight
    
    def get_neighbors(self, node_id: str, edge_type: str = None, 
                     max_neighbors: int = 10) -> List[Tuple[str, float]]:
        """获取邻居节点"""
        if node_id not in self.graph.nodes:
            return []
        
        neighbors = []
        
        for neighbor in self.graph.neighbors(node_id):
            # 获取边权重
            edges = self.graph[node_id][neighbor]
            total_weight = 0.0
            edge_count = 0
            
            for edge_key, edge_data in edges.items():
                if edge_type is None or edge_data.get('edge_type') == edge_type:
                    total_weight += edge_data.get('weight', 1.0)
                    edge_count += 1
            
            if edge_count > 0:
                avg_weight = total_weight / edge_count
                neighbors.append((neighbor, avg_weight))
        
        # 按权重排序
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        return neighbors[:max_neighbors]
    
    def get_subgraph(self, center_nodes: List[str], 
                    max_depth: int = 2) -> nx.Graph:
        """获取子图"""
        subgraph_nodes = set(center_nodes)
        
        # BFS扩展
        current_nodes = set(center_nodes)
        for depth in range(max_depth):
            next_nodes = set()
            for node in current_nodes:
                if node in self.graph.nodes:
                    neighbors = list(self.graph.neighbors(node))
                    next_nodes.update(neighbors)
            
            subgraph_nodes.update(next_nodes)
            current_nodes = next_nodes
        
        return self.graph.subgraph(subgraph_nodes)
    
    def calculate_semantic_similarity(self, node1: str, node2: str) -> float:
        """计算语义相似度（简化版）"""
        if node1 not in self.graph.nodes or node2 not in self.graph.nodes:
            return 0.0
        
        # 基于共同邻居的相似度
        neighbors1 = set(self.graph.neighbors(node1))
        neighbors2 = set(self.graph.neighbors(node2))
        
        if not neighbors1 and not neighbors2:
            return 0.0
        
        intersection = len(neighbors1.intersection(neighbors2))
        union = len(neighbors1.union(neighbors2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_trending_entities(self, current_time: datetime, 
                             top_k: int = 20) -> List[Tuple[str, float]]:
        """获取趋势实体"""
        entity_scores = []
        
        for node_id in self.graph.nodes:
            if self.graph.nodes[node_id]['node_type'] == 'entity':
                # 计算综合分数：中心性 + 时间分数
                centrality = self.get_node_centrality('pagerank').get(node_id, 0.0)
                temporal_score = self.get_temporal_score(node_id, current_time)
                
                combined_score = 0.6 * centrality + 0.4 * temporal_score
                entity_scores.append((node_id, combined_score))
        
        # 排序并返回top-k
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        return entity_scores[:top_k]
    
    def update_graph(self, new_documents: List[Dict], 
                    data_processor, current_time: datetime):
        """增量更新图谱"""
        logger.info(f"增量更新知识图谱，新文档数量: {len(new_documents)}")
        
        # 清除缓存
        self.centrality_cache.clear()
        
        # 构建新的关系
        self.build_from_documents(new_documents, data_processor, current_time)
        
        # 更新所有节点的时间权重
        for node_id in self.graph.nodes:
            self._update_temporal_weight(node_id, current_time)
    
    def save_graph(self, file_path: str):
        """保存图谱"""
        graph_data = {
            'nodes': dict(self.graph.nodes(data=True)),
            'edges': [(u, v, k, d) for u, v, k, d in self.graph.edges(keys=True, data=True)],
            'temporal_weights': self.temporal_weights,
            'config': self.config
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(graph_data, f)
        
        logger.info(f"知识图谱已保存到: {file_path}")
    
    def load_graph(self, file_path: str):
        """加载图谱"""
        try:
            with open(file_path, 'rb') as f:
                graph_data = pickle.load(f)
            
            # 重建图谱
            self.graph = nx.MultiDiGraph()
            
            # 添加节点
            for node_id, data in graph_data['nodes'].items():
                self.graph.add_node(node_id, **data)
            
            # 添加边
            for u, v, k, d in graph_data['edges']:
                self.graph.add_edge(u, v, key=k, **d)
            
            # 恢复时间权重
            self.temporal_weights = graph_data.get('temporal_weights', {})
            
            logger.info(f"知识图谱已从 {file_path} 加载")
            
        except Exception as e:
            logger.error(f"加载知识图谱失败: {e}")
    
    def get_statistics(self) -> Dict:
        """获取图谱统计信息"""
        stats = {
            'num_nodes': len(self.graph.nodes),
            'num_edges': len(self.graph.edges),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]) if self.graph.nodes else 0,
            'density': nx.density(self.graph),
            'num_components': nx.number_weakly_connected_components(self.graph)
        }
        
        # 节点类型分布
        node_types = Counter([data.get('node_type', 'unknown') for _, data in self.graph.nodes(data=True)])
        stats['node_type_distribution'] = dict(node_types)
        
        # 边类型分布
        edge_types = Counter([data.get('edge_type', 'unknown') for _, _, data in self.graph.edges(data=True)])
        stats['edge_type_distribution'] = dict(edge_types)
        
        # Multi-layer 统计
        stats['layer_statistics'] = {}
        for layer_name, layer_graph in self.layer_graphs.items():
            stats['layer_statistics'][layer_name] = {
                'num_nodes': len(layer_graph.nodes),
                'num_edges': len(layer_graph.edges),
            }
        
        return stats
    
    # ==================== Multi-layer KG 支持 ====================
    
    def add_node_to_layer(self, node_id: str, node_type: str,
                          attributes: Dict, timestamp: datetime,
                          layer: str = None) -> bool:
        """向指定图层添加节点
        
        Args:
            node_id: 节点ID
            node_type: 节点类型
            attributes: 节点属性
            timestamp: 时间戳
            layer: 图层名称，None表示添加到所有层
            
        Returns:
            是否成功添加
        """
        layers = [layer] if layer else list(self.layer_graphs.keys())
        
        for l in layers:
            if l not in self.layer_graphs:
                logger.warning(f"未知图层: {l}")
                continue
            self.layer_graphs[l].add_node(
                node_id,
                node_type=node_type,
                attributes=attributes,
                created_at=timestamp,
                last_updated=timestamp,
                frequency=1
            )
        
        # 初始化时间权重
        if node_id not in self.temporal_weights:
            self.temporal_weights[node_id] = 1.0
        
        return True
    
    def add_edge_to_layer(self, source: str, target: str, edge_type: str,
                          weight: float, timestamp: datetime,
                          attributes: Dict = None,
                          layer: str = LAYER_CO_OCCURRENCE):
        """向指定图层添加边
        
        Args:
            source: 源节点
            target: 目标节点
            edge_type: 边类型
            weight: 边权重
            timestamp: 时间戳
            attributes: 边属性
            layer: 目标图层
        """
        if layer not in self.layer_graphs:
            logger.warning(f"未知图层: {layer}")
            return
        
        if attributes is None:
            attributes = {}
        
        edge_id = f"{source}_{target}_{edge_type}_{layer}_{timestamp.isoformat()}"
        
        self.layer_graphs[layer].add_edge(
            source, target,
            key=edge_id,
            edge_type=edge_type,
            weight=weight,
            timestamp=timestamp,
            attributes=attributes,
            layer=layer
        )
    
    def get_merged_graph(self) -> nx.MultiDiGraph:
        """获取合并所有层的图视图
        
        Returns:
            合并后的MultiDiGraph
        """
        merged = nx.MultiDiGraph()
        for layer_name, layer_graph in self.layer_graphs.items():
            for node, data in layer_graph.nodes(data=True):
                if node not in merged:
                    merged.add_node(node, **data)
            for u, v, k, data in layer_graph.edges(keys=True, data=True):
                merged.add_edge(u, v, key=f"{k}_{layer_name}", **data)
        return merged
    
    def query_layer(self, layer: str) -> nx.MultiDiGraph:
        """查询指定图层
        
        Args:
            layer: 图层名称
            
        Returns:
            该层的图
        """
        if layer not in self.layer_graphs:
            logger.warning(f"未知图层: {layer}，返回空图")
            return nx.MultiDiGraph()
        return self.layer_graphs[layer]
    
    def cross_layer_traverse(self, node_id: str, max_depth: int = 2) -> Dict[str, List[str]]:
        """跨层遍历：从指定节点出发，在各层中收集邻居
        
        Args:
            node_id: 起始节点
            max_depth: 遍历深度
            
        Returns:
            {layer_name: [neighbor_nodes]} 各层的邻居节点
        """
        result = {}
        for layer_name, layer_graph in self.layer_graphs.items():
            neighbors = set()
            current_level = {node_id}
            
            for _ in range(max_depth):
                next_level = set()
                for n in current_level:
                    if n in layer_graph:
                        next_level.update(layer_graph.neighbors(n))
                neighbors.update(next_level)
                current_level = next_level
            
            neighbors.discard(node_id)
            result[layer_name] = list(neighbors)
        
        return result
    
    # ==================== Temporal Decay 增量更新 ====================
    
    def incremental_update_edge(self, source: str, target: str,
                                delta_w: float, timestamp: datetime,
                                edge_type: str = 'co_occur',
                                layer: str = LAYER_CO_OCCURRENCE):
        """Temporal Decay 增量更新边权重
        
        公式: w_t = lambda^(Delta_t) * w_{t-1} + Delta_w
        
        其中:
        - lambda: 衰减系数 (0 < lambda <= 1)
        - Delta_t: 距上次更新的时间差（小时）
        - w_{t-1}: 上一次的边权重
        - Delta_w: 本次增量权重
        
        Args:
            source: 源节点
            target: 目标节点
            delta_w: 增量权重
            timestamp: 当前时间戳
            edge_type: 边类型
            layer: 目标图层
        """
        edge_key = f"{source}_{target}_{edge_type}_{layer}"
        
        # 获取上一次的权重和时间
        prev_weight = self.last_edge_weights.get(edge_key, 0.0)
        
        # 计算时间差
        last_update_time = self._get_last_edge_update_time(edge_key, timestamp)
        delta_hours = max((timestamp - last_update_time).total_seconds() / 3600, 0.0)
        
        # 应用时间衰减增量更新公式
        decay_factor = self.lambda_decay ** delta_hours
        new_weight = decay_factor * prev_weight + delta_w
        
        # 确保权重非负
        new_weight = max(new_weight, 0.0)
        
        # 更新缓存
        self.last_edge_weights[edge_key] = new_weight
        self._set_last_edge_update_time(edge_key, timestamp)
        
        # 更新图中的边
        self.add_edge_to_layer(
            source, target, edge_type,
            new_weight, timestamp,
            attributes={'delta_w': delta_w, 'decay_factor': decay_factor},
            layer=layer
        )
        
        logger.debug(f"增量更新边 {source}->{target}: "
                      f"w_{prev_weight:.4f} -> w_{new_weight:.4f} "
                      f"(lambda^{delta_hours:.1f}={decay_factor:.4f}, +{delta_w:.4f})")
    
    def _get_last_edge_update_time(self, edge_key: str, default_time: datetime) -> datetime:
        """获取边上次更新时间"""
        # 使用实例属性而非类属性
        if not hasattr(self, '_edge_update_times_store'):
            self._edge_update_times_store = {}
        return self._edge_update_times_store.get(edge_key, default_time)
    
    def _set_last_edge_update_time(self, edge_key: str, timestamp: datetime):
        """设置边上次更新时间"""
        if not hasattr(self, '_edge_update_times_store'):
            self._edge_update_times_store = {}
        self._edge_update_times_store[edge_key] = timestamp
    
    # ==================== Edge Pruning ====================
    
    def prune_edges(self, epsilon: float = None, layer: str = None) -> int:
        """Edge Pruning: 删除权重低于阈值的边
        
        当 w < epsilon 时，删除该边以保持图谱稀疏性和质量。
        
        Args:
            epsilon: 剪枝阈值，None则使用self.epsilon_prune
            layer: 指定图层，None则对所有层执行
            
        Returns:
            被删除的边数量
        """
        if epsilon is None:
            epsilon = self.epsilon_prune
        
        layers = [layer] if layer else list(self.layer_graphs.keys())
        total_pruned = 0
        
        for l in layers:
            if l not in self.layer_graphs:
                continue
            
            graph = self.layer_graphs[l]
            edges_to_remove = []
            
            for u, v, k, data in graph.edges(keys=True, data=True):
                weight = data.get('weight', 0.0)
                if weight < epsilon:
                    edges_to_remove.append((u, v, k))
            
            for u, v, k in edges_to_remove:
                graph.remove_edge(u, v, k)
                total_pruned += 1
        
        if total_pruned > 0:
            self.centrality_cache.clear()
            logger.info(f"Edge pruning完成: 删除 {total_pruned} 条边 (epsilon={epsilon})")
        
        return total_pruned
    
    # ==================== Degree Cap ====================
    
    def apply_degree_cap(self, M: int = None, layer: str = None) -> int:
        """Degree Cap: 限制每个节点的邻居数量为top-M
        
        对每个节点，只保留权重最高的M条边，删除其余边。
        用于控制图谱规模，防止hub节点过度膨胀。
        
        Args:
            M: 最大邻居数，None则使用self.degree_cap_M
            layer: 指定图层，None则对所有层执行
            
        Returns:
            被删除的边数量
        """
        if M is None:
            M = self.degree_cap_M
        
        layers = [layer] if layer else list(self.layer_graphs.keys())
        total_removed = 0
        
        for l in layers:
            if l not in self.layer_graphs:
                continue
            
            graph = self.layer_graphs[l]
            
            for node in list(graph.nodes()):
                # 获取该节点所有边的权重
                edges_with_weights = []
                for neighbor in graph.neighbors(node):
                    edge_data = graph[node][neighbor]
                    for edge_key, edge_info in edge_data.items():
                        w = edge_info.get('weight', 0.0)
                        edges_with_weights.append((neighbor, edge_key, w))
                
                if len(edges_with_weights) <= M:
                    continue
                
                # 按权重降序排序，保留top-M
                edges_with_weights.sort(key=lambda x: x[2], reverse=True)
                edges_to_remove = edges_with_weights[M:]
                
                for neighbor, edge_key, _ in edges_to_remove:
                    try:
                        graph.remove_edge(node, neighbor, edge_key)
                        total_removed += 1
                    except nx.NetworkXError:
                        pass
        
        if total_removed > 0:
            self.centrality_cache.clear()
            logger.info(f"Degree cap完成: 删除 {total_removed} 条边 (M={M})")
        
        return total_removed
    
    # ==================== Incremental PageRank ====================
    
    def compute_pagerank(self, reset: bool = False) -> Dict[str, float]:
        """计算PageRank（支持增量更新）
        
        首次调用时使用networkx的标准PageRank算法。
        后续调用时以上一次的结果为初始向量，加速收敛。
        
        Args:
            reset: 是否重置为均匀初始分布
            
        Returns:
            {node_id: pagerank_score} 字典
        """
        # 使用合并图计算
        merged = self.get_merged_graph()
        
        if not merged.nodes():
            return {}
        
        if reset or not self.pagerank_scores:
            # 首次计算或重置：使用标准PageRank
            self.pagerank_scores = nx.pagerank(
                merged,
                alpha=self.pagerank_damping,
                max_iter=self.pagerank_iterations,
                tol=self.pagerank_tolerance,
                weight='weight'
            )
            logger.info(f"PageRank计算完成（全量），节点数: {len(self.pagerank_scores)}")
        else:
            # 增量更新：以上一次结果作为personalization向量
            # 过滤掉已删除的节点，添加新节点
            existing_nodes = set(merged.nodes())
            prev_nodes = set(self.pagerank_scores.keys())
            
            # 新节点赋予平均初始值
            avg_pr = np.mean(list(self.pagerank_scores.values())) if self.pagerank_scores else 1.0 / len(existing_nodes)
            
            personalization = {}
            for node in existing_nodes:
                if node in self.pagerank_scores:
                    personalization[node] = self.pagerank_scores[node]
                else:
                    personalization[node] = avg_pr
            
            # 归一化personalization
            total = sum(personalization.values())
            if total > 0:
                personalization = {k: v / total for k, v in personalization.items()}
            
            self.pagerank_scores = nx.pagerank(
                merged,
                alpha=self.pagerank_damping,
                max_iter=self.pagerank_iterations,
                tol=self.pagerank_tolerance,
                weight='weight',
                personalization=personalization
            )
            
            new_nodes = existing_nodes - prev_nodes
            logger.info(f"PageRank增量更新完成，节点数: {len(self.pagerank_scores)}，"
                         f"新增节点: {len(new_nodes)}")
        
        # 更新缓存
        self.centrality_cache['pagerank'] = self.pagerank_scores
        
        return self.pagerank_scores
    
    def incremental_pagerank_step(self, new_edges: List[Tuple[str, str, float]]):
        """单步增量PageRank更新
        
        当只有少量边变化时，使用幂迭代法的少量步骤进行近似更新，
        避免完整重新计算。
        
        Args:
            new_edges: 新增/变化的边列表 [(source, target, weight), ...]
        """
        if not self.pagerank_scores:
            self.compute_pagerank()
            return
        
        merged = self.get_merged_graph()
        n = len(merged.nodes())
        if n == 0:
            return
        
        d = self.pagerank_damping
        
        # 构建出度映射
        out_degree = defaultdict(float)
        for node in merged.nodes():
            out_degree[node] = sum(
                data.get('weight', 1.0)
                for _, _, data in merged.out_edges(node, data=True)
            )
        
        # 执行少量幂迭代步骤
        for _ in range(5):  # 5步通常足够收敛小幅变化
            new_scores = {}
            for node in merged.nodes():
                rank = (1 - d) / n
                
                for predecessor in merged.predecessors(node):
                    edge_weight = sum(
                        data.get('weight', 1.0)
                        for _, _, data in merged[predecessor][node].items()
                    )
                    if out_degree[predecessor] > 0:
                        rank += d * self.pagerank_scores.get(predecessor, 0) * edge_weight / out_degree[predecessor]
                
                new_scores[node] = rank
            
            self.pagerank_scores = new_scores
        
        self.centrality_cache['pagerank'] = self.pagerank_scores
        logger.debug(f"增量PageRank单步更新完成，影响边数: {len(new_edges)}")
    
    # ==================== Approximate kNN Index ====================
    
    def build_knn_index(self, embeddings: Dict[str, np.ndarray] = None):
        """构建近似kNN索引
        
        支持两种后端：
        1. scikit-learn的NearestNeighbors（默认，无需额外依赖）
        2. FAISS（如果可用，性能更优）
        
        Args:
            embeddings: {node_id: embedding_vector} 字典
                        None则使用self.entity_embeddings
        """
        if embeddings is None:
            embeddings = self.entity_embeddings
        
        if not embeddings:
            logger.warning("无实体embedding，无法构建kNN索引")
            return
        
        node_ids = list(embeddings.keys())
        vectors = np.array([embeddings[nid] for nid in node_ids])
        
        try:
            # 尝试使用FAISS（如果已安装）
            import faiss
            dim = vectors.shape[1]
            index = faiss.IndexFlatIP(dim)  # 内积索引
            faiss.normalize_L2(vectors)  # 归一化后内积=余弦相似度
            index.add(vectors.astype(np.float32))
            self.knn_index = {
                'backend': 'faiss',
                'index': index,
                'node_ids': node_ids,
                'vectors': vectors
            }
            logger.info(f"FAISS kNN索引构建完成，向量数: {len(node_ids)}，维度: {dim}")
            
        except ImportError:
            # 回退到scikit-learn
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='auto')
            nn.fit(vectors)
            self.knn_index = {
                'backend': 'sklearn',
                'model': nn,
                'node_ids': node_ids,
                'vectors': vectors
            }
            logger.info(f"sklearn kNN索引构建完成，向量数: {len(node_ids)}")
        
        self.knn_index_built = True
    
    def query_knn(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """查询近似kNN
        
        Args:
            query_embedding: 查询向量
            k: 返回的最近邻数量
            
        Returns:
            [(node_id, similarity), ...] 按相似度降序排列
        """
        if not self.knn_index_built or self.knn_index is None:
            logger.warning("kNN索引未构建，请先调用build_knn_index")
            return []
        
        query = query_embedding.reshape(1, -1)
        
        if self.knn_index['backend'] == 'faiss':
            import faiss
            q = query.astype(np.float32)
            faiss.normalize_L2(q)
            distances, indices = self.knn_index['index'].search(q, k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.knn_index['node_ids']):
                    results.append((self.knn_index['node_ids'][idx], float(dist)))
            return results
            
        else:  # sklearn
            distances, indices = self.knn_index['model'].kneighbors(query, n_neighbors=k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.knn_index['node_ids']):
                    similarity = 1.0 - dist  # cosine距离转相似度
                    results.append((self.knn_index['node_ids'][idx], float(similarity)))
            return results
    
    def update_knn_index(self, new_embeddings: Dict[str, np.ndarray]):
        """增量更新kNN索引
        
        Args:
            new_embeddings: 新增的 {node_id: embedding_vector}
        """
        if not self.knn_index_built:
            self.build_knn_index(new_embeddings)
            return
        
        # 合并新embedding到现有数据
        self.entity_embeddings.update(new_embeddings)
        self.build_knn_index(self.entity_embeddings)
        
        logger.info(f"kNN索引增量更新完成，新增向量: {len(new_embeddings)}")