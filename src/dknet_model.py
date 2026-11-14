# -*- coding: utf-8 -*-
"""
DKNet主模型类
整合所有组件，实现完整的动态关键词生成流程
"""

import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np
from loguru import logger

from .data_processor import DataProcessor
from .temporal_kg import TemporalKnowledgeGraph
from .llm_candidate_generator import LLMCandidateGenerator
from .adaptive_fusion import AdaptiveFusionAlgorithm
from .mmr_optimizer import MMROptimizer

class DKNet:
    """
    DKNet主模型类
    整合时间感知知识图谱、LLM候选生成、自适应融合和MMR优化
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化DKNet模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model_config = config.get('model', {})
        
        # 初始化各个组件
        self.data_processor = DataProcessor(config)
        self.temporal_kg = TemporalKnowledgeGraph(config)
        self.llm_generator = LLMCandidateGenerator(config)
        self.adaptive_fusion = AdaptiveFusionAlgorithm(config)
        self.mmr_optimizer = MMROptimizer(config)
        
        logger.info("DKNet模型初始化完成")
    
    def run(self, documents: List[Dict[str, Any]], 
            ground_truth_keywords: Optional[List[List[str]]] = None) -> Tuple[List[List[str]], Dict[str, Any]]:
        """
        运行DKNet完整流程
        
        Args:
            documents: 文档列表，每个文档包含text, timestamp等字段
            ground_truth_keywords: 真实关键词列表（用于训练）
            
        Returns:
            Tuple[List[List[str]], Dict[str, Any]]: (预测关键词列表, 统计信息)
        """
        logger.info(f"开始处理 {len(documents)} 个文档")
        
        all_predictions = []
        stats = {
            'total_documents': len(documents),
            'kg_stats': [],
            'fusion_weights': [],
            'mmr_scores': [],
            'processing_times': []
        }
        
        for i, doc in enumerate(documents):
            start_time = datetime.now()
            
            # 1. 数据预处理
            processed_text = self.data_processor.preprocess_text(doc['text'])
            
            # 2. 更新时间感知知识图谱
            self.temporal_kg.build_from_documents(
                [doc], 
                self.data_processor,
                doc.get('timestamp', datetime.now())
            )
            
            # 3. LLM候选生成
            llm_candidates = self.llm_generator.generate_candidates_with_context(
                processed_text,
                context_texts=[d['text'] for d in documents[:i]],  # 使用之前的文档作为上下文
                top_k=self.config.get('model', {}).get('llm', {}).get('max_candidates', 20)
            )
            
            # 4. 知识图谱候选生成
            kg_candidates = self._get_kg_candidates(processed_text, doc.get('timestamp'))
            
            # 5. 自适应权重融合
            # 获取所有候选词并对齐分数
            kg_keys = list(kg_candidates.keys())
            llm_keys = [word for word, _, _ in llm_candidates]
            all_candidates = list(set(kg_keys + llm_keys))
            
            # 为所有候选词创建对齐的分数数组
            aligned_kg_scores = []
            aligned_llm_scores = []
            
            for candidate in all_candidates:
                kg_score = kg_candidates.get(candidate, 0.0)
                llm_score = next((score for word, score, _ in llm_candidates if word == candidate), 0.0)
                aligned_kg_scores.append(kg_score)
                aligned_llm_scores.append(llm_score)
            
            kg_scores_array = np.array(aligned_kg_scores)
            llm_scores_array = np.array(aligned_llm_scores)
            
            if ground_truth_keywords is not None and i < len(ground_truth_keywords):
                # 训练模式：使用真实标签更新权重
                fusion_scores_array, _ = self.adaptive_fusion.batch_adaptive_fusion(
                    kg_scores_array, llm_scores_array, ground_truth_keywords[i]
                )
            else:
                # 推理模式：使用当前权重
                fusion_scores_array, _ = self.adaptive_fusion.batch_adaptive_fusion(
                    kg_scores_array, llm_scores_array, update_weights=False
                )
            
            # 将融合分数与候选词对应
            fusion_scores = {candidate: score for candidate, score in zip(all_candidates, fusion_scores_array)}
            
            # 6. MMR多样性优化
            # 将融合分数转换为候选词列表格式
            candidates = [(word, score) for word, score in fusion_scores.items()]
            
            # 获取候选词的嵌入表示
            embeddings = {}
            for word, _ in candidates:
                # 使用数据处理器获取词嵌入（将单词作为文本处理）
                embedding = self.data_processor.get_text_embedding(word)
                if embedding is not None:
                    embeddings[word] = embedding
            
            # 获取文档嵌入作为查询嵌入
            query_embedding = self.data_processor.get_text_embedding(processed_text)
            
            final_keywords = self.mmr_optimizer.mmr_diversify(
                candidates,
                embeddings,
                query_embedding,
                top_k=self.config.get('experiment', {}).get('top_k', 10)
            )
            
            all_predictions.append(final_keywords)
            
            # 记录统计信息
            processing_time = (datetime.now() - start_time).total_seconds()
            stats['processing_times'].append(processing_time)
            stats['kg_stats'].append(self.temporal_kg.get_statistics())
            stats['fusion_weights'].append(self.adaptive_fusion.get_weight_statistics())
            
            if i % 10 == 0:
                logger.info(f"已处理 {i+1}/{len(documents)} 个文档")
        
        # 计算平均统计信息
        stats['avg_processing_time'] = np.mean(stats['processing_times'])
        stats['total_processing_time'] = sum(stats['processing_times'])
        
        logger.info(f"DKNet处理完成，平均处理时间: {stats['avg_processing_time']:.3f}秒")
        
        return all_predictions, stats
    
    def _get_kg_candidates(self, text: str, timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """
        从知识图谱获取候选关键词
        
        Args:
            text: 处理后的文本
            timestamp: 时间戳
            
        Returns:
            Dict[str, float]: 候选关键词及其分数
        """
        # 提取文本中的实体
        entities = self.data_processor.extract_entities(text)
        
        kg_candidates = {}
        
        for entity in entities:
            if entity in self.temporal_kg.graph.nodes:
                # 计算时间分数
                time_score = self.temporal_kg.get_temporal_score(
                    entity, timestamp or datetime.now()
                )
                
                # 计算中心性分数
                centrality_scores = self.temporal_kg.get_node_centrality('pagerank')
                centrality_score = centrality_scores.get(entity, 0.0)
                
                # 综合分数
                kg_candidates[entity] = 0.6 * time_score + 0.4 * centrality_score
                
                # 添加相关邻居节点
                neighbors = self.temporal_kg.get_neighbors(entity, max_neighbors=3)
                for neighbor, similarity in neighbors:
                    if neighbor not in kg_candidates:
                        neighbor_score = kg_candidates[entity] * similarity * 0.5
                        kg_candidates[neighbor] = neighbor_score
        
        return kg_candidates
    
    def save_model(self, save_path: str):
        """
        保存模型状态
        
        Args:
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_state = {
            'config': self.config,
            'temporal_kg': self.temporal_kg,
            'adaptive_fusion_weights': self.adaptive_fusion.get_weights(),
            'model_version': '1.0'
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_state, f)
        
        logger.info(f"模型已保存到: {save_path}")
    
    def load_model(self, load_path: str):
        """
        加载模型状态
        
        Args:
            load_path: 加载路径
        """
        with open(load_path, 'rb') as f:
            model_state = pickle.load(f)
        
        self.config = model_state['config']
        self.temporal_kg = model_state['temporal_kg']
        
        # 恢复自适应融合权重
        if 'adaptive_fusion_weights' in model_state:
            weights = model_state['adaptive_fusion_weights']
            self.adaptive_fusion.llm_weight = weights.get('llm_weight', 0.5)
            self.adaptive_fusion.kg_weight = weights.get('kg_weight', 0.5)
        
        logger.info(f"模型已从 {load_path} 加载")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            'model_name': 'DKNet',
            'version': '1.0',
            'config': self.config,
            'kg_stats': self.temporal_kg.get_stats(),
            'fusion_weights': self.adaptive_fusion.get_weights(),
            'components': {
                'data_processor': type(self.data_processor).__name__,
                'temporal_kg': type(self.temporal_kg).__name__,
                'llm_generator': type(self.llm_generator).__name__,
                'adaptive_fusion': type(self.adaptive_fusion).__name__,
                'mmr_optimizer': type(self.mmr_optimizer).__name__
            }
        }