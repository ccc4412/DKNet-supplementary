# -*- coding: utf-8 -*-
"""
基线方法对比模块
实现TF-IDF、TextRank、KeyBERT等传统关键词提取方法
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional, Union
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
import jieba
import jieba.posseg as pseg
from transformers import AutoTokenizer, AutoModel
import torch
from loguru import logger
import re
import math
from datetime import datetime

class BaselineMethod:
    """基线方法基类"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.name = "BaselineMethod"
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """提取关键词
        
        Args:
            text: 输入文本
            top_k: 返回的关键词数量
            
        Returns:
            关键词及其分数的列表
        """
        raise NotImplementedError
    
    def batch_extract_keywords(self, texts: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """批量提取关键词"""
        results = []
        for text in texts:
            keywords = self.extract_keywords(text, top_k)
            results.append(keywords)
        return results


class TFIDFMethod(BaselineMethod):
    """TF-IDF方法"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.name = "TF-IDF"
        self.vectorizer = None
        self.stop_words = self._load_stop_words()
        
        # 获取TF-IDF配置
        tfidf_config = self.config.get('baseline', {}).get('tfidf', {})
        self.max_features = tfidf_config.get('max_features', 1000)
        self.ngram_range = tuple(tfidf_config.get('ngram_range', [1, 2]))
        self.min_df = tfidf_config.get('min_df', 2)
        self.max_df = tfidf_config.get('max_df', 0.8)
        
        logger.info("TF-IDF方法初始化完成")
    
    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        stop_words = set()
        try:
            # 尝试加载停用词文件
            with open('data/stop_words.txt', 'r', encoding='utf-8') as f:
                stop_words = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            # 使用默认停用词
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        return stop_words
    
    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        # 分词
        words = jieba.lcut(text)
        
        # 过滤
        filtered_words = []
        for word in words:
            # 去除停用词、标点符号、单字符
            if (len(word) > 1 and 
                word not in self.stop_words and 
                not re.match(r'^[\W\d]+$', word)):
                filtered_words.append(word)
        
        return filtered_words
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用TF-IDF提取关键词"""
        words = self._preprocess_text(text)
        
        if not words:
            return []
        
        # 计算词频
        word_freq = Counter(words)
        total_words = len(words)
        
        # 计算TF
        tf_scores = {word: freq / total_words for word, freq in word_freq.items()}
        
        # 简化的IDF计算（假设文档集合）
        # 在实际应用中，应该使用整个语料库计算IDF
        unique_words = len(word_freq)
        idf_scores = {word: math.log(unique_words / freq) for word, freq in word_freq.items()}
        
        # 计算TF-IDF
        tfidf_scores = {word: tf_scores[word] * idf_scores[word] for word in tf_scores}
        
        # 排序并返回top_k
        sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_keywords[:top_k]
    
    def fit_corpus(self, corpus: List[str]):
        """在语料库上训练TF-IDF向量化器"""
        # 预处理语料库
        processed_corpus = []
        for text in corpus:
            words = self._preprocess_text(text)
            processed_corpus.append(' '.join(words))
        
        # 训练TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=list(self.stop_words)
        )
        
        self.vectorizer.fit(processed_corpus)
        logger.info(f"TF-IDF向量化器训练完成，词汇表大小: {len(self.vectorizer.vocabulary_)}")
    
    def extract_keywords_with_vectorizer(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用训练好的向量化器提取关键词"""
        if self.vectorizer is None:
            logger.warning("向量化器未训练，使用基础TF-IDF方法")
            return self.extract_keywords(text, top_k)
        
        words = self._preprocess_text(text)
        processed_text = ' '.join(words)
        
        # 计算TF-IDF向量
        tfidf_vector = self.vectorizer.transform([processed_text])
        
        # 获取特征名称和分数
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_vector.toarray()[0]
        
        # 创建词汇-分数对
        word_scores = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names)) if tfidf_scores[i] > 0]
        
        # 排序并返回top_k
        sorted_keywords = sorted(word_scores, key=lambda x: x[1], reverse=True)
        
        return sorted_keywords[:top_k]


class TextRankMethod(BaselineMethod):
    """TextRank方法"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.name = "TextRank"
        
        # 获取TextRank配置
        textrank_config = self.config.get('baseline', {}).get('textrank', {})
        self.window_size = textrank_config.get('window_size', 5)
        self.damping_factor = textrank_config.get('damping_factor', 0.85)
        self.max_iter = textrank_config.get('max_iter', 100)
        self.stop_words = self._load_stop_words()
        
        logger.info(f"TextRank方法初始化完成，窗口大小: {self.window_size}")
    
    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        stop_words = set()
        try:
            with open('data/stop_words.txt', 'r', encoding='utf-8') as f:
                stop_words = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        return stop_words
    
    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        # 使用词性标注
        words = pseg.cut(text)
        
        # 过滤名词、动词、形容词
        filtered_words = []
        for word, flag in words:
            if (len(word) > 1 and 
                word not in self.stop_words and 
                flag.startswith(('n', 'v', 'a')) and
                not re.match(r'^[\W\d]+$', word)):
                filtered_words.append(word)
        
        return filtered_words
    
    def _build_graph(self, words: List[str]) -> nx.Graph:
        """构建词汇图"""
        graph = nx.Graph()
        
        # 添加节点
        for word in set(words):
            graph.add_node(word)
        
        # 添加边（基于滑动窗口）
        for i in range(len(words)):
            for j in range(i + 1, min(i + self.window_size, len(words))):
                word1, word2 = words[i], words[j]
                if word1 != word2:
                    if graph.has_edge(word1, word2):
                        graph[word1][word2]['weight'] += 1
                    else:
                        graph.add_edge(word1, word2, weight=1)
        
        return graph
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用TextRank提取关键词"""
        words = self._preprocess_text(text)
        
        if len(words) < 2:
            return [(word, 1.0) for word in words[:top_k]]
        
        # 构建图
        graph = self._build_graph(words)
        
        if len(graph.nodes()) == 0:
            return []
        
        # 运行PageRank算法
        try:
            pagerank_scores = nx.pagerank(
                graph, 
                alpha=self.damping_factor, 
                max_iter=self.max_iter,
                weight='weight'
            )
        except:
            # 如果PageRank失败，使用度中心性
            pagerank_scores = nx.degree_centrality(graph)
        
        # 排序并返回top_k
        sorted_keywords = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_keywords[:top_k]


class KeyBERTMethod(BaselineMethod):
    """KeyBERT方法"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.name = "KeyBERT"
        
        # 获取KeyBERT配置
        keybert_config = self.config.get('baseline', {}).get('keybert', {})
        self.model_name = keybert_config.get('model_name', 'bert-base-chinese')
        self.use_mmr = keybert_config.get('use_mmr', True)
        self.diversity = keybert_config.get('diversity', 0.5)
        
        # 初始化BERT模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.stop_words = self._load_stop_words()
        
        logger.info(f"KeyBERT方法初始化完成，模型: {self.model_name}")
    
    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        stop_words = set()
        try:
            with open('data/stop_words.txt', 'r', encoding='utf-8') as f:
                stop_words = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        return stop_words
    
    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """获取BERT嵌入"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]标记的嵌入
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding[0]
    
    def _extract_candidates(self, text: str) -> List[str]:
        """提取候选关键词"""
        # 分词
        words = jieba.lcut(text)
        
        # 过滤
        candidates = []
        for word in words:
            if (len(word) > 1 and 
                word not in self.stop_words and 
                not re.match(r'^[\W\d]+$', word)):
                candidates.append(word)
        
        # 添加n-gram短语
        for i in range(len(candidates) - 1):
            bigram = candidates[i] + candidates[i + 1]
            if len(bigram) <= 10:  # 限制短语长度
                candidates.append(bigram)
        
        return list(set(candidates))
    
    def _mmr_selection(self, doc_embedding: np.ndarray, 
                      candidate_embeddings: Dict[str, np.ndarray],
                      candidates: List[str], 
                      top_k: int) -> List[Tuple[str, float]]:
        """MMR选择"""
        selected = []
        remaining = candidates.copy()
        
        for _ in range(min(top_k, len(candidates))):
            if not remaining:
                break
            
            best_candidate = None
            best_score = float('-inf')
            
            for candidate in remaining:
                if candidate not in candidate_embeddings:
                    continue
                
                # 计算与文档的相似度
                doc_sim = cosine_similarity(
                    candidate_embeddings[candidate].reshape(1, -1),
                    doc_embedding.reshape(1, -1)
                )[0][0]
                
                # 计算与已选择关键词的最大相似度
                max_sim = 0.0
                for selected_word, _ in selected:
                    if selected_word in candidate_embeddings:
                        sim = cosine_similarity(
                            candidate_embeddings[candidate].reshape(1, -1),
                            candidate_embeddings[selected_word].reshape(1, -1)
                        )[0][0]
                        max_sim = max(max_sim, sim)
                
                # MMR分数
                mmr_score = (1 - self.diversity) * doc_sim - self.diversity * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected.append((best_candidate, best_score))
                remaining.remove(best_candidate)
        
        return selected
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用KeyBERT提取关键词"""
        # 获取文档嵌入
        doc_embedding = self._get_bert_embedding(text)
        
        # 提取候选词
        candidates = self._extract_candidates(text)
        
        if not candidates:
            return []
        
        # 获取候选词嵌入
        candidate_embeddings = {}
        for candidate in candidates:
            try:
                embedding = self._get_bert_embedding(candidate)
                candidate_embeddings[candidate] = embedding
            except:
                continue
        
        if not candidate_embeddings:
            return []
        
        if self.use_mmr:
            # 使用MMR选择
            return self._mmr_selection(doc_embedding, candidate_embeddings, candidates, top_k)
        else:
            # 直接基于相似度排序
            similarities = []
            for candidate, embedding in candidate_embeddings.items():
                sim = cosine_similarity(
                    embedding.reshape(1, -1),
                    doc_embedding.reshape(1, -1)
                )[0][0]
                similarities.append((candidate, sim))
            
            # 排序并返回top_k
            sorted_keywords = sorted(similarities, key=lambda x: x[1], reverse=True)
            return sorted_keywords[:top_k]


class LDAMethod(BaselineMethod):
    """LDA主题模型方法"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.name = "LDA"
        
        # 获取LDA配置
        lda_config = self.config.get('baseline', {}).get('lda', {})
        self.n_topics = lda_config.get('n_topics', 10)
        self.max_iter = lda_config.get('max_iter', 100)
        self.lda_model = None
        self.vectorizer = None
        self.stop_words = self._load_stop_words()
        
        logger.info(f"LDA方法初始化完成，主题数: {self.n_topics}")
    
    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        stop_words = set()
        try:
            with open('data/stop_words.txt', 'r', encoding='utf-8') as f:
                stop_words = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        return stop_words
    
    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        words = jieba.lcut(text)
        
        filtered_words = []
        for word in words:
            if (len(word) > 1 and 
                word not in self.stop_words and 
                not re.match(r'^[\W\d]+$', word)):
                filtered_words.append(word)
        
        return filtered_words
    
    def fit_corpus(self, corpus: List[str]):
        """在语料库上训练LDA模型"""
        # 预处理语料库
        processed_corpus = []
        for text in corpus:
            words = self._preprocess_text(text)
            processed_corpus.append(' '.join(words))
        
        # 创建TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=list(self.stop_words),
            ngram_range=(1, 2)
        )
        
        # 向量化
        tfidf_matrix = self.vectorizer.fit_transform(processed_corpus)
        
        # 训练LDA模型
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=self.max_iter,
            random_state=42
        )
        
        self.lda_model.fit(tfidf_matrix)
        
        logger.info(f"LDA模型训练完成，主题数: {self.n_topics}")
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用LDA提取关键词"""
        if self.lda_model is None or self.vectorizer is None:
            logger.warning("LDA模型未训练，无法提取关键词")
            return []
        
        words = self._preprocess_text(text)
        processed_text = ' '.join(words)
        
        # 向量化
        tfidf_vector = self.vectorizer.transform([processed_text])
        
        # 获取主题分布
        topic_distribution = self.lda_model.transform(tfidf_vector)[0]
        
        # 获取主要主题
        main_topic = np.argmax(topic_distribution)
        
        # 获取主题词汇
        feature_names = self.vectorizer.get_feature_names_out()
        topic_words = self.lda_model.components_[main_topic]
        
        # 获取top词汇
        top_indices = topic_words.argsort()[-top_k:][::-1]
        keywords = [(feature_names[i], topic_words[i]) for i in top_indices]
        
        return keywords


class YAKEMethod(BaselineMethod):
    """YAKE方法"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.name = "YAKE"
        
        # 获取YAKE配置
        yake_config = self.config.get('baseline', {}).get('yake', {})
        self.window_size = yake_config.get('window_size', 3)
        self.max_ngram_size = yake_config.get('max_ngram_size', 3)
        self.stop_words = self._load_stop_words()
        
        logger.info(f"YAKE方法初始化完成，窗口大小: {self.window_size}")
    
    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        stop_words = set()
        try:
            with open('data/stop_words.txt', 'r', encoding='utf-8') as f:
                stop_words = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        return stop_words
    
    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        words = jieba.lcut(text)
        
        filtered_words = []
        for word in words:
            if (len(word) > 1 and 
                word not in self.stop_words and 
                not re.match(r'^[\W\d]+$', word)):
                filtered_words.append(word)
        
        return filtered_words
    
    def _calculate_word_features(self, words: List[str]) -> Dict[str, Dict]:
        """计算词汇特征"""
        word_features = defaultdict(lambda: {
            'freq': 0,
            'positions': [],
            'left_contexts': [],
            'right_contexts': []
        })
        
        for i, word in enumerate(words):
            word_features[word]['freq'] += 1
            word_features[word]['positions'].append(i)
            
            # 左上下文
            if i > 0:
                word_features[word]['left_contexts'].append(words[i-1])
            
            # 右上下文
            if i < len(words) - 1:
                word_features[word]['right_contexts'].append(words[i+1])
        
        return word_features
    
    def _calculate_yake_score(self, candidate: str, word_features: Dict, total_words: int) -> float:
        """计算YAKE分数"""
        words = candidate.split()
        
        # 词频特征
        freq_sum = sum(word_features[word]['freq'] for word in words if word in word_features)
        
        # 位置特征
        position_sum = 0
        for word in words:
            if word in word_features:
                positions = word_features[word]['positions']
                position_sum += sum(pos for pos in positions)
        
        # 上下文多样性
        context_diversity = 0
        for word in words:
            if word in word_features:
                left_contexts = set(word_features[word]['left_contexts'])
                right_contexts = set(word_features[word]['right_contexts'])
                context_diversity += len(left_contexts) + len(right_contexts)
        
        # YAKE分数（越小越好）
        if freq_sum == 0:
            return float('inf')
        
        score = (position_sum / total_words) / (freq_sum * (context_diversity + 1))
        
        return score
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用YAKE提取关键词"""
        words = self._preprocess_text(text)
        
        if not words:
            return []
        
        # 计算词汇特征
        word_features = self._calculate_word_features(words)
        
        # 生成候选短语
        candidates = set()
        
        # 单词
        for word in words:
            candidates.add(word)
        
        # n-gram短语
        for n in range(2, min(self.max_ngram_size + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                candidates.add(ngram)
        
        # 计算YAKE分数
        candidate_scores = []
        for candidate in candidates:
            score = self._calculate_yake_score(candidate, word_features, len(words))
            candidate_scores.append((candidate, score))
        
        # 排序（YAKE分数越小越好）
        sorted_keywords = sorted(candidate_scores, key=lambda x: x[1])
        
        # 转换为正分数（1/score）
        result = [(word, 1.0 / (score + 1e-10)) for word, score in sorted_keywords[:top_k]]
        
        return result


class PositionRankMethod(BaselineMethod):
    """PositionRank方法 - 基于词位置加权的PageRank"""

    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.name = "PositionRank"

        positionrank_config = self.config.get('baseline', {}).get('positionrank', {})
        self.window_size = positionrank_config.get('window_size', 5)
        self.damping_factor = positionrank_config.get('damping_factor', 0.85)
        self.max_iter = positionrank_config.get('max_iter', 100)
        self.decay_factor = positionrank_config.get('decay_factor', 0.05)
        self.stop_words = self._load_stop_words()

        logger.info(f"PositionRank方法初始化完成，窗口大小: {self.window_size}，衰减因子: {self.decay_factor}")

    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        stop_words = set()
        try:
            with open('data/stop_words.txt', 'r', encoding='utf-8') as f:
                stop_words = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        return stop_words

    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理，使用词性标注过滤"""
        words = pseg.cut(text)
        filtered_words = []
        for word, flag in words:
            if (len(word) > 1 and
                word not in self.stop_words and
                flag.startswith(('n', 'v', 'a')) and
                not re.match(r'^[\W\d]+$', word)):
                filtered_words.append(word)
        return filtered_words

    def _build_weighted_graph(self, words: List[str]) -> nx.Graph:
        """构建位置加权的词汇共现图"""
        graph = nx.Graph()

        # 计算每个词的位置权重
        position_weights = {}
        for idx, word in enumerate(words):
            if word not in position_weights:
                position_weights[word] = 0.0
            position_weights[word] += 1.0 / (1.0 + idx * self.decay_factor)

        # 添加节点，节点权重为位置权重之和
        for word in set(words):
            graph.add_node(word, weight=position_weights.get(word, 1.0))

        # 添加边（基于滑动窗口），边权重考虑两端节点的位置权重
        for i in range(len(words)):
            for j in range(i + 1, min(i + self.window_size, len(words))):
                word1, word2 = words[i], words[j]
                if word1 != word2:
                    edge_weight = position_weights.get(word1, 1.0) * position_weights.get(word2, 1.0)
                    if graph.has_edge(word1, word2):
                        graph[word1][word2]['weight'] += edge_weight
                    else:
                        graph.add_edge(word1, word2, weight=edge_weight)

        return graph

    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用PositionRank提取关键词"""
        words = self._preprocess_text(text)

        if len(words) < 2:
            return [(word, 1.0) for word in words[:top_k]]

        graph = self._build_weighted_graph(words)

        if len(graph.nodes()) == 0:
            return []

        try:
            pagerank_scores = nx.pagerank(
                graph,
                alpha=self.damping_factor,
                max_iter=self.max_iter,
                weight='weight',
                personalization={n: d.get('weight', 1.0) for n, d in graph.nodes(data=True)}
            )
        except Exception:
            pagerank_scores = nx.degree_centrality(graph)

        sorted_keywords = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_k]


class RAKEMethod(BaselineMethod):
    """RAKE (Rapid Automatic Keyword Extraction) 方法"""

    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.name = "RAKE"

        rake_config = self.config.get('baseline', {}).get('rake', {})
        self.max_ngram_size = rake_config.get('max_ngram_size', 3)
        self.min_char_length = rake_config.get('min_char_length', 2)
        self.stop_words = self._load_stop_words()

        logger.info(f"RAKE方法初始化完成，最大ngram: {self.max_ngram_size}")

    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        stop_words = set()
        try:
            with open('data/stop_words.txt', 'r', encoding='utf-8') as f:
                stop_words = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        return stop_words

    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        words = jieba.lcut(text)
        filtered_words = []
        for word in words:
            if (len(word) >= self.min_char_length and
                word not in self.stop_words and
                not re.match(r'^[\W\d]+$', word)):
                filtered_words.append(word)
        return filtered_words

    def _extract_candidate_phrases(self, words: List[str]) -> List[str]:
        """提取候选短语（连续非停用词组成的1-3 gram）"""
        candidates = set()

        # 单词候选
        for word in words:
            candidates.add(word)

        # 多词短语候选
        for n in range(2, min(self.max_ngram_size + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                phrase = ''.join(words[i:i + n])
                if len(phrase) >= self.min_char_length:
                    candidates.add(phrase)

        return list(candidates)

    def _calculate_rake_score(self, candidate: str, word_freq: Dict[str, int],
                               word_degree: Dict[str, int]) -> float:
        """计算RAKE分数 = 词频 / 词度"""
        # 将候选短语拆分为单字/词（简单按字符拆分）
        chars = list(candidate)
        freq_sum = 0.0
        degree_sum = 0.0

        for char in chars:
            if char in word_freq:
                freq_sum += word_freq[char]
                degree_sum += word_degree[char]

        if degree_sum == 0:
            return 0.0

        return freq_sum / degree_sum

    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用RAKE提取关键词"""
        words = self._preprocess_text(text)

        if not words:
            return []

        # 计算词频和词度（度 = 词频 + 共现词数）
        word_freq = Counter(words)
        word_degree = defaultdict(int)

        for word in words:
            word_degree[word] += len(words)  # 简化：每个词的度等于总词数

        # 提取候选短语
        candidates = self._extract_candidate_phrases(words)

        # 计算综合分数：词频 + 位置 + 上下文多样性 + 词长
        candidate_scores = []
        total_words = len(words)

        for candidate in candidates:
            # 词频分数
            freq = 0
            for w in words:
                if w in candidate or candidate in w:
                    freq += 1
            freq_score = freq / total_words if total_words > 0 else 0

            # 位置分数（首次出现越靠前越好）
            position_score = 0.0
            for i, w in enumerate(words):
                if w in candidate or candidate in w:
                    position_score = 1.0 / (1.0 + i)
                    break

            # 上下文多样性分数
            context_set = set()
            for i, w in enumerate(words):
                if w in candidate or candidate in w:
                    if i > 0:
                        context_set.add(words[i - 1])
                    if i < len(words) - 1:
                        context_set.add(words[i + 1])
            context_score = len(context_set) / total_words if total_words > 0 else 0

            # 词长分数（适中的词长更好）
            length_score = min(len(candidate), 6) / 6.0

            # 综合分数
            rake_score = self._calculate_rake_score(candidate, word_freq, word_degree)
            total_score = 0.3 * freq_score + 0.25 * position_score + 0.2 * context_score + 0.1 * length_score + 0.15 * rake_score

            candidate_scores.append((candidate, total_score))

        sorted_keywords = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_k]


class SIFRankMethod(BaselineMethod):
    """SIFRank方法 - SIF加权 + MMR多样性选择"""

    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.name = "SIFRank"

        sifrank_config = self.config.get('baseline', {}).get('sifrank', {})
        self.model_name = sifrank_config.get('model_name', 'bert-base-chinese')
        self.diversity = sifrank_config.get('diversity', 0.5)
        self.sif_weight_param = sifrank_config.get('sif_weight_param', 1e-3)
        self.stop_words = self._load_stop_words()

        # 初始化BERT模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        logger.info(f"SIFRank方法初始化完成，模型: {self.model_name}")

    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        stop_words = set()
        try:
            with open('data/stop_words.txt', 'r', encoding='utf-8') as f:
                stop_words = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        return stop_words

    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        words = jieba.lcut(text)
        filtered_words = []
        for word in words:
            if (len(word) > 1 and
                word not in self.stop_words and
                not re.match(r'^[\W\d]+$', word)):
                filtered_words.append(word)
        return filtered_words

    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """获取BERT嵌入"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding[0]

    def _compute_sif_embedding(self, words: List[str], word_freq: Dict[str, float],
                                total_vocab: int) -> np.ndarray:
        """计算SIF（Smooth Inverse Frequency）加权句子嵌入"""
        if not words:
            return np.zeros(self.model.config.hidden_size)

        word_embeddings = []
        weights = []

        for word in words:
            try:
                emb = self._get_bert_embedding(word)
                word_embeddings.append(emb)
                # SIF权重：a / (a + p(w))，p(w)用词频近似
                pw = word_freq.get(word, 1e-10) / total_vocab
                weight = self.sif_weight_param / (self.sif_weight_param + pw)
                weights.append(weight)
            except Exception:
                continue

        if not word_embeddings:
            return np.zeros(self.model.config.hidden_size)

        word_embeddings = np.array(word_embeddings)
        weights = np.array(weights).reshape(-1, 1)

        # 加权平均
        sif_embedding = np.sum(word_embeddings * weights, axis=0) / np.sum(weights)

        # 移除第一主成分（简化版，直接返回加权平均）
        return sif_embedding

    def _mmr_selection(self, doc_embedding: np.ndarray,
                       candidate_embeddings: Dict[str, np.ndarray],
                       candidates: List[str],
                       top_k: int) -> List[Tuple[str, float]]:
        """MMR多样性选择"""
        selected = []
        remaining = candidates.copy()

        for _ in range(min(top_k, len(candidates))):
            if not remaining:
                break

            best_candidate = None
            best_score = float('-inf')

            for candidate in remaining:
                if candidate not in candidate_embeddings:
                    continue

                # 与文档的相似度
                doc_sim = cosine_similarity(
                    candidate_embeddings[candidate].reshape(1, -1),
                    doc_embedding.reshape(1, -1)
                )[0][0]

                # 与已选择关键词的最大相似度
                max_sim = 0.0
                for selected_word, _ in selected:
                    if selected_word in candidate_embeddings:
                        sim = cosine_similarity(
                            candidate_embeddings[candidate].reshape(1, -1),
                            candidate_embeddings[selected_word].reshape(1, -1)
                        )[0][0]
                        max_sim = max(max_sim, sim)

                mmr_score = (1 - self.diversity) * doc_sim - self.diversity * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate

            if best_candidate is not None:
                selected.append((best_candidate, best_score))
                remaining.remove(best_candidate)

        return selected

    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用SIFRank提取关键词"""
        words = self._preprocess_text(text)

        if not words:
            return []

        # 计算词频
        word_freq = Counter(words)
        total_vocab = len(word_freq)

        # 计算文档SIF嵌入
        doc_embedding = self._compute_sif_embedding(words, dict(word_freq), total_vocab)

        # 计算候选词的SIF嵌入
        candidate_embeddings = {}
        for word in set(words):
            try:
                emb = self._compute_sif_embedding([word], dict(word_freq), total_vocab)
                candidate_embeddings[word] = emb
            except Exception:
                continue

        if not candidate_embeddings:
            return []

        # 使用MMR进行多样性选择
        return self._mmr_selection(doc_embedding, candidate_embeddings, list(set(words)), top_k)


class BERTopicMethod(BaselineMethod):
    """BERTopic方法 - 基于BERT嵌入的主题模型"""

    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.name = "BERTopic"

        bertopic_config = self.config.get('baseline', {}).get('bertopic', {})
        self.model_name = bertopic_config.get('model_name', 'bert-base-chinese')
        self.n_clusters = bertopic_config.get('n_clusters', 10)
        self.top_k_words = bertopic_config.get('top_k_words', 10)
        self.stop_words = self._load_stop_words()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.topic_keywords = None
        self.doc_embeddings = None

        logger.info(f"BERTopic方法初始化完成，模型: {self.model_name}")

    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        stop_words = set()
        try:
            with open('data/stop_words.txt', 'r', encoding='utf-8') as f:
                stop_words = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        return stop_words

    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        words = jieba.lcut(text)
        filtered_words = []
        for word in words:
            if (len(word) > 1 and
                word not in self.stop_words and
                not re.match(r'^[\W\d]+$', word)):
                filtered_words.append(word)
        return filtered_words

    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """获取BERT嵌入"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding[0]

    def fit_corpus(self, corpus: List[str]):
        """在语料库上训练BERTopic模型"""
        # 获取文档嵌入
        self.doc_embeddings = np.array([self._get_bert_embedding(text) for text in corpus])

        # 使用KMeans聚类（替代UMAP+HDBSCAN，简化实现）
        n_clusters = min(self.n_clusters, len(corpus))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=300)
        cluster_labels = kmeans.fit_predict(self.doc_embeddings)

        # 从每个聚类中提取关键词
        self.topic_keywords = {}
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]

            if not cluster_indices:
                continue

            # 收集该聚类中所有文档的词频
            cluster_word_freq = Counter()
            for idx in cluster_indices:
                words = self._preprocess_text(corpus[idx])
                cluster_word_freq.update(words)

            # 取top词汇作为主题关键词
            top_words = cluster_word_freq.most_common(self.top_k_words)
            self.topic_keywords[cluster_id] = top_words

        logger.info(f"BERTopic模型训练完成，聚类数: {n_clusters}")

    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用BERTopic提取关键词"""
        if self.topic_keywords is None or self.doc_embeddings is None:
            logger.warning("BERTopic模型未训练，无法提取关键词")
            return []

        # 获取文档嵌入
        doc_embedding = self._get_bert_embedding(text)

        # 找到最近的聚类
        similarities = cosine_similarity(doc_embedding.reshape(1, -1), self.doc_embeddings)
        nearest_idx = np.argmax(similarities)

        # 确定该文档所属的聚类
        n_clusters = len(self.topic_keywords)
        kmeans_refit = KMeans(n_clusters=n_clusters, random_state=42, max_iter=300)
        cluster_labels = kmeans_refit.fit_predict(self.doc_embeddings)
        topic_id = cluster_labels[nearest_idx]

        # 返回该主题的关键词
        if topic_id in self.topic_keywords:
            keywords = self.topic_keywords[topic_id]
            # 归一化分数
            if keywords:
                max_freq = keywords[0][1]
                normalized = [(word, freq / max_freq) for word, freq in keywords]
                return normalized[:top_k]

        return []


class DTMMethod(BaselineMethod):
    """DTM (Dynamic Topic Model) 方法 - 动态主题模型"""

    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.name = "DTM"

        dtm_config = self.config.get('baseline', {}).get('dtm', {})
        self.n_topics = dtm_config.get('n_topics', 10)
        self.n_time_slices = dtm_config.get('n_time_slices', 5)
        self.max_iter = dtm_config.get('max_iter', 100)
        self.stop_words = self._load_stop_words()

        self.lda_models = None
        self.vectorizer = None
        self.time_slice_keywords = None

        logger.info(f"DTM方法初始化完成，主题数: {self.n_topics}，时间切片数: {self.n_time_slices}")

    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        stop_words = set()
        try:
            with open('data/stop_words.txt', 'r', encoding='utf-8') as f:
                stop_words = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        return stop_words

    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        words = jieba.lcut(text)
        filtered_words = []
        for word in words:
            if (len(word) > 1 and
                word not in self.stop_words and
                not re.match(r'^[\W\d]+$', word)):
                filtered_words.append(word)
        return filtered_words

    def fit_corpus(self, corpus: List[str], timestamps: List = None):
        """在语料库上训练动态主题模型

        Args:
            corpus: 文档列表
            timestamps: 可选的时间戳列表，用于按时间切片
        """
        if not corpus:
            logger.warning("语料库为空，无法训练DTM模型")
            return

        # 按时间切片分组
        if timestamps is not None and len(timestamps) == len(corpus):
            # 按时间戳排序并分组
            sorted_pairs = sorted(zip(timestamps, corpus))
            corpus = [text for _, text in sorted_pairs]

        # 将语料库均匀切分为时间切片
        slice_size = max(1, len(corpus) // self.n_time_slices)
        time_slices = []
        for i in range(0, len(corpus), slice_size):
            time_slices.append(corpus[i:i + slice_size])

        # 为每个时间切片训练独立的LDA模型
        self.lda_models = []
        self.time_slice_keywords = {}

        # 创建共享的向量化器
        all_processed = []
        for text in corpus:
            words = self._preprocess_text(text)
            all_processed.append(' '.join(words))

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=list(self.stop_words),
            ngram_range=(1, 2)
        )
        self.vectorizer.fit(all_processed)

        for t, slice_corpus in enumerate(time_slices):
            processed_slice = []
            for text in slice_corpus:
                words = self._preprocess_text(text)
                processed_slice.append(' '.join(words))

            tfidf_matrix = self.vectorizer.transform(processed_slice)

            n_topics = min(self.n_topics, len(slice_corpus))
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=self.max_iter,
                random_state=42
            )
            lda.fit(tfidf_matrix)
            self.lda_models.append(lda)

            # 提取每个时间切片的主题关键词
            feature_names = self.vectorizer.get_feature_names_out()
            slice_keywords = []
            for topic_id in range(n_topics):
                topic_words = lda.components_[topic_id]
                top_indices = topic_words.argsort()[-10:][::-1]
                keywords = [(feature_names[i], float(topic_words[i])) for i in top_indices]
                slice_keywords.append(keywords)

            self.time_slice_keywords[t] = slice_keywords

        logger.info(f"DTM模型训练完成，时间切片数: {len(self.lda_models)}")

    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用DTM提取关键词（使用最后一个时间切片的主题）"""
        if self.lda_models is None or self.vectorizer is None:
            logger.warning("DTM模型未训练，无法提取关键词")
            return []

        # 使用最后一个时间切片的LDA模型
        lda_model = self.lda_models[-1]

        words = self._preprocess_text(text)
        processed_text = ' '.join(words)

        tfidf_vector = self.vectorizer.transform([processed_text])
        topic_distribution = lda_model.transform(tfidf_vector)[0]
        main_topic = np.argmax(topic_distribution)

        feature_names = self.vectorizer.get_feature_names_out()
        topic_words = lda_model.components_[main_topic]
        top_indices = topic_words.argsort()[-top_k:][::-1]
        keywords = [(feature_names[i], float(topic_words[i])) for i in top_indices]

        return keywords


class PromptLLMMethod(BaselineMethod):
    """Prompt-LLM方法 - 使用BERT + Prompt模板生成关键词"""

    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.name = "Prompt-LLM"

        prompt_config = self.config.get('baseline', {}).get('prompt_llm', {})
        self.model_name = prompt_config.get('model_name', 'bert-base-chinese')
        self.prompt_template = prompt_config.get(
            'prompt_template',
            '请从以下文本中提取关键词：{text}'
        )
        self.stop_words = self._load_stop_words()

        # 初始化BERT模型（用于文本理解和评分）
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        logger.info(f"Prompt-LLM方法初始化完成，模型: {self.model_name}")

    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        stop_words = set()
        try:
            with open('data/stop_words.txt', 'r', encoding='utf-8') as f:
                stop_words = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        return stop_words

    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        words = jieba.lcut(text)
        filtered_words = []
        for word in words:
            if (len(word) > 1 and
                word not in self.stop_words and
                not re.match(r'^[\W\d]+$', word)):
                filtered_words.append(word)
        return filtered_words

    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """获取BERT嵌入"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding[0]

    def _compute_prompt_score(self, text: str, candidate: str) -> float:
        """计算候选词的Prompt引导分数

        通过将prompt+文本与候选词的语义相似度作为评分
        """
        # 构造prompt输入
        prompt_text = self.prompt_template.format(text=text[:256])

        # 获取prompt嵌入
        prompt_embedding = self._get_bert_embedding(prompt_text)

        # 获取候选词嵌入
        candidate_embedding = self._get_bert_embedding(candidate)

        # 计算余弦相似度作为分数
        sim = cosine_similarity(
            prompt_embedding.reshape(1, -1),
            candidate_embedding.reshape(1, -1)
        )[0][0]

        return float(sim)

    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用Prompt-LLM提取关键词"""
        words = self._preprocess_text(text)

        if not words:
            return []

        # 提取候选词（去重）
        candidates = list(set(words))

        # 计算每个候选词的prompt引导分数
        candidate_scores = []
        for candidate in candidates:
            try:
                score = self._compute_prompt_score(text, candidate)
                candidate_scores.append((candidate, score))
            except Exception:
                continue

        if not candidate_scores:
            return []

        # 按分数排序
        sorted_keywords = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_k]


class OnlyKGMethod(BaselineMethod):
    """Only-KG方法 - 仅KG中心性排序（消融实验用）

    使用jieba分词 + 词频作为简单中心性分数，
    模拟仅使用知识图谱结构信号的场景。
    """

    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.name = "Only-KG"

        onlykg_config = self.config.get('baseline', {}).get('only_kg', {})
        self.stop_words = self._load_stop_words()

        logger.info("Only-KG方法初始化完成（消融实验用）")

    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        stop_words = set()
        try:
            with open('data/stop_words.txt', 'r', encoding='utf-8') as f:
                stop_words = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        return stop_words

    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        words = jieba.lcut(text)
        filtered_words = []
        for word in words:
            if (len(word) > 1 and
                word not in self.stop_words and
                not re.match(r'^[\W\d]+$', word)):
                filtered_words.append(word)
        return filtered_words

    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用词频作为中心性分数提取关键词

        模拟知识图谱中仅基于结构中心性（度中心性）的排序。
        在KG中，节点的度中心性近似于词在共现网络中的词频。
        """
        words = self._preprocess_text(text)

        if not words:
            return []

        # 计算词频作为中心性分数
        word_freq = Counter(words)
        total_words = len(words)

        # 中心性分数 = 归一化词频
        centrality_scores = {
            word: freq / total_words
            for word, freq in word_freq.items()
        }

        # 排序并返回top_k
        sorted_keywords = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_k]


class StaticKGMethod(BaselineMethod):
    """Static-KG方法 - 静态KG排序（消融实验用）

    类似Only-KG但使用TF-IDF分数作为静态权重，
    无时间衰减，模拟静态知识图谱场景。
    """

    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.name = "Static-KG"

        statickg_config = self.config.get('baseline', {}).get('static_kg', {})
        self.stop_words = self._load_stop_words()

        logger.info("Static-KG方法初始化完成（消融实验用）")

    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        stop_words = set()
        try:
            with open('data/stop_words.txt', 'r', encoding='utf-8') as f:
                stop_words = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        return stop_words

    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        words = jieba.lcut(text)
        filtered_words = []
        for word in words:
            if (len(word) > 1 and
                word not in self.stop_words and
                not re.match(r'^[\W\d]+$', word)):
                filtered_words.append(word)
        return filtered_words

    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用TF-IDF作为静态权重提取关键词

        模拟静态知识图谱中节点权重不随时间变化的场景。
        与动态KG不同，这里不考虑时间衰减，使用纯TF-IDF分数。
        """
        words = self._preprocess_text(text)

        if not words:
            return []

        # 计算词频
        word_freq = Counter(words)
        total_words = len(words)

        # 计算TF
        tf_scores = {word: freq / total_words for word, freq in word_freq.items()}

        # 计算IDF（简化版，基于文档内词频）
        unique_words = len(word_freq)
        idf_scores = {word: math.log(unique_words / freq) for word, freq in word_freq.items()}

        # 计算TF-IDF作为静态权重（无时间衰减）
        static_scores = {
            word: tf_scores[word] * idf_scores[word]
            for word in tf_scores
        }

        # 排序并返回top_k
        sorted_keywords = sorted(static_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_k]


class BaselineEvaluator:
    """基线方法评估器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.methods = self._initialize_methods()
        
        logger.info(f"基线评估器初始化完成，包含 {len(self.methods)} 个方法")
    
    def _initialize_methods(self) -> Dict[str, BaselineMethod]:
        """初始化所有基线方法"""
        methods = {}
        baseline_config = self.config.get('baseline', {})

        if baseline_config.get('use_tfidf', True): methods['TF-IDF'] = TFIDFMethod(self.config)
        if baseline_config.get('use_textrank', True): methods['TextRank'] = TextRankMethod(self.config)
        if baseline_config.get('use_positionrank', True): methods['PositionRank'] = PositionRankMethod(self.config)
        if baseline_config.get('use_rake', True): methods['RAKE'] = RAKEMethod(self.config)
        if baseline_config.get('use_yake', True): methods['YAKE'] = YAKEMethod(self.config)
        if baseline_config.get('use_keybert', True): methods['KeyBERT'] = KeyBERTMethod(self.config)
        if baseline_config.get('use_sifrank', True): methods['SIFRank'] = SIFRankMethod(self.config)
        if baseline_config.get('use_lda', True): methods['LDA'] = LDAMethod(self.config)
        if baseline_config.get('use_bertopic', True): methods['BERTopic'] = BERTopicMethod(self.config)
        if baseline_config.get('use_dtm', True): methods['DTM'] = DTMMethod(self.config)
        if baseline_config.get('use_prompt_llm', True): methods['Prompt-LLM'] = PromptLLMMethod(self.config)
        if baseline_config.get('use_only_kg', True): methods['Only-KG'] = OnlyKGMethod(self.config)
        if baseline_config.get('use_static_kg', True): methods['Static-KG'] = StaticKGMethod(self.config)

        return methods
    
    def evaluate_all_methods(self, texts: List[str], 
                           ground_truths: List[List[str]],
                           top_k: int = 10) -> Dict[str, Dict[str, float]]:
        """评估所有基线方法"""
        results = {}
        
        for method_name, method in self.methods.items():
            logger.info(f"评估方法: {method_name}")
            
            # 如果方法需要训练，先训练
            if hasattr(method, 'fit_corpus'):
                method.fit_corpus(texts)
            
            # 提取关键词
            all_predictions = []
            for text in texts:
                keywords = method.extract_keywords(text, top_k)
                predictions = [word for word, _ in keywords]
                all_predictions.append(predictions)
            
            # 计算评估指标
            from .evaluation_metrics import EvaluationMetrics
            evaluator = EvaluationMetrics(self.config)
            
            method_results = evaluator.evaluate_batch_results(
                all_predictions, ground_truths
            )
            
            results[method_name] = method_results
            
            logger.info(f"{method_name} 评估完成")
        
        return results
    
    def get_method(self, method_name: str) -> Optional[BaselineMethod]:
        """获取指定的基线方法"""
        return self.methods.get(method_name)
    
    def add_custom_method(self, method_name: str, method: BaselineMethod):
        """添加自定义基线方法"""
        self.methods[method_name] = method
        logger.info(f"添加自定义方法: {method_name}")