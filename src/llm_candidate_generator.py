# -*- coding: utf-8 -*-
"""
LLM候选生成模块
实现基于BERT的实体抽取和关键词候选生成

增强功能：
- Constrained JSON Schema Prompting
- Alias Resolution (基于embedding相似度的实体聚类)
- Multi-modal Fusion Interface (预留OCR/ASR输入)
- Confidence Calibration (Platt Scaling)
- Post-processing Pipeline (实体规范化、低置信度过滤)
"""

import re
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Set, Optional, Any, Union
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
from loguru import logger
import jieba
import jieba.posseg as pseg

class LLMCandidateGenerator:
    """LLM候选生成器
    
    增强版：支持constrained JSON schema prompting、alias resolution、
    multi-modal fusion、confidence calibration和post-processing pipeline。
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = config['model']['llm']['device']
        
        # 初始化模型
        self.tokenizer = None
        self.model = None
        self.ner_model = None
        self.ner_tokenizer = None
        
        # TF-IDF向量化器
        self.tfidf_vectorizer = None
        
        # 停用词
        self.stopwords = self._load_stopwords()
        
        # Alias resolution 相关
        self.alias_cluster_cache: Dict[str, str] = {}  # alias -> canonical name
        self.entity_embedding_cache: Dict[str, np.ndarray] = {}
        self.alias_similarity_threshold = config.get('alias_similarity_threshold', 0.85)
        
        # Confidence calibration 相关 (Platt Scaling)
        self.platt_scaler = StandardScaler()
        self.platt_model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.platt_fitted = False
        self.calibration_samples: List[Tuple[float, int]] = []  # (raw_score, label)
        
        # Multi-modal fusion 相关
        self.ocr_enabled = False
        self.asr_enabled = False
        self.modal_weights = {
            'text': 1.0,
            'ocr': 0.8,
            'asr': 0.6
        }
        
        # Constrained JSON schema
        self.json_schema = self._build_default_json_schema()
        
        # 初始化模型
        self._init_models()
        
        logger.info("LLM候选生成器初始化完成（增强版）")
    
    def _build_default_json_schema(self) -> Dict:
        """构建默认的constrained JSON schema
        
        即使使用BERT而非LLM，也模拟prompt结构以保持接口一致性。
        该schema定义了候选实体输出的标准格式。
        """
        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "minLength": 2},
                            "type": {
                                "type": "string",
                                "enum": ["person", "organization", "location",
                                         "event", "product", "concept", "other"]
                            },
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "aliases": {"type": "array", "items": {"type": "string"}},
                            "source_span": {"type": "string"}
                        },
                        "required": ["name", "type", "confidence"]
                    }
                },
                "keywords": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "minLength": 2},
                            "relevance_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "source": {
                                "type": "string",
                                "enum": ["ner", "bert", "tfidf", "phrase", "context", "ocr", "asr"]
                            }
                        },
                        "required": ["text", "relevance_score", "source"]
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "extraction_method": {"type": "string"},
                        "timestamp": {"type": "string"},
                        "model_version": {"type": "string"}
                    }
                }
            },
            "required": ["entities", "keywords"]
        }
        return schema
    
    def generate_with_json_schema(self, text: str, top_k: int = 30) -> Dict:
        """Constrained JSON Schema Prompting
        
        模拟LLM的structured output格式，使用BERT提取结果后组装为符合
        JSON schema的标准化输出。保证输出格式的一致性和可解析性。
        
        Args:
            text: 输入文本
            top_k: 返回的候选数量上限
            
        Returns:
            符合self.json_schema定义的字典
        """
        # 使用现有方法提取实体和关键词
        entities_raw = self.extract_entities_with_ner(text)
        keywords_raw = self.generate_candidates_with_context(text, top_k=top_k)
        
        # 映射词性到schema中的实体类型
        pos_to_type = {
            'nr': 'person', 'ns': 'location', 'nt': 'organization',
            'nz': 'other', 'n': 'concept', 'v': 'event',
            'vn': 'event', 'a': 'concept', 'ad': 'concept'
        }
        
        # 构建entities数组
        entities = []
        for name, pos_tag, confidence in entities_raw[:top_k]:
            entity_entry = {
                "name": name,
                "type": pos_to_type.get(pos_tag, 'other'),
                "confidence": round(float(confidence), 4),
                "aliases": self._resolve_aliases(name),
                "source_span": name
            }
            entities.append(entity_entry)
        
        # 构建keywords数组
        keywords = []
        for text_kw, score, source in keywords_raw[:top_k]:
            keyword_entry = {
                "text": text_kw,
                "relevance_score": round(float(score), 4),
                "source": source
            }
            keywords.append(keyword_entry)
        
        # 构建metadata
        metadata = {
            "extraction_method": "bert_constrained_json",
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "model_version": self.config['model']['llm'].get('model_name', 'unknown')
        }
        
        result = {
            "entities": entities,
            "keywords": keywords,
            "metadata": metadata
        }
        
        # 验证输出是否符合schema
        self._validate_json_schema(result)
        
        return result
    
    def _validate_json_schema(self, data: Dict) -> bool:
        """验证输出数据是否符合JSON schema
        
        Args:
            data: 待验证的字典数据
            
        Returns:
            是否通过验证
        """
        try:
            # 基础结构验证
            if not isinstance(data, dict):
                logger.warning("JSON schema验证失败: 顶层非dict")
                return False
            
            if 'entities' not in data or 'keywords' not in data:
                logger.warning("JSON schema验证失败: 缺少必需字段")
                return False
            
            if not isinstance(data['entities'], list) or not isinstance(data['keywords'], list):
                logger.warning("JSON schema验证失败: entities/keywords非list")
                return False
            
            # 验证每个entity
            for entity in data['entities']:
                if not all(k in entity for k in ['name', 'type', 'confidence']):
                    logger.warning(f"JSON schema验证失败: entity缺少必需字段 {entity}")
                    return False
                if not (0.0 <= entity['confidence'] <= 1.0):
                    logger.warning(f"JSON schema验证失败: confidence超出范围 {entity['confidence']}")
                    return False
            
            # 验证每个keyword
            for kw in data['keywords']:
                if not all(k in kw for k in ['text', 'relevance_score', 'source']):
                    logger.warning(f"JSON schema验证失败: keyword缺少必需字段 {kw}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"JSON schema验证异常: {e}")
            return False
    
    def resolve_aliases(self, entities: List[str]) -> Dict[str, str]:
        """Alias Resolution: 基于embedding相似度的实体聚类
        
        对输入实体列表进行聚类，将语义相似的实体归并为同一规范名称。
        使用BERT embedding计算余弦相似度，超过阈值的实体视为别名。
        
        Args:
            entities: 实体名称列表
            
        Returns:
            {alias_name: canonical_name} 映射字典
        """
        if not entities:
            return {}
        
        # 获取所有实体的embedding
        embeddings = {}
        for entity in entities:
            if entity in self.entity_embedding_cache:
                embeddings[entity] = self.entity_embedding_cache[entity]
            else:
                emb = self._get_text_embedding(entity)
                self.entity_embedding_cache[entity] = emb
                embeddings[entity] = emb
        
        # 构建相似度矩阵
        entity_list = list(embeddings.keys())
        n = len(entity_list)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_similarity(
                    embeddings[entity_list[i]].reshape(1, -1),
                    embeddings[entity_list[j]].reshape(1, -1)
                )[0][0]
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim
        
        # Union-Find聚类
        parent = list(range(n))
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                # 选择长度更长或频率更高的作为canonical
                if len(entity_list[px]) >= len(entity_list[py]):
                    parent[py] = px
                else:
                    parent[px] = py
        
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] >= self.alias_similarity_threshold:
                    union(i, j)
        
        # 构建alias -> canonical映射
        clusters = defaultdict(list)
        for i in range(n):
            root = find(i)
            clusters[root].append(entity_list[i])
        
        alias_map = {}
        for root, members in clusters.items():
            # 选择最长的作为canonical name
            canonical = max(members, key=lambda x: (len(x), members.count(x)))
            for member in members:
                alias_map[member] = canonical
                self.alias_cluster_cache[member] = canonical
        
        logger.info(f"Alias resolution完成: {len(entities)}个实体 -> {len(clusters)}个规范名称")
        return alias_map
    
    def _resolve_aliases(self, entity: str) -> List[str]:
        """查询单个实体的已知别名"""
        aliases = []
        for alias, canonical in self.alias_cluster_cache.items():
            if canonical == entity and alias != entity:
                aliases.append(alias)
        return aliases
    
    def fuse_multi_modal(self,
                         text: str,
                         ocr_text: Optional[str] = None,
                         asr_text: Optional[str] = None,
                         top_k: int = 30) -> List[Tuple[str, float, str]]:
        """Multi-modal Fusion Interface
        
        预留OCR和ASR输入接口，支持文本、OCR、ASR三模态融合。
        各模态独立提取候选，通过加权融合得到最终候选列表。
        
        Args:
            text: 主文本输入
            ocr_text: OCR识别文本（可选，如图片中的文字）
            asr_text: ASR转写文本（可选，如视频/音频转写）
            top_k: 返回候选数量上限
            
        Returns:
            融合后的候选列表 [(candidate, score, source), ...]
        """
        all_candidates: Dict[str, Tuple[float, str]] = {}
        
        # 1. 文本模态（主模态）
        text_candidates = self.generate_candidates_with_context(text, top_k=top_k * 2)
        for candidate, score, source in text_candidates:
            weighted_score = score * self.modal_weights['text']
            if candidate not in all_candidates or weighted_score > all_candidates[candidate][0]:
                all_candidates[candidate] = (weighted_score, source)
        
        # 2. OCR模态
        if ocr_text and self.ocr_enabled:
            ocr_candidates = self.generate_candidates_with_context(ocr_text, top_k=top_k)
            for candidate, score, source in ocr_candidates:
                weighted_score = score * self.modal_weights['ocr']
                if candidate not in all_candidates:
                    all_candidates[candidate] = (weighted_score, 'ocr')
                else:
                    # 取加权最大值
                    existing_score = all_candidates[candidate][0]
                    if weighted_score > existing_score:
                        all_candidates[candidate] = (weighted_score, 'ocr_fused')
        
        # 3. ASR模态
        if asr_text and self.asr_enabled:
            asr_candidates = self.generate_candidates_with_context(asr_text, top_k=top_k)
            for candidate, score, source in asr_candidates:
                weighted_score = score * self.modal_weights['asr']
                if candidate not in all_candidates:
                    all_candidates[candidate] = (weighted_score, 'asr')
                else:
                    existing_score = all_candidates[candidate][0]
                    if weighted_score > existing_score:
                        all_candidates[candidate] = (weighted_score, 'asr_fused')
        
        # 排序并返回
        result = [(k, v[0], v[1]) for k, v in all_candidates.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result[:top_k]
    
    def enable_ocr_mode(self, weight: float = 0.8):
        """启用OCR模态"""
        self.ocr_enabled = True
        self.modal_weights['ocr'] = weight
        logger.info(f"OCR模态已启用，权重: {weight}")
    
    def enable_asr_mode(self, weight: float = 0.6):
        """启用ASR模态"""
        self.asr_enabled = True
        self.modal_weights['asr'] = weight
        logger.info(f"ASR模态已启用，权重: {weight}")
    
    def calibrate_confidence(self, raw_scores: np.ndarray, labels: np.ndarray):
        """Confidence Calibration: Platt Scaling
        
        使用Platt Scaling（逻辑回归）将原始分数校准为真实概率。
        需要在有标签数据上拟合。
        
        Platt Scaling公式: P(y=1|s) = 1 / (1 + exp(A * s + B))
        其中A和B通过在验证集上最大化似然得到。
        
        Args:
            raw_scores: 原始模型输出分数 (N,)
            labels: 二值标签 (N,)，1表示正例，0表示负例
        """
        if len(raw_scores) < 10:
            logger.warning("校准样本不足（<10），跳过Platt Scaling拟合")
            return
        
        # 标准化输入分数
        scaled_scores = self.platt_scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()
        
        # 拟合逻辑回归
        self.platt_model.fit(scaled_scores.reshape(-1, 1), labels)
        self.platt_fitted = True
        
        # 存储校准样本
        for score, label in zip(raw_scores, labels):
            self.calibration_samples.append((float(score), int(label)))
        
        train_acc = self.platt_model.score(scaled_scores.reshape(-1, 1), labels)
        logger.info(f"Platt Scaling校准完成，训练准确率: {train_acc:.4f}，样本数: {len(raw_scores)}")
    
    def predict_calibrated(self, raw_score: float) -> float:
        """使用Platt Scaling校准单个分数
        
        Args:
            raw_score: 原始分数
            
        Returns:
            校准后的概率值 [0, 1]
        """
        if not self.platt_fitted:
            logger.debug("Platt Scaling未拟合，返回原始分数")
            return float(np.clip(raw_score, 0.0, 1.0))
        
        scaled_score = self.platt_scaler.transform([[raw_score]])[0, 0]
        calibrated_prob = self.platt_model.predict_proba([[scaled_score]])[0, 1]
        
        return float(np.clip(calibrated_prob, 0.0, 1.0))
    
    def batch_calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        """批量校准分数
        
        Args:
            raw_scores: 原始分数数组 (N,)
            
        Returns:
            校准后的概率数组 (N,)
        """
        if not self.platt_fitted:
            return np.clip(raw_scores, 0.0, 1.0)
        
        scaled_scores = self.platt_scaler.transform(raw_scores.reshape(-1, 1))
        calibrated_probs = self.platt_model.predict_proba(scaled_scores)[:, 1]
        
        return np.clip(calibrated_probs, 0.0, 1.0)
    
    def post_process(self,
                     candidates: List[Tuple[str, float, str]],
                     apply_calibration: bool = True,
                     confidence_threshold: float = 0.1,
                     normalize_entities: bool = True,
                     max_candidates: int = 50) -> List[Tuple[str, float, str]]:
        """Post-processing Pipeline
        
        完整的后处理流水线：
        1. 实体规范化（别名归并）
        2. 置信度校准（Platt Scaling）
        3. 低置信度过滤
        4. 候选去重与排序
        
        Args:
            candidates: 原始候选列表 [(candidate, score, source), ...]
            apply_calibration: 是否应用Platt Scaling校准
            confidence_threshold: 置信度过滤阈值
            normalize_entities: 是否进行实体规范化
            max_candidates: 最大返回候选数
            
        Returns:
            后处理后的候选列表
        """
        if not candidates:
            return []
        
        processed = []
        
        # Step 1: 实体规范化（alias resolution）
        if normalize_entities and len(candidates) > 1:
            entity_names = [c[0] for c in candidates]
            alias_map = self.resolve_aliases(entity_names)
            
            # 归并别名：同一规范名称的候选取最高分
            canonical_scores: Dict[str, Tuple[float, str]] = {}
            for candidate, score, source in candidates:
                canonical = alias_map.get(candidate, candidate)
                if canonical not in canonical_scores or score > canonical_scores[canonical][0]:
                    canonical_scores[canonical] = (score, source)
            
            candidates = [(k, v[0], v[1]) for k, v in canonical_scores.items()]
        
        # Step 2: 置信度校准
        if apply_calibration and self.platt_fitted:
            raw_scores = np.array([c[1] for c in candidates])
            calibrated_scores = self.batch_calibrate(raw_scores)
            candidates = [(c[0], float(calibrated_scores[i]), c[2])
                          for i, c in enumerate(candidates)]
        
        # Step 3: 低置信度过滤
        filtered = [(c, s, src) for c, s, src in candidates if s >= confidence_threshold]
        
        # Step 4: 质量过滤
        filtered = self.filter_candidates_by_quality(filtered, min_score=0.0)
        
        # Step 5: 排序
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        result = filtered[:max_candidates]
        
        logger.info(f"Post-processing: {len(candidates)} -> {len(result)} 候选"
                     f" (阈值={confidence_threshold}, 校准={apply_calibration})")
        
        return result
    
    def _load_stopwords(self) -> Set[str]:
        """加载停用词"""
        # 中文停用词列表
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '们', '这个', '那个', '什么', '怎么', '为什么', '哪里', '哪个', '多少', '几个', '一些', '所有', '每个', '任何', '没', '还', '再', '又', '已经', '正在', '将要', '可以', '应该', '必须', '能够', '可能', '或者', '但是', '然后', '因为', '所以', '如果', '虽然', '除了', '包括', '关于', '对于', '根据', '按照', '通过', '由于', '为了', '以便', '以及', '以上', '以下', '之前', '之后', '之间', '之外', '其中', '其他', '另外', '此外', '而且', '并且', '或者', '不过', '不仅', '不但', '不只', '不光'
        }
        return stopwords
    
    def _init_models(self):
        """初始化模型"""
        model_name = self.config['model']['llm']['model_name']
        
        # 加载BERT模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # 设置设备
        if self.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # 初始化TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words=list(self.stopwords)
        )
        
        logger.info(f"已加载模型: {model_name}")
    
    def extract_entities_with_ner(self, text: str) -> List[Tuple[str, str, float]]:
        """使用NER模型提取实体"""
        # 简化版NER，使用词性标注
        entities = []
        words = pseg.cut(text)
        
        for word, flag in words:
            # 过滤停用词和标点
            if word in self.stopwords or len(word) < 2:
                continue
                
            # 根据词性判断是否为实体
            if flag in ['n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn', 'a', 'ad']:
                # 计算实体置信度（简化版）
                confidence = self._calculate_entity_confidence(word, flag)
                entities.append((word, flag, confidence))
        
        # 按置信度排序
        entities.sort(key=lambda x: x[2], reverse=True)
        return entities
    
    def _calculate_entity_confidence(self, word: str, pos_tag: str) -> float:
        """计算实体置信度"""
        base_score = 0.5
        
        # 根据词性调整分数
        pos_weights = {
            'nr': 0.9,  # 人名
            'ns': 0.9,  # 地名
            'nt': 0.8,  # 机构名
            'nz': 0.7,  # 其他专名
            'n': 0.6,   # 名词
            'v': 0.4,   # 动词
            'vn': 0.5,  # 名动词
            'a': 0.3,   # 形容词
            'ad': 0.3   # 副形词
        }
        
        pos_score = pos_weights.get(pos_tag, 0.3)
        
        # 根据长度调整分数
        length_score = min(len(word) / 4.0, 1.0)
        
        # 综合分数
        confidence = base_score * pos_score * length_score
        return min(confidence, 1.0)
    
    def extract_keywords_with_bert(self, text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """使用BERT提取关键词"""
        # 分词
        words = list(jieba.cut(text))
        words = [w for w in words if w not in self.stopwords and len(w) > 1]
        
        if not words:
            return []
        
        # 获取文本嵌入
        text_embedding = self._get_text_embedding(text)
        
        # 计算每个词的重要性分数
        word_scores = []
        for word in set(words):  # 去重
            # 获取词嵌入
            word_embedding = self._get_text_embedding(word)
            
            # 计算与文本的相似度
            similarity = cosine_similarity(
                word_embedding.reshape(1, -1),
                text_embedding.reshape(1, -1)
            )[0][0]
            
            # 计算词频权重
            freq_weight = words.count(word) / len(words)
            
            # 综合分数
            score = 0.7 * similarity + 0.3 * freq_weight
            word_scores.append((word, score))
        
        # 排序并返回top-k
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return word_scores[:top_k]
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入"""
        inputs = self.tokenizer(text, return_tensors='pt', 
                               max_length=512, truncation=True, padding=True)
        
        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]标记的嵌入
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.flatten()
    
    def extract_phrases_with_ngrams(self, text: str, max_ngram: int = 3) -> List[Tuple[str, float]]:
        """提取n-gram短语"""
        words = list(jieba.cut(text))
        words = [w for w in words if w not in self.stopwords and len(w) > 1]
        
        phrases = []
        
        # 生成n-gram
        for n in range(1, max_ngram + 1):
            for i in range(len(words) - n + 1):
                phrase = ''.join(words[i:i+n])
                if len(phrase) >= 2:  # 最小长度限制
                    phrases.append(phrase)
        
        # 计算短语频率
        phrase_counts = Counter(phrases)
        
        # 计算TF-IDF分数（简化版）
        phrase_scores = []
        total_phrases = len(phrases)
        
        for phrase, count in phrase_counts.items():
            tf = count / total_phrases
            # 简化的IDF计算
            idf = np.log(total_phrases / count)
            score = tf * idf
            phrase_scores.append((phrase, score))
        
        # 排序
        phrase_scores.sort(key=lambda x: x[1], reverse=True)
        return phrase_scores
    
    def generate_candidates_with_context(self, text: str, 
                                       context_texts: List[str] = None,
                                       top_k: int = 30) -> List[Tuple[str, float, str]]:
        """基于上下文生成候选关键词"""
        candidates = []
        
        # 1. 实体提取
        entities = self.extract_entities_with_ner(text)
        for entity, pos_tag, confidence in entities[:15]:
            candidates.append((entity, confidence, 'entity'))
        
        # 2. BERT关键词提取
        bert_keywords = self.extract_keywords_with_bert(text, top_k=15)
        for keyword, score in bert_keywords:
            candidates.append((keyword, score, 'bert_keyword'))
        
        # 3. 短语提取
        phrases = self.extract_phrases_with_ngrams(text, max_ngram=3)
        for phrase, score in phrases[:10]:
            candidates.append((phrase, score, 'phrase'))
        
        # 4. 如果有上下文，计算上下文相关性
        if context_texts:
            context_candidates = self._extract_context_keywords(text, context_texts)
            candidates.extend(context_candidates)
        
        # 去重并排序
        unique_candidates = {}
        for candidate, score, source in candidates:
            if candidate not in unique_candidates:
                unique_candidates[candidate] = (score, source)
            else:
                # 如果已存在，取最高分数
                if score > unique_candidates[candidate][0]:
                    unique_candidates[candidate] = (score, source)
        
        # 转换为列表并排序
        final_candidates = [(k, v[0], v[1]) for k, v in unique_candidates.items()]
        final_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return final_candidates[:top_k]
    
    def _extract_context_keywords(self, text: str, 
                                 context_texts: List[str]) -> List[Tuple[str, float, str]]:
        """基于上下文提取关键词"""
        if not context_texts:
            return []
        
        # 合并上下文文本
        context_text = ' '.join(context_texts)
        
        # 提取上下文关键词
        context_keywords = self.extract_keywords_with_bert(context_text, top_k=20)
        
        # 计算与当前文本的相关性
        text_embedding = self._get_text_embedding(text)
        
        context_candidates = []
        for keyword, base_score in context_keywords:
            # 检查关键词是否在当前文本中
            if keyword in text:
                # 计算语义相似度
                keyword_embedding = self._get_text_embedding(keyword)
                similarity = cosine_similarity(
                    keyword_embedding.reshape(1, -1),
                    text_embedding.reshape(1, -1)
                )[0][0]
                
                # 综合分数
                context_score = 0.6 * base_score + 0.4 * similarity
                context_candidates.append((keyword, context_score, 'context'))
        
        return context_candidates
    
    def filter_candidates_by_quality(self, candidates: List[Tuple[str, float, str]],
                                   min_score: float = 0.1,
                                   max_length: int = 10) -> List[Tuple[str, float, str]]:
        """根据质量过滤候选词"""
        filtered = []
        
        for candidate, score, source in candidates:
            # 分数过滤
            if score < min_score:
                continue
            
            # 长度过滤
            if len(candidate) > max_length:
                continue
            
            # 内容过滤
            if self._is_valid_candidate(candidate):
                filtered.append((candidate, score, source))
        
        return filtered
    
    def _is_valid_candidate(self, candidate: str) -> bool:
        """检查候选词是否有效"""
        # 过滤纯数字
        if candidate.isdigit():
            return False
        
        # 过滤纯标点
        if re.match(r'^[^\w\u4e00-\u9fa5]+$', candidate):
            return False
        
        # 过滤停用词
        if candidate in self.stopwords:
            return False
        
        # 过滤过短的词
        if len(candidate) < 2:
            return False
        
        return True
    
    def generate_diverse_candidates(self, text: str, 
                                  context_texts: List[str] = None,
                                  diversity_threshold: float = 0.7,
                                  top_k: int = 20) -> List[Tuple[str, float, str]]:
        """生成多样化的候选关键词"""
        # 生成初始候选词
        all_candidates = self.generate_candidates_with_context(
            text, context_texts, top_k=50
        )
        
        # 质量过滤
        filtered_candidates = self.filter_candidates_by_quality(all_candidates)
        
        if not filtered_candidates:
            return []
        
        # 多样性选择
        diverse_candidates = [filtered_candidates[0]]  # 选择最高分的
        
        for candidate, score, source in filtered_candidates[1:]:
            # 检查与已选择候选词的相似度
            is_diverse = True
            candidate_embedding = self._get_text_embedding(candidate)
            
            for selected_candidate, _, _ in diverse_candidates:
                selected_embedding = self._get_text_embedding(selected_candidate)
                similarity = cosine_similarity(
                    candidate_embedding.reshape(1, -1),
                    selected_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity > diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_candidates.append((candidate, score, source))
                
                if len(diverse_candidates) >= top_k:
                    break
        
        return diverse_candidates
    
    def batch_generate_candidates(self, texts: List[str],
                                context_texts_list: List[List[str]] = None,
                                top_k: int = 20) -> List[List[Tuple[str, float, str]]]:
        """批量生成候选关键词"""
        results = []
        
        for i, text in enumerate(texts):
            context_texts = context_texts_list[i] if context_texts_list else None
            candidates = self.generate_diverse_candidates(
                text, context_texts, top_k=top_k
            )
            results.append(candidates)
        
        logger.info(f"批量生成候选关键词完成，处理文本数量: {len(texts)}")
        return results
    
    def get_candidate_statistics(self, candidates: List[Tuple[str, float, str]]) -> Dict:
        """获取候选词统计信息"""
        if not candidates:
            return {}
        
        scores = [score for _, score, _ in candidates]
        sources = [source for _, _, source in candidates]
        lengths = [len(candidate) for candidate, _, _ in candidates]
        
        stats = {
            'total_candidates': len(candidates),
            'avg_score': np.mean(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'avg_length': np.mean(lengths),
            'source_distribution': dict(Counter(sources))
        }
        
        return stats