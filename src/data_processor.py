# -*- coding: utf-8 -*-
"""
数据处理模块
实现文本预处理、数据加载和清洗功能
"""

import re
import json
import pandas as pd
import numpy as np
import jieba
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = None
        self.model = None
        self._init_models()
        
    def _init_models(self):
        """初始化模型"""
        model_name = self.config['model']['llm']['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # 设置设备
        device = self.config['model']['llm']['device']
        if device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        logger.info(f"已加载模型: {model_name}")
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not text:
            return ""
        
        # 清洗文本
        cleaned_text = self.clean_text(text)
        
        # 过滤长度
        min_length = self.config.get('data', {}).get('preprocessing', {}).get('min_text_length', 20)
        max_length = self.config.get('data', {}).get('preprocessing', {}).get('max_text_length', 500)
        
        if len(cleaned_text) < min_length or len(cleaned_text) > max_length:
            return ""
            
        return cleaned_text
    
    def clean_text(self, text: str) -> str:
        """文本清洗"""
        if not text:
            return ""
            
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除表情符号
        text = re.sub(r'\[.*?\]', '', text)
        
        # 移除特殊字符，保留中文、英文、数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def segment_text(self, text: str) -> List[str]:
        """中文分词"""
        return list(jieba.cut(text))
    
    def extract_entities(self, text: str) -> List[str]:
        """实体提取（简化版，使用BERT-NER的占位实现）"""
        # 这里使用简化的实体提取，实际应该使用专门的NER模型
        words = self.segment_text(text)
        
        # 简单的实体识别规则
        entities = []
        for word in words:
            # 长度大于1的词作为潜在实体
            if len(word) > 1 and word not in ['的', '了', '在', '是', '有', '和', '与']:
                entities.append(word)
                
        return entities
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入"""
        inputs = self.tokenizer(text, return_tensors='pt', 
                               max_length=512, truncation=True, padding=True)
        
        if self.config['model']['llm']['device'] == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]标记的嵌入作为文本表示
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        return embedding.flatten()
    
    def calculate_edit_distance(self, text1: str, text2: str) -> float:
        """计算编辑距离"""
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    
        return dp[m][n] / max(m, n)
    
    def remove_duplicates(self, texts: List[str], threshold: float = 0.9) -> List[str]:
        """去重"""
        unique_texts = []
        
        for text in texts:
            is_duplicate = False
            for unique_text in unique_texts:
                if self.calculate_edit_distance(text, unique_text) < threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_texts.append(text)
                
        logger.info(f"去重前: {len(texts)}, 去重后: {len(unique_texts)}")
        return unique_texts
    
    def filter_by_length(self, texts: List[str], 
                        min_length: int = 20, max_length: int = 500) -> List[str]:
        """按长度过滤"""
        filtered = [text for text in texts if min_length <= len(text) <= max_length]
        logger.info(f"长度过滤前: {len(texts)}, 过滤后: {len(filtered)}")
        return filtered
    
    def add_time_decay(self, timestamp: datetime, current_time: datetime, 
                      tau: float = 24) -> float:
        """计算时间衰减因子"""
        delta_hours = (current_time - timestamp).total_seconds() / 3600
        return np.exp(-delta_hours / tau)
    
    def load_weibo_data(self, file_path: str) -> pd.DataFrame:
        """加载微博数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            
            # 转换时间格式
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['crawl_time'] = pd.to_datetime(df['crawl_time'])
            
            # 文本预处理
            df['cleaned_text'] = df['text'].apply(self.clean_text)
            
            # 过滤
            df = df[df['cleaned_text'].str.len() >= self.config['data']['weibo']['min_text_length']]
            df = df[df['cleaned_text'].str.len() <= self.config['data']['weibo']['max_text_length']]
            
            logger.info(f"加载微博数据: {len(df)} 条")
            return df
            
        except Exception as e:
            logger.error(f"加载微博数据失败: {e}")
            return pd.DataFrame()
    
    def load_news_data(self, file_path: str) -> pd.DataFrame:
        """加载新闻数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            
            # 转换时间格式
            df['pub_time'] = pd.to_datetime(df['pub_time'])
            df['crawl_time'] = pd.to_datetime(df['crawl_time'])
            
            # 文本预处理
            df['cleaned_title'] = df['title'].apply(self.clean_text)
            if 'summary' in df.columns:
                df['cleaned_summary'] = df['summary'].apply(self.clean_text)
            
            logger.info(f"加载新闻数据: {len(df)} 条")
            return df
            
        except Exception as e:
            logger.error(f"加载新闻数据失败: {e}")
            return pd.DataFrame()
    
    def split_data_by_time(self, df: pd.DataFrame, time_col: str, 
                          train_ratio: float = 0.7, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按时间划分数据集"""
        df_sorted = df.sort_values(time_col)
        n = len(df_sorted)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]
        
        logger.info(f"数据划分 - 训练: {len(train_df)}, 验证: {len(val_df)}, 测试: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_descriptive_statistics(self, df: pd.DataFrame) -> Dict:
        """获取描述性统计"""
        stats = {}
        
        if 'cleaned_text' in df.columns:
            text_lengths = df['cleaned_text'].str.len()
            stats['text_length'] = {
                'mean': text_lengths.mean(),
                'std': text_lengths.std(),
                'min': text_lengths.min(),
                'max': text_lengths.max(),
                'median': text_lengths.median()
            }
        
        # 实体数统计
        if 'cleaned_text' in df.columns:
            entity_counts = df['cleaned_text'].apply(lambda x: len(self.extract_entities(x)))
            stats['entity_count'] = {
                'mean': entity_counts.mean(),
                'std': entity_counts.std(),
                'min': entity_counts.min(),
                'max': entity_counts.max(),
                'median': entity_counts.median()
            }
        
        return stats
    
    def load_d1_dataset(self, file_path: str) -> pd.DataFrame:
        """加载D1数据集（微博时序数据）"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 确保时间列为datetime类型
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
                
            logger.info(f"成功加载D1数据集，共{len(df)}条记录")
            return df
            
        except Exception as e:
            logger.error(f"加载D1数据集失败: {e}")
            raise
    
    def load_d2_dataset(self, file_path: str) -> pd.DataFrame:
        """加载D2数据集（新闻门户数据）"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 确保时间列为datetime类型
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'publish_time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['publish_time'])
                
            logger.info(f"成功加载D2数据集，共{len(df)}条记录")
            return df
            
        except Exception as e:
            logger.error(f"加载D2数据集失败: {e}")
            raise