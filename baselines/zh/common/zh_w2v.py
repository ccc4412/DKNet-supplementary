# -*- coding: utf-8 -*-
"""Word2Vec hashtag 潜在空间：官方 word_embedding_model.py 的等价移植。

gensim 4.x 中 `size=` 改名 `vector_size=`，其余训练参数逐项保持官方默认值:
min_count=mincount, window=5, sample=1e-5, alpha=0.01, min_alpha=0.001,
negative=10, epochs=50。retain_hts / tweet_arith_embedding 逻辑不变。
"""


import numpy as np
from gensim.models import Word2Vec

from common import zh_config as cfg


def create_and_train_word2vec(sentences, mincount=5, save_path=None):
    # workers=1: 规避 Windows 多进程偶发死锁, 且保证可复现（小语料无速度损失）
    w2v = Word2Vec(min_count=mincount, window=cfg.WINDOW_SIZE,
                   vector_size=cfg.LATENT_SPACE_DIM, sample=1e-5, alpha=0.01,
                   min_alpha=0.001, negative=10, workers=1, seed=cfg.SEED)
    w2v.build_vocab(sentences, progress_per=10)
    w2v.train(sentences, total_examples=w2v.corpus_count, epochs=50, report_delay=1)
    if save_path:
        w2v.save(save_path)
    return w2v


def load_word2vec(path):
    return Word2Vec.load(path)


def retain_hts(top_emb):
    """官方原样: 从近邻列表中保留 hashtag（含下划线 bigram 拆分分支）。"""
    out = []
    for tuple_ in top_emb:
        if "#" in tuple_[0]:
            if "_" in tuple_[0]:
                for h in tuple_[0].split("_"):
                    if "#" in h:
                        out.append((h, tuple_[1]))
            else:
                out.append(tuple_)
    return out


def tweet_arith_embedding(w2v, tokens_line):
    """官方原样: hashtag 词向量均值；均不在词表则返回 None。"""
    den = 0
    emb = np.zeros(cfg.LATENT_SPACE_DIM)
    for word in tokens_line.split():
        try:
            emb += w2v.wv[word]
            den += 1
        except KeyError:
            pass
    return None if den == 0 else emb / den
