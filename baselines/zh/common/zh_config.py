# -*- coding: utf-8 -*-
"""公共配置：两个 baseline 共享的潜空间维度、随机种子与编码器选择。

官方 constants.py 中语言无关的超参数（LATENT_SPACE_DIM=150, WINDOW_SIZE=5）
原值保留；每个 baseline 特有的参数在其各自 constants 中。
"""
import os

import numpy as np

SEED = int(os.environ.get("ZH_SEED", "42"))
LATENT_SPACE_DIM = 150   # 官方原值
WINDOW_SIZE = 5          # 官方原值


def seed_all(seed=None):
    """统一随机种子（random/numpy/torch/gensim 经构造参数传入）。"""
    import random
    seed = seed if seed is not None else SEED
    globals()["SEED"] = seed  # 让 zh_w2v 等后读取方拿到同一种子
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# 编码器后端: 'mock' 仅用于流水线冒烟测试（确定性哈希向量），
# 'bert' = bert-base-chinese（论文对比使用的正式配置）
ENCODER = os.environ.get("ZH_ENCODER", "mock")
BERT_MODEL_NAME = "bert-base-chinese"
ENCODER_DIM = 768
