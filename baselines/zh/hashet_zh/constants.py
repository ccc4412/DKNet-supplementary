# -*- coding: utf-8 -*-
"""HASHET-zh 配置。官方 constants.py 的中文适配版:
- SENTENCE_ENCODING_MODEL 固定 BERT 分支（GUSE 为英文 TF-Hub 模型, 弃用, 偏离 D5）
- 编码器权重名 bert-base-uncased -> bert-base-chinese（见 model_bert_zh.py）
- PERC_TEST 保留官方值, 但划分方式改为按时间先后（偏离 D2, 论文中声明）
- MINCOUNT / W2V_MINCOUNT / LATENT_SPACE_DIM 等官方默认值原样保留
"""
import os

SEED = 42
LATENT_SPACE_DIM = 150      # 官方原值
WINDOW_SIZE = 5             # 官方原值
W2V_MINCOUNT = 5            # 官方默认（大语料 README 建议 5, 保持不变）
MINCOUNT = 10               # hashtag 入表阈值, 官方默认
BATCH_SIZE = 32             # 官方默认
PATIENCE = 3                # 官方默认
MAX_EPOCHS = 50             # 官方默认
LR_TRANSFER = 1e-3          # Keras Adam 默认学习率（官方 compile 默认值）
LR_FINETUNE = 3e-5          # 官方 model_BERT.py 微调学习率原值

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(ROOT, "save_files")

W2V_INPUT = os.path.join(SAVE_FOLDER, "sentences.txt")
W2V_SAVE_FILE_NAME = os.path.join(SAVE_FOLDER, "w2v_emb.model")
TRAIN_TEST_INPUT = os.path.join(SAVE_FOLDER, "model_sentences.txt")
TRAIN_CORPUS = os.path.join(SAVE_FOLDER, "train_corpus.txt")
TEST_CORPUS = os.path.join(SAVE_FOLDER, "test_corpus.txt")
TL_WEIGHTS = os.path.join(SAVE_FOLDER, "BERT_TLweights.pt")
FT_WEIGHTS = os.path.join(SAVE_FOLDER, "BERT_FTweights.pt")
RESULTS_TXT = os.path.join(SAVE_FOLDER, "results.txt")
METRICS_JSON = os.path.join(SAVE_FOLDER, "metrics.json")

SKIP_HASHTAG_REMOVING = True   # 官方默认
EXPANSION_STRATEGY = "global"  # 官方默认（论文报告 global 最优）
MAX_EXPANSION_ITERATIONS = 6   # 官方默认
PERC_TEST = 0.25               # 官方默认

# H-ADAPTS 复用本模型类时的渐进微调参数（官方 constants 原值）
MAX_EPOCHS_FT = 12
RT_LR = 3e-5

# 编码骨干选择: 'mock' 冒烟测试 / 'bert' = bert-base-chinese（正式实验）
ENCODER_BACKEND = os.environ.get("ZH_ENCODER", "mock")
