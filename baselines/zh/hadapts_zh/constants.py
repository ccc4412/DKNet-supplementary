# -*- coding: utf-8 -*-
"""H-ADAPTS-zh 配置。窗口/阈值均为官方 constants.py 原值, 仅时间单位按 D1
仿真日历换算（官方按天, D1 时间轴即真实秒, 不做缩放）。
"""
import os

SEED = 42
# ---- 官方原值 ----
LOGIC_WINDOW_LENGTH = 14 * 86400   # 14 天（重训 W2V + MLP 的数据窗口）
FT_WINDOW_LENGTH = 4 * 86400       # 4 天（渐进微调数据窗口）
TUMBLING_WINDOW_LENGTH = 1 * 86400 # 1 天（检测/预测的滚动窗口）
K = 10                             # top-K trending
THRESHOLD = 0.9                    # 切换阈值
W2V_MINCOUNT = 40                  # 官方实时分支原值
MINCOUNT = 60                      # 官方实时分支原值
MAX_EPOCHS_FT = 12                 # 官方 rt_training 微调轮数
RT_LR = 3e-5                       # 官方实时微调学习率
PERC_TEST = 0.25                   # 官方原值（窗口内 train/val 划分比例）

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(ROOT, "save_files")
RESULTS_FOLDER = os.path.join(ROOT, "results")
os.makedirs(SAVE_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

W2V_SAVE_FILE_NAME = os.path.join(SAVE_FOLDER, "w2v_emb.model")
FT_WEIGHTS = os.path.join(SAVE_FOLDER, "BERT_FTweights.pt")
METRICS_JSON = os.path.join(RESULTS_FOLDER, "metrics.json")
SHIFTS_JSON = os.path.join(RESULTS_FOLDER, "shifts.json")
PREDICTIONS_JSONL = os.path.join(RESULTS_FOLDER, "predictions.jsonl")

ENCODER_BACKEND = os.environ.get("ZH_ENCODER", "mock")
