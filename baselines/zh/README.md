# HASHET-zh / H-ADAPTS-zh — 审稿人 #2 点名方法的中文适配实现

对应论文修订需求：PONE-D-26-27804 Reviewer #2 要求在 Related Work 与实验中
对比两篇社交媒体动态 hashtag 推荐：

- **HASHET** — Cantini et al., *Learning Sentence-to-Hashtags Semantic Mapping
  for Hashtag Recommendation on Microblogs*, ACM TKDD 16(2), 2022.
  官方代码: https://github.com/SCAlabUnical/HASHET （已克隆至 `../HASHET`）
- **H-ADAPTS** — Cantini et al., *Dynamic Hashtag Recommendation in Social Media
  with Trend Shift Detection and Adaptation*, arXiv:2504.00044, 2025.
  官方代码: https://github.com/SCAlabUnical/H-ADAPTS （已克隆至 `../H-ADAPTS`）

本目录是**忠实中文适配**（faithful Chinese adaptation），方法本体零改动。

## 目录结构

```
zh/
├── common/
│   ├── zh_config.py        # 共享超参（潜空间 150 维等, 官方原值）
│   ├── zh_text.py          # 中文清洗/jieba 分词/#话题# 语法（替换 spaCy 英文）
│   ├── zh_w2v.py           # gensim W2V hashtag 潜在空间（官方参数原值）
│   ├── zh_encoder.py       # bert-base-chinese 编码器（+mock 冒烟后端）
│   └── d1_adapter.py       # D1<->官方 schema 转换 + 小型样例语料生成器
├── hashet_zh/              # HASHET 静态基线
│   ├── constants.py  dataset_handler_zh.py  model_bert_zh.py
│   ├── evaluation_zh.py    # 官方 P/R/F1@k + global/local 扩展, 逐行等价移植
│   └── run.py
├── hadapts_zh/             # H-ADAPTS 动态基线
│   ├── constants.py        # 窗口/阈值/K 官方原值（14d/4d/1d, K=10, τ=0.9）
│   ├── trend_shift.py      # CounterBolt.java 秩加权 Jaccard 检测器的忠实复刻
│   └── run.py              # bootstrap 静态训练 + 滚动窗口检测/重训/预测
├── eval_bridge.py          # 逐窗口预测 -> DKNet 主表四指标（P@10/Coverage/ER/TtC）
├── smoke_test.py           # 端到端冒烟（mock 编码器, 含检测器数学单测）
└── run_experiments.sh      # 5-seeds 正式实验协议
```

## 快速开始

```bash
# 冒烟测试（无需 GPU/权重, ~2 分钟）
python smoke_test.py

# 真实 D1 数据端到端（data/generate_datasets.py 产出的语料）
python run_real_smoke.py --encoder mock                       # 快速验证
python run_real_smoke.py --encoder bert                       # 正式编码器（需权重）
#   结果目录: real_run/ (mock) 或 real_run_bert/ (bert)

# 单独运行
python hashet_zh/run.py  <official.jsonl> --encoder bert --out <dir>
python hadapts_zh/run.py <official.jsonl> --split-date <unix秒> \
    --encoder bert --w2v-mincount 2 --mincount 3 --out <dir>

# 批量 5 seeds:
bash run_experiments.sh <D1_official_schema.jsonl>
```

真实 D1 转换：`run_real_smoke.py` 内部调用 `convert_real_d1_to_official()`——
keywords 标注以 `#词#` 语法嵌入正文（论文 Table 2 的 hashtag-based 标注协议），
时间戳转 unix 秒。桥接指标（P@10/Coverage/ER/TtC + 漂移词级 TtC）见
`eval_bridge.py`，结果汇总写入 `<workdir>/bridge_metrics.json`。

## 偏离声明表（论文/回复信逐条引用）

| # | 官方实现 | 本适配 | 理由 |
|---|---|---|---|
| D1 | spaCy en_core_web_lg 清洗+词形还原 | jieba 分词 + 中文停用词, 保留原词 | 中文无形态变化; 词形还原不适用 |
| D2 | HASHET 随机 shuffle 后 75/25 划分 | **时序**先后划分（比例仍官方 25%） | 与 DKNet 及其余 14 个 baseline 的时序协议对齐, 避免时间泄漏 |
| D3 | hashtag 语法 `#tag` | 微博 `#话题#` 双井号, 规范化为内部 `#话题` 形态 | 平台语法差异; 内部形态使官方 W2V/近邻逻辑零改动 |
| D4 | bert-base-uncased | bert-base-chinese | 中文语料必需; 与论文 Prompt-LLM baseline 同款编码器, 公平 |
| D5 | GUSE 分支（TF-Hub 英文句编码） | 弃用, 固定 BERT 分支 | GUSE 为英文模型, 无中文等价物 |
| D6 | TensorFlow/Keras（TF1 风格 API） | PyTorch 等价复刻（结构/损失/两阶段日程逐项一致） | 官方代码依赖 Python≤3.11 的 TF; 结构等价性见 model_bert_zh.py 头注释 |
| D7 | 官方按天窗口, 数据为真实推特时间戳 | 窗口时长不变（秒换算）, D1 仿真时间轴直接映射 | 保持官方参数原值, 不替 baseline 调参 |
| D8 | Apache Storm(Java) 流式外壳 + socket 信号 | 单进程同步循环; CounterBolt 检测数学逐行复刻（trend_shift.py 含手算用例单测） | 检测逻辑仅 ~30 行, 复刻等价性可验证; 免去 Storm 集群依赖 |
| D9 | W2V_MINCOUNT=40 / MINCOUNT=60（官方为百万级推特语料默认） | `--w2v-mincount/--mincount` 覆盖参数; 万条工作语料用 2/3, 全量 150k 语料恢复官方值 | 阈值语义是"hashtag 最低频次", 必须随语料规模缩放; 论文中按实际规模声明 |

公平性协议（与其他 14 个 baseline 一致）：相同候选池、相同过滤规则（hashtag
最低频次 MINCOUNT）、相同 top-K 截断、相同 5 seeds 与配对检验。

## 引用

若使用本目录结果，请在论文中引用上述两篇原始论文，并在 Experimental setup
中说明 "faithful Chinese adaptation"（偏离 D1–D8）。
