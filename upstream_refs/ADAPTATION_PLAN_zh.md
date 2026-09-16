# HASHET / H-ADAPTS 中文适配实现方案

> 目标：将审稿人 #2 点名的两个 hashtag 推荐方法（官方代码已克隆至 `baselines/HASHET` 与 `baselines/H-ADAPTS`）适配为中文微博式合成语料 D1 上的 baseline，纳入主表对比。
> 原则：**最小改动、忠实复现**——只替换语言相关的组件（分词、编码器、hashtag 语法），算法逻辑与超参数保持原样；所有偏离处 在论文/回复信中显式声明为 "faithful Chinese adaptation"。

---

## 1. 官方代码结构分析

### 1.1 HASHET（静态模型，纯 Python，约 1,200 行）

```
HASHET/
├── constants.py            # 全部超参数与路径（编码器选择、数据集选择、主题词表等）
├── dataset_handler.py      # 预处理：spaCy 清洗/词形还原、hashtag 提取、train/test 划分、
│                           #   构造 (句子 → hashtag 目标向量) 训练对
├── word_embedding_model.py # gensim Word2Vec：hashtag 潜在空间（dim=150）
├── model.py / model_BERT.py / model_GUSE.py
│                           # MLP 映射网络：BERT pooler(768) → 768 → 512 → 341 → 150，
│                           #   cosine distance 损失；两阶段：transfer(冻结BERT) → fine-tune(3e-5)
│                           #   预测：句向量 → W2V 空间 similar_by_vector → 取 hashtag → top-k
├── evaluation.py           # precision/recall/F1 @ k=1..6，global/local 语义扩展策略
├── topic_discovery.py      # 主题发现任务评测（可不用）
└── run.py                  # 主流程：预处理 → W2V → 句嵌入 → MLP 训练 → 预测 → 评测
```

**输入格式**（JSONL，每行一条）：
```json
{"text": "...", "date": "...", "isRetweet": false, "hashtags": ["#tag1", ...]}
```

**核心管线**：`文本 → spaCy(en_core_web_lg) 清洗/lemma → BERT 句嵌入` 与 `hashtag 集合 → Word2Vec 潜在空间`，MLP 学习句空间 → hashtag 空间的"翻译"，预测时在 W2V 词表中做向量近邻检索。

**关键超参**（`constants.py`）：`LATENT_SPACE_DIM=150`、`MINCOUNT=10`（hashtag 入表阈值）、`PERC_TEST=0.25`、`MAX_EXPANSION_ITERATIONS=6`、`BATCH_SIZE=32`、`PATIENCE=3`。

### 1.2 H-ADAPTS（动态模型 = HASHET + 流式自适应）

```
H-ADAPTS/
├── src/main/java/          # Apache Storm 流处理（趋势切换检测部分）
│   ├── spout/FileSpout.java            # 按时间戳逐条发射推文
│   ├── bolt/HashtagReaderBolt.java     # 提取 hashtag
│   ├── bolt/CounterBolt.java           # ★ 趋势切换检测器（见下）
│   └── bolt/SocketBolt.java            # 将 (endTimestamp, train?) 信号发给 Python
└── src/main/resources/     # Python 组件（复用 HASHET 的模型代码 + 实时重训逻辑）
    ├── constants.py        # 窗口/阈值参数、SPLIT_DATE（bootstrap 分界）
    ├── run.py              # ModelRunner：接收切换信号 → 重训/微调 → 滚动预测评测
    ├── model_BERT.py       # static_training() + rt_training()（渐进微调）
    └── dataset_handler.py / evaluation.py / word_embedding_model.py（同 HASHET）
```

**趋势切换检测**（`CounterBolt.java:58-88`，核心仅 ~30 行）：
1. 按 **tumbling window（默认 1 天）** 统计窗口内 hashtag 频次，取 **top-K（K=10）** 为当前 trending 集合；
2. 与上一窗口的 top-K 计算**秩加权 Jaccard 距离**：
   `intersection = Σ min(k−i, k−rank_new(old_i))`，`union = k(k+1)/2`，`jd = 1 − intersection/union`；
3. `jd ≥ THRESHOLD(0.9)` → 判定趋势切换，向 Python 端发 `train=true`；
4. Python 端收到信号后：在 **logic window（14 天）** 上重训 Word2Vec、在 logic + FT window（4 天）上**渐进微调** MLP（`RT_LR=3e-5`，`MAX_EPOCHS_FT=12`）；未检测到切换则沿用当前模型，每个 tumbling window 做一次预测+评测。

**关键超参**：`LOGIC_WINDOW_LENGTH=14d`、`FT_WINDOW_LENGTH=4d`、`TUMBLING_WINDOW_LENGTH=1d`、`K=10`、`THRESHOLD=0.9`、bootstrap 用 `SPLIT_DATE` 之前的数据静态训练。

### 1.3 结论（可行性）

- 两套代码**算法本体完全语言无关**（W2V + 句嵌入 + MLP 翻译 + 窗口统计），语言绑定只发生在三处：**分词/预处理（spaCy 英文模型）、句编码器（bert-base-uncased / GUSE）、hashtag 语法（`#tag`）**。
- 没有任何英文-only 的算法假设 → 中文适配是**替换组件**而非重写，忠实性有保障。
- H-ADAPTS 的 Storm(Java) 只是流式调度外壳，检测逻辑本身 ~30 行 Python 可精确复刻（见 §3.4）。

---

## 2. 语言绑定点清单（需要动的所有位置）

| # | 原实现 | 位置 | 中文适配 | 类型 |
|---|---|---|---|---|
| L1 | spaCy `en_core_web_lg` 清洗+lemma | `dataset_handler.py:34-53`（两仓库相同） | jieba 分词 + 停用词表 + 中文标点正则；**去掉 lemma**（中文无形态变化，保留原词） | 替换 |
| L2 | URL/用户名清理正则 `URL_REGEX` | `dataset_handler.py:11` | 保留并增加中文@、【】话题括号等 | 增强 |
| L3 | hashtag 提取/拼装（`#` + token） | `dataset_handler.py:19-32 (cleaning)`、`hashtags_list()` | 微博话题语法 `#话题#`（双井号包裹）；正则改为 `#([^#\s]{1,30})#`；正文清洗时**保留话题文本** | 替换 |
| L4 | `bert-base-uncased` tokenizer+encoder | `model_BERT.py:30,67,118`（HASHET）、`model_BERT.py`（H-ADAPTS 同名函数） | `bert-base-chinese`（与论文 Prompt-LLM baseline 用的 bert-base-chinese 一致，天然公平） | 替换 |
| L5 | GUSE (TF-Hub 英文句编码) 分支 | `model_GUSE.py`、`constants.py: SENTENCE_ENCODING_MODEL` | 弃用，固定 `BERT` 分支 | 删除 |
| L6 | 英文停用词/`REMOVE_WORDS` | `dataset_handler.py:12` | 中文停用词表（哈工大/百度停用词表） | 替换 |
| L7 | `isRetweet` 字段 | 输入 schema | D1 转换器把 repost 类信号映射为 `isRetweet`（或全置 False，与原文"剔除转推"逻辑对齐） | 映射 |
| L8 | Storm 流处理（Java） | H-ADAPTS 全部 Java 代码 | 纯 Python 复刻 CounterBolt 窗口检测（~40 行），窗口/阈值/K 逐参数一致 | 复刻 |

**不需要动的**：W2V 训练（gensim 对分词后中文无区别）、MLP 结构与两阶段训练、cosine 距离损失、top-k 近邻检索、global/local 扩展评测、H-ADAPTS 的渐进微调与窗口机制、全部超参数值。

---

## 3. 适配实现方案

### 3.1 目录规划

```
baselines/
├── HASHET/  H-ADAPTS/          # 官方原版（只读，作为引用基准）
└── zh/
    ├── common/
    │   ├── d1_adapter.py       # D1 合成语料 → 两仓库 JSON schema 转换器
    │   ├── zh_dataset_handler.py  # 中文分词/清洗/hashtag 提取（替换 L1/L2/L3/L6）
    │   └── zh_encoder.py       # bert-base-chinese 封装（L4）
    ├── hashet_zh/
    │   ├── constants.py        # 复制官方，改编码器=bert-base-chinese、路径、MINCOUNT
    │   ├── run.py              # 流程同官方 run.py，替换 import
    │   └── model_bert_zh.py    # 官方 model_BERT.py，仅换 tokenizer/model 名
    ├── hadapts_zh/
    │   ├── constants.py        # 官方参数原值 + D1 的 SPLIT_DATE / 窗口换算
    │   ├── trend_shift.py      # CounterBolt 的 Python 复刻（秩加权 Jaccard）
    │   ├── model_runner.py     # 官方 run.py ModelRunner 的单进程同步版（去 socket/Storm）
    │   └── model_bert_zh.py    # static_training + rt_training，换编码器
    ├── eval_bridge.py          # 两 baseline 输出 → DKNet 评测协议（P@10 / Coverage / ER / TtC）
    └── run_experiments.sh      # 5 seeds × {HASHET-zh, H-ADAPTS-zh} × D1
```

### 3.2 数据适配（d1_adapter.py）

D1（微博式，15 万帖，10 主题，第 20/36 周脚本化漂移，hashtag-based 标注，5.2 关键词/帖）→ 输出两份：

1. **HASHET 输入**：D1 训练段（按论文协议 80% 时序切分）转 JSONL：
   `{"text": 帖子文本, "date": 时间戳, "isRetweet": 转发标志, "hashtags": 该帖标注的 hashtag 列表}`；
   测试段 20% 作为验证/测试（对齐官方 `PERC_TEST` 逻辑但改用时序切分，避免随机划分破坏时间序——**这是对官方协议的一处必要偏离，需在论文中说明**：原版随机 75/25，我们用时序切分且与 DKNet/其他 baseline 相同）。
2. **H-ADAPTS 输入**：全量按 `date` 排序的 JSONL；`SPLIT_DATE` = 第 20 周漂移点之前的 bootstrap 边界（与 DKNet 的验证期/评测期切分对齐，保证无泄漏）。

注意：hashtag 语法统一转换——D1 的标注若是裸关键词，需包成 `#关键词#` 微博形式；若是原生 hashtag 则直接用。候选池过滤（低频 hashtag `MINCOUNT`）参数对两个 baseline 与 DKNet 的候选集一致。

### 3.3 HASHET-zh 修改点（按文件）

- `zh_dataset_handler.py`：
  - `cleaning()` → jieba.lcut + 停用词过滤 + 保留 `#...#` 话题整体（不切开）；
  - `hashtags_list()` → 正则 `#([^#\s]{1,30})#` 抽取；
  - `clean_and_phrase()` → 去掉 spaCy 依赖（中文无需 phrase detection），URL/emoji 规则保留（D1 本身注入了 emoji/slang/typo 噪声，正好检验噪声鲁棒性）。
- `model_bert_zh.py`：`BertTokenizer.from_pretrained('bert-base-uncased')` → `'bert-base-chinese'`（`model_BERT.py:30` 与 `:118` 两处）；`sequence_len=100` 对中文足够（D1 平均 21.3 tokens）。
- `constants_zh.py`：`SENTENCE_ENCODING_MODEL=BERT`；`LATENT_SPACE_DIM=150`（不动）；`MINCOUNT=10`、`W2V_MINCOUNT=5` 先按官方值跑，若 D1 hashtag 长尾过稀再按 README 建议调大并在论文中记录；`TOPICS=[]`（topic discovery 任务不做，DKNet 无对应任务）。
- 其余（训练两阶段、预测、评测）不改。

### 3.4 H-ADAPTS-zh 修改点

- **trend_shift.py（Storm 复刻）**，严格对照 `CounterBolt.java:58-88`：

  ```python
  def detect_shift(window_hashtags, old_topk, k=10, threshold=0.9):
      counts = Counter(window_hashtags)
      new_topk = [h for h, _ in counts.most_common(k)]
      intersection = sum(
          min(k - i, k - new_topk.index(h))
          for i, h in enumerate(old_topk) if h in new_topk)
      jd = 1 - intersection / (k * (k + 1) / 2)
      return (jd >= threshold), new_topk, jd
  ```
  窗口滚动周期 = 官方 1 天（D1 时间轴按仿真日历换算）；初始 `old_topk` 用 bootstrap 段 top-K（对齐 `run.py` 写 `java_constants.txt` 的 bootstrap 逻辑）。

- **model_runner.py**：把 `run.py` 的 `ModelRunner.process(tup)` 改为直接循环调用（去 socket/tuple），`train` 信号来自 `trend_shift.detect_shift`；重训逻辑（logic window 重训 W2V + logic/FT window 渐进微调，`rt_training` 的 `MAX_EPOCHS_FT=12`、`RT_LR=3e-5`）**逐行保持**。
- Java/Storm/Maven 全部不装——检测数学已精确复刻，论文中注明 "the trend-shift detector is a faithful single-process reimplementation of the released CounterBolt logic (identical windows, K, and rank-weighted Jaccard threshold)"。

### 3.5 评测桥接（eval_bridge.py）——进入 DKNet 主表的关键

两个 baseline 原生输出是"每帖 top-k hashtag"，DKNet 主表指标是 P@10 / Topic Coverage / Entity Recall / TtC。桥接规则（写入论文 Evaluation metrics 节，保证公平）：

| DKNet 指标 | hashtag baseline 的对应计算 |
|---|---|
| Precision@10 | 每窗口对候选池输出 top-10 关键词，与窗口相关性标注比对（与其他 14 个 baseline 同一评测器实例） |
| Topic Coverage | 预测 hashtag 集合按 D1 主题标签映射后，统计窗口内覆盖的 ground-truth 主题数 |
| Entity Recall | hashtag 去话题符号后与实体标注匹配（字符串归一化同 DKNet） |
| TtC | 新主题出现后首次被该 baseline 的 top-K 捕获的延迟，窗口粒度与 DKNet 相同 |

候选池、过滤规则、top-K 协议与主表其余 14 个 baseline 完全一致（论文已有该声明，直接适用）。

### 3.6 环境与依赖

- 官方要求 Python 3.7 + TF（Keras）。建议：`tensorflow==2.15`（`tf.compat.v1` 兼容层可用）+ `transformers==4.30`（`TFBertModel` 可用）+ `gensim==4.3` + `jieba`。若旧 Keras API 报错（`keras.layers.core` 路径已废），最小修补 import 路径，不动模型结构。
- 硬件：BERT-base 训练两阶段，单张 A100 足够（与论文环境一致）。
- 预计工作量：数据适配 + HASHET-zh 1.5–2 天；H-ADAPTS-zh（含复刻验证）2–3 天；评测桥接 + 5 seeds 实验 1–2 天。

### 3.7 公平性风险与论文声明清单

1. **编码器差异**：官方用英文 BERT，我们用 bert-base-chinese——与论文 Prompt-LLM baseline 同款编码器，公平且需声明。
2. **时序切分 vs 随机切分**：官方 HASHET 用随机 75/25，我们统一用时序 80/20（对所有方法一致），需声明。
3. **Storm → 单进程复刻**：仅调度方式变化，检测数学一致，需声明（见 §3.4）。
4. **合成语料与真实推特分布差异**：H-ADAPTS 的窗口参数（14/4/1 天）在 D1 的 12 个月仿真时间轴上换算后保持原值，不做调参——避免"我们替 baseline 调参"的质疑，也避免"故意不调"的质疑：**用官方默认值 + 敏感性说明**。
5. 引用条目：HASHET → Cantini et al., ACM TKDD 16(2), 2022, DOI 10.1145/3466876；H-ADAPTS → Cantini et al., arXiv:2504.00044（查证其期刊版信息后按正式出处引用）。加入 `references_plos.bib`，Related Work 新增 "Dynamic hashtag recommendation on social media" 小节。

---

## 4. 验收标准

- [ ] HASHET-zh 在 D1 测试段输出与官方代码在英文样本上行为等价（先用官方 `input/Sample.txt` 做冒烟测试，确认改动只来自语言组件）。
- [ ] H-ADAPTS-zh 的趋势切换检测在第 20/36 周脚本化漂移点附近触发（定性行为与原文案例一致），触发次数记录进论文。
- [ ] 两个 baseline 以 5 seeds 进主表，配对 t 检验协议与其余 14 baseline 一致。
- [ ] 论文新增对比小节 + 回复信逐点说明适配声明清单（§3.7）。
