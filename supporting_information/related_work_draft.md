# Related Work 新小节 + Reviewer #2 回复素材

> 供修订使用：新 Related Work 小节草稿、与 DKNet 的差异化定位、实验对比的
> 描述模板。两篇目标文献已入库 references_plos.bib（cantini2022hashet,
> cantini2025hadapts）。

## 1. Related Work 新小节草稿（Dynamic hashtag and keyword recommendation）

> Hashtag and keyword recommendation on social media has evolved from static
> supervised mappings toward methods that explicitly model the temporal
> dynamics of term usage. HASHET~\cite{cantini2022hashet} learns a translation
> between two latent spaces---sentence embeddings and hashtag
> embeddings---with a multilayer perceptron, recommending hashtags for a post
> by nearest-neighbor search in the hashtag space. H-ADAPTS~\cite{cantini2025hadapts}
> extends this line to evolving streams: it monitors the top-$K$ trending
> hashtags per window, detects trend shifts via a rank-weighted Jaccard
> distance, and asynchronously fine-tunes the recommendation model upon a
> detected shift. Both methods operate directly in the hashtag space and adapt
> (or train) on labeled hashtag observations. DKNet differs in three respects:
> (i) it does not assume hashtags are observed in the stream, since the
> keyword frontier must be maintained from entity-level evidence extracted by
> an LLM; (ii) its structural signal comes from a multi-layer temporal
> knowledge graph rather than from hashtag co-occurrence statistics; and
> (iii) adaptation is driven by a contextual bandit over drift, volume, and
> precision-proxy features rather than by a threshold on hashtag-rank overlap.
> We therefore include both methods as strong baselines under our evaluation
> protocol (Section~X), adapting their language components to Chinese
> microblog text.

（若正文篇幅紧张，可压缩为一段，保留 (i)(ii)(iii) 三点差异即可。）

## 2. 差异化定位要点（回复信用）

- 共同点：都面向"演化中的社交媒体词汇"，都有适应机制。
- 关键差异（novelty 声明的锚点）：
  1. **任务输入不同**：H-ADAPTS/HASHET 依赖流中已观测的 hashtag 做监督信号；
     DKNet 的关键词前沿不假设 hashtag 可观测（上游关键词生成循环）。
  2. **结构信号不同**：hashtag 共现统计 vs 多层时序知识图谱（跨层消息传递 +
     时序衰减 + InfoNCE 嵌入）。
  3. **适应机制不同**：top-K 秩加权 Jaccard 阈值触发重训 vs 上下文老虎机
     （漂移指数/音量/精度代理三特征）在线调 (α, β)，并与 MMR 多样化联动。
- 承认点（避免过度声称）：在"hashtag 充分可观测"的场景下，H-ADAPTS 的
  检测-重训机制与 DKNet 的漂移适应解决的是同一问题家族；我们的贡献是
  把关键词生成放进 LLM+KG 管线后的增益。

## 3. 实验小节描述模板（Baselines 段新增）

> \textbf{HASHET-zh}~\cite{cantini2022hashet} adapts the sentence-to-hashtag
> embedding-translation model to Chinese microblog text, replacing the English
> sentence encoder with bert-base-chinese and segmentation with jieba while
> keeping all model hyperparameters at their released defaults.
> \textbf{H-ADAPTS-zh}~\cite{cantini2025hadapts} additionally streams posts
> through the released trend-shift detector (top-10 trending hashtags per
> daily window, rank-weighted Jaccard threshold 0.9) and progressively
> fine-tunes the recommendation model upon detected shifts; the Storm-based
> streaming shell is reimplemented as a single-process loop with identical
> detection semantics. Both baselines receive the same candidate pools,
> filtering rules, and top-$K$ protocol as all other baselines; language
> adaptations are detailed in the Supporting Information.

## 4. 待办提醒

- [ ] H-ADAPTS 目前引用 arXiv 版；投稿前查证其期刊版（ResearchGate 有记录），
      若已见刊改为正式出处。
- [ ] 两个 baseline 的 5-seeds 结果出来后，主表新增两行，摘要/结论中的
      "+7pp / +14pp / −31%" 需重新核算（若最强基线被刷新）。
- [ ] 回复信引用偏离声明表（baselines/zh/README.md D1–D8）。
