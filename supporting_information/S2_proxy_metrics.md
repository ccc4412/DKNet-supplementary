# S2 — Real weak-label proxy metrics (MicroBlogPCU)

Protocol and package notes: `data/README_proxy.md`, `data/eval_protocol.md`.

## DKNet proxy (from weak_eval / ablations; n_label_rows = 4834 for full weak_eval)

| Method | P@10 | R@10 | F1@10 | hit@10 |
|---|---:|---:|---:|---:|
| exact_label_string_match (reference) | 0.1806 | 0.1960 | 0.1880 | 0.2116 |
| char_ngram_tfidf | 0.0100 | 0.0716 | 0.0176 | 0.0807 |
| char_ngram_cooccurrence | 0.0092 | 0.0638 | 0.0161 | 0.0728 |
| proxy_kg_only | 0.0076 | 0.0673 | 0.0137 | 0.0757 |
| proxy_hybrid_no_mmr | 0.0128 | 0.0967 | 0.0226 | 0.1096 |
| proxy_hybrid_mmr | 0.0130 | 0.0981 | 0.0230 | 0.1113 |
| proxy_adaptive_hybrid_mmr | 0.0130 | 0.0979 | 0.0230 | 0.1111 |

## HASHET-zh / H-ADAPTS-zh

As-of BERT runs under `baselines/zh/proxy_runs_bert/` (as-of refresh=1500). Aggregate file: `aggregate_5seeds.json`.

### Multi-seed mean ± std (n = 5)

| Method | Seeds | P@10 | R@10 | F1@10 | hit@10 |
|---|---|---:|---:|---:|---:|
| HASHET-zh | 42, 1, 2, 3, 4 | **0.0070 ± 0.0005** | 0.061 ± 0.004 | 0.0123 ± 0.0008 | 0.068 ± 0.005 |
| H-ADAPTS-zh | 42, 1, 2, 3, 4 | **0.0084 ± 0.0003** | 0.072 ± 0.003 | 0.0146 ± 0.0005 | 0.080 ± 0.004 |

Per-seed P@10: HASHET {0.0064, 0.0072, 0.0066, 0.0076, 0.0072}; H-ADAPTS {0.0084, 0.0082, 0.0088, 0.0086, 0.0080}.

### Single full-stream reference (refresh≈1000)

| Method | n labeled rows | P@10 | R@10 | hit@10 |
|---|---:|---:|---:|---:|
| HASHET-zh | 3060 | 0.0102 (bootstrap CI [0.0092, 0.0114]) | 0.085 (CI [0.076, 0.095]) | 0.099 (CI [0.089, 0.110]) |

Note: absolute values are an order of magnitude below the synthetic main table because labels are sparse and post-specific. H-ADAPTS is slightly above HASHET on this sparse weak-label stream (supervised in hashtag space); DKNet still does not require hashtags at inference time.
