# DKNet — PLOS ONE Supplementary Package

**Manuscript ID:** PONE-D-26-27804  
**Title:** Dynamic keyword generation for social media trend monitoring via LLM-enhanced knowledge graphs  

This repository is the public data-and-code package accompanying the manuscript revision. It contains Supporting Information (S1–S3), the MicroBlogPCU weak-label proxy materials, Chinese adaptations of HASHET / H-ADAPTS baselines, and statistical scripts (bootstrap CI / Wilcoxon).

## DOI / permanent link

- **Repository URL:** *(fill after first push)*
- **DOI:** *pending mint (Zenodo/GitHub Release) — replace this line before production*

## Contents

| Path | Description |
|------|-------------|
| `supporting_information/S1_baseline_deviations.md` | Chinese adaptation deviations (HASHET-zh / H-ADAPTS-zh) |
| `supporting_information/S2_proxy_metrics.md` | MicroBlogPCU weak-label proxy metrics (n=5 seeds for hashtag baselines) |
| `supporting_information/S3_*.md` / `S3_*.json` | Bootstrap CIs, Wilcoxon signed-rank outputs, multi-seed summary |
| `supporting_information/data/` | Weak-label protocol, proxy CSV (~31.5k posts), weak-eval summary |
| `baselines/zh/` | Runnable Chinese adaptations and unified evaluation scripts |
| `aggregate/aggregate_5seeds.json` | 5-seed mean±std for HASHET-zh / H-ADAPTS-zh |
| `upstream_refs/` | Notes on official HASHET / H-ADAPTS repositories |

Official upstream projects (please cite the original papers if you use those methods):

- HASHET — *Learning Sentence-to-Hashtags Semantic Mapping for Hashtag Recommendation on Microblogs*
- H-ADAPTS — *Dynamic Hashtag Recommendation in Social Media with Trend Shift Detection and Adaptation*

## Evaluation layers (important)

Absolute P@10 values are **not** comparable across layers:

| Layer | Label density | Typical P@10 order |
|-------|---------------|--------------------|
| Synthetic main table (D1/D2 in the paper) | dense controlled labels | ~0.65–0.83 |
| Semi-real event probe | external event seeds + synthetic surfaces | ~0.70–0.78 |
| **This package: MicroBlogPCU weak labels** | ~15% rows, mean 1.31 tags/row | **~0.007–0.013** |

This package answers whether **directional** conclusions (KG contribution; hybrid ranking vs simple statistics; competitive hashtag baselines without inference-time hashtags) hold on real text under a shared top-K protocol.

## Headline 5-seed results (BERT as-of / streaming, refresh=1500)

| Method | P@10 | R@10 | hit@10 |
|--------|------|------|--------|
| HASHET-zh | 0.0070 ± 0.0005 | 0.061 | 0.068 |
| H-ADAPTS-zh | 0.0084 ± 0.0003 | 0.072 | 0.080 |
| DKNet proxy hybrid (reference) | ~0.013 | ~0.098 | ~0.111 |

See `aggregate/aggregate_5seeds.json` and `supporting_information/S2_proxy_metrics.md`.

## Reproducing baseline metrics (overview)

Requirements: Python 3.10+, PyTorch, gensim, transformers (`bert-base-chinese`), jieba.

```bash
# from baselines/zh
# 1) unified proxy comparison (single seed)
python run_proxy_comparison.py --encoder bert --seed 42 --refresh 1500 --out /tmp/seed42

# 2) multi-seed aggregate
python run_proxy_5seeds.py --encoder bert --refresh 1500

# 3) bootstrap / Wilcoxon
python stats_bootstrap_wilcoxon.py --jsonl /tmp/seed42/hashet_zh.jsonl --out S3_bootstrap_hashet_seed42.json
python stats_bootstrap_wilcoxon.py --compare /tmp/seed42/hashet_zh.jsonl /tmp/seed42/hadapts_zh.jsonl --out S3_wilcoxon.json
```

Data path defaults are documented in `baselines/zh/eval_proxy.py` and `supporting_information/data/README_proxy.md`. Point the proxy loader at `supporting_information/data/microblogpcu_dknet_proxy.csv`.

## Ethics / license

- MicroBlogPCU posts follow the source dataset distribution terms. Hashtags were stripped from model inputs and retained only as weak evaluation labels.
- No private identifiable user data beyond the public MicroBlogPCU release is used.
- Code in this package: MIT (see `LICENSE`).
- Document/data narrative: CC BY 4.0 preferred for the paper SI text.

## Citation

Please cite the manuscript (PLOS ONE, PONE-D-26-27804) and the original HASHET / H-ADAPTS papers if you use those baselines.
