# S1 — HASHET-zh / H-ADAPTS-zh deviation table

Faithful Chinese adaptations of the released models. Algorithmic logic and official hyperparameters are preserved; only language-bound components change.

| # | Official | This adaptation | Rationale |
|---|---|---|---|
| D1 | spaCy en_core_web_lg clean+lemma | jieba + Chinese stopwords | Chinese has no morphological lemma |
| D2 | HASHET random 75/25 | chronological split (still 25% test) | align with temporal protocol; avoid leakage |
| D3 | `#tag` syntax | Weibo `#话题#` normalized to `#话题` | platform syntax; internal form keeps W2V logic |
| D4 | bert-base-uncased | bert-base-chinese | Chinese corpus; same encoder as Prompt-LLM baseline |
| D5 | GUSE branch | dropped; BERT only | GUSE is English TF-Hub |
| D6 | TensorFlow/Keras | PyTorch equivalent (structure/loss/schedule matched) | official stack is TF1-era |
| D7 | day windows on real Twitter timestamps | same window lengths in seconds | no retuning of official windows |
| D8 | Apache Storm + socket | single-process loop; CounterBolt math line-faithful | detection is ~30 lines; unit-tested |
| D9 | W2V_MINCOUNT / MINCOUNT defaults for 100M-scale | scaled for weak-label density (documented per run) | thresholds are frequency semantics |

Source code: `baselines/zh/`. Official clones: `baselines/HASHET`, `baselines/H-ADAPTS`.
