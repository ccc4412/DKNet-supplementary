# -*- coding: utf-8 -*-
"""microblogPCU 真实数据统一评测。

同一弱标注协议（hashtag 从输入剥离、仅作弱标签）与同一指标公式下评测:
  - DKNet 已保存的预测（proxy_real_data 的 jsonl）— 复算验证 + 分段对比
  - HASHET-zh / H-ADAPTS-zh 的预测

指标（与 DKNet weak_eval / checkpoint summary 一致）:
  P@10 = mean over eval rows of |pred∩gold| / 10
  R@10 = |pred∩gold| / |gold|
  F1@10、hit@10 = fraction of rows with >=1 match
"""
import csv
import json
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY_DIR = os.path.join(os.path.dirname(HERE), "..", "proxy_real_data")
PROXY_CSV = os.path.join(PROXY_DIR, "microblogpcu_dknet_proxy.csv")


def load_proxy_rows():
    """读 DKNet proxy CSV: text/keywords(弱标签)/timestamp, 保持文件顺序。"""
    rows = []
    with open(PROXY_CSV, encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            kws = [k for k in (r.get("keywords") or "").split("|") if k]
            rows.append({"idx": i, "text": r["text"], "labels": kws,
                         "ts": r["timestamp"], "has_label": bool(kws),
                         "post_id": r.get("source_post_id", "")})
    return rows


def metrics_from_preds(pred_by_row, rows):
    """pred_by_row: {row_idx: [kw, ...]}（已截 top-10）; rows: 评测行子集。
    与 DKNet 汇总相同的公式（P 分母固定 10, 只在含标签行上平均）。"""
    eval_rows = [r for r in rows if r["has_label"]]
    P = R = F1 = HIT = 0.0
    n = 0
    for r in eval_rows:
        preds = pred_by_row.get(r["idx"], [])[:10]
        gold = set(r["labels"])
        if not gold:
            continue
        inter = sum(1 for p in preds if p in gold)
        p = inter / 10.0
        rec = inter / len(gold)
        f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
        P += p; R += rec; F1 += f1; HIT += 1.0 if inter else 0.0
        n += 1
    return {"n_label_rows": n,
            "precision_at_10": round(P / n, 4) if n else 0.0,
            "recall_at_10": round(R / n, 4) if n else 0.0,
            "f1_at_10": round(F1 / n, 4) if n else 0.0,
            "hit_at_10": round(HIT / n, 4) if n else 0.0}


def load_dknet_predictions(jsonl_path, topk=10):
    """DKNet checkpoint jsonl -> {row_index: [kw,...]}。"""
    out = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("status") != "ok":
                continue
            out[d["row_index"]] = [p[0] for p in (d.get("predicted") or [])][:topk]
    return out


def slice_rows(rows, lo, hi):
    """CSV 行区间 [lo, hi)（0-based, 文件顺序 = 时间顺序）。"""
    return [r for r in rows if lo <= r["idx"] < hi]
