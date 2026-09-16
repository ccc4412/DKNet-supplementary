# -*- coding: utf-8 -*-
"""对 proxy / 合成实验的成对结果做 bootstrap CI 与 Wilcoxon 符号秩检验。

回应 Reviewer #1: 样本量小、TtC 非正态时，补充非参数/重采样稳健性。

用法:
  python stats_bootstrap_wilcoxon.py --jsonl baselines/zh/proxy_runs_bert/seed42/hashet_zh.jsonl \
      [--ref-jsonl proxy_real_data/microblogpcu_dknet_proxy.csv] \
      [--out stats_out.json]
  python stats_bootstrap_wilcoxon.py --compare a.jsonl b.jsonl --out cmp.json
"""
import argparse
import json
import os
import random
from collections import defaultdict


def load_jsonl_metrics(path, topk=10):
    """返回每行 (P,R,F1,HIT) 列表（含标签行）。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            gold = set(d.get("labels") or [])
            if not gold:
                continue
            preds = [x[0] for x in (d.get("predicted") or [])][:topk]
            inter = sum(1 for x in preds if x in gold)
            p = inter / topk
            r = inter / len(gold)
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            hit = 1.0 if inter else 0.0
            rows.append((p, r, f1, hit))
    return rows


def bootstrap_ci(vals, n_boot=5000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    means = []
    for _ in range(n_boot):
        s = sum(vals[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    m = sum(vals) / n
    return {"mean": round(m, 6), "ci_low": round(lo, 6),
            "ci_high": round(hi, 6), "n": n, "n_boot": n_boot}


def wilcoxon_signed_rank(a, b):
    """精确/近似 Wilcoxon 符号秩（双侧）。a,b 等长；忽略差为 0 的对。"""
    import math
    diffs = [ai - bi for ai, bi in zip(a, b) if ai != bi]
    n = len(diffs)
    if n == 0:
        return {"statistic": 0.0, "n_pairs": 0, "p_two_sided": 1.0}
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for d, r in zip(diffs, ranks) if d > 0)
    w_minus = sum(r for d, r in zip(diffs, ranks) if d < 0)
    stat = min(w_plus, w_minus)
    # 正态近似（n>=10 时足够；小 n 用连续性校正）
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        p = 1.0
    else:
        z = (stat - mu + 0.5) / sigma  # 连续性校正
        # 双侧 p
        p = 2.0 * (0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        p = min(1.0, max(0.0, p))
    return {"statistic": round(stat, 4), "n_pairs": n, "p_two_sided": round(p, 6),
            "w_plus": round(w_plus, 2), "w_minus": round(w_minus, 2)}


def paired_metrics(path_a, path_b, topk=10):
    """按行序对齐两份 jsonl 的指标（要求同一批评测行顺序一致）。"""
    def by_key(path):
        out = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                gold = set(d.get("labels") or [])
                if not gold:
                    continue
                preds = [x[0] for x in (d.get("predicted") or [])][:topk]
                inter = sum(1 for x in preds if x in gold)
                key = d.get("row_index")
                out[key] = (inter / topk, inter / len(gold),
                            1.0 if inter else 0.0)
        return out

    A, B = by_key(path_a), by_key(path_b)
    keys = sorted(set(A) & set(B))
    cols = {"precision": [], "recall": [], "hit": []}
    for k in keys:
        cols["precision"].append((A[k][0], B[k][0]))
        cols["recall"].append((A[k][1], B[k][1]))
        cols["hit"].append((A[k][2], B[k][2]))
    report = {"n_paired_rows": len(keys)}
    for name, pairs in cols.items():
        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        diffs = [x - y for x, y in pairs]
        report[name] = {
            "mean_a": round(sum(a) / len(a), 4) if a else 0,
            "mean_b": round(sum(b) / len(b), 4) if b else 0,
            "mean_diff": round(sum(diffs) / len(diffs), 6) if diffs else 0,
            "bootstrap_diff_ci": bootstrap_ci(diffs),
            "wilcoxon": wilcoxon_signed_rank(a, b),
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", help="单模型 jsonl: 报告 bootstrap CI")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="两份预测 jsonl: 成对 bootstrap + Wilcoxon")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.compare:
        report = paired_metrics(args.compare[0], args.compare[1])
    elif args.jsonl:
        rows = load_jsonl_metrics(args.jsonl)
        names = ["P@10", "R@10", "F1@10", "hit@10"]
        report = {"n": len(rows)}
        for i, name in enumerate(names):
            vals = [r[i] for r in rows]
            report[name] = bootstrap_ci(vals)
    else:
        ap.error("need --jsonl or --compare")
    text = json.dumps(report, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
