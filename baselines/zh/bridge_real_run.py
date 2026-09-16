# -*- coding: utf-8 -*-
"""补全 real_run / real_run_bert 的 DKNet 风格桥接指标（HASHET + H-ADAPTS）。

用法:
  python bridge_real_run.py --workdir real_run_bert
  python bridge_real_run.py --workdir real_run
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common.d1_adapter import load_real_d1_groundtruth
from eval_bridge import build_real_windows, window_predictions_to_metrics, drift_keyword_ttc


def preds_from_hashet_metrics(hdir):
    """HASHET 官方 metrics.json 不含逐帖预测；用 evaluate 过程中的 TEST 重跑成本高。
    若存在 predictions.jsonl 则直接读；否则返回 None。"""
    p = os.path.join(hdir, "predictions.jsonl")
    if not os.path.exists(p):
        return None
    by_win = {}
    # 无时间戳时无法分窗——交由 hadapts 侧
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--d1", default=os.path.join(
        HERE, "..", "..", "data", "datasets_d10k", "d1_weibo_temporal.json"))
    args = ap.parse_args()
    wd = os.path.abspath(args.workdir)

    real = load_real_d1_groundtruth(args.d1)
    recs_by_win = build_real_windows(real)
    out = {"workdir": wd}

    # ---- H-ADAPTS predictions.jsonl ----
    pred_file = os.path.join(wd, "hadapts", "predictions.jsonl")
    if os.path.exists(pred_file):
        from collections import OrderedDict
        preds_by_win = OrderedDict()
        t0 = real[0]["ts"]
        for line in open(pred_file, encoding="utf-8"):
            p = json.loads(line)
            wi = int((p["ts"] - t0) // 86400)
            preds_by_win.setdefault(wi, [])
            seen = {x[0] for x in preds_by_win[wi]}
            for t in p.get("pred_tags", []):
                if t not in seen:
                    preds_by_win[wi].append((t, 0.0))
                    seen.add(t)
        m = window_predictions_to_metrics(preds_by_win, recs_by_win, topk=10,
                                          window_minutes=1440)
        out["hadapts_bridge"] = m

        stats_path = os.path.join(os.path.dirname(os.path.abspath(args.d1)),
                                  "dataset_statistics.json")
        drift_kws = set()
        if os.path.exists(stats_path):
            st = json.load(open(stats_path, encoding="utf-8"))["concept_drift"]
            for mp in ("drift_week_1_new_topics", "drift_week_2_new_topics"):
                if mp in st and st[mp]:
                    drift_kws.update(st[mp].values())
        if drift_kws:
            dt = drift_keyword_ttc(preds_by_win, real, drift_kws, topk=10,
                                   window_minutes=1440)
            m["drift_ttc"] = {k: v for k, v in dt.items() if k != "per_keyword"}
            out["drift_ttc"] = m["drift_ttc"]

    # ---- HASHET: 若有 metrics.json 仅记录官方指标（主表桥接需逐帖预测）----
    hm = os.path.join(wd, "hashet", "metrics.json")
    if os.path.exists(hm):
        d = json.load(open(hm, encoding="utf-8"))
        r0 = d["rounds"][0]["weighted"]
        out["hashet_official_weighted"] = {
            "precision": r0["precision"], "recall": r0["recall"],
            "f1": r0["f1"], "n_test": d.get("n_test")}

    path = os.path.join(wd, "bridge_metrics.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
