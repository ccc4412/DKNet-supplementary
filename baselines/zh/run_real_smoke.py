# -*- coding: utf-8 -*-
"""真实 D1 数据端到端流水线（data/datasets 产出 -> DKNet 桥接指标）。

用法:
  python run_real_smoke.py [--d1 data/datasets/d1_weibo_temporal.json] [--encoder mock]
  可选: --quick 只跑 HASHET-zh; --max-windows N 限制 H-ADAPTS 窗口数

步骤:
  1. 真实 D1 -> 官方 schema（关键词以 #词# 嵌入正文, 论文 hashtag-based 标注协议）
  2. HASHET-zh（时序 75/25, mock/bert 编码器）-> 官方 P/R/F1
  3. H-ADAPTS-zh（bootstrap 至第 19 周末, 全年滚动窗口, mincount 按语料规模缩放）
     -> 逐窗口预测 + 切换日志
  4. eval_bridge: 逐窗口 top-10 预测 vs keywords 标注 -> P@10/Coverage/ER/TtC
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common.d1_adapter import convert_real_d1_to_official, load_real_d1_groundtruth
from eval_bridge import (window_predictions_to_metrics, build_real_windows,
                         drift_keyword_ttc)
from collections import OrderedDict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1", default=os.path.join(HERE, "..", "..", "data", "datasets_d10k",
                                                 "d1_weibo_temporal.json"))
    ap.add_argument("--encoder", default="mock", choices=["mock", "bert"])
    ap.add_argument("--quick", action="store_true", help="只跑 HASHET-zh")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--split-date", default=None,
                    help="bootstrap 切分 (ISO 或 unix 秒); 默认 2023-05-15(第20周漂移前)")
    ap.add_argument("--workdir", default=None, help="结果目录（默认 real_run[_encoder]）")
    args = ap.parse_args()

    workdir = args.workdir or (os.path.join(HERE, "real_run")
                               if args.encoder == "mock"
                               else os.path.join(HERE, f"real_run_{args.encoder}"))
    os.makedirs(workdir, exist_ok=True)
    official = os.path.join(workdir, "official.jsonl")
    n = convert_real_d1_to_official(args.d1, official)
    print(f"[1] converted {n} real D1 records -> {official}")

    env = dict(os.environ, ZH_ENCODER=args.encoder, PYTHONIOENCODING="utf-8")

    # ---- 2. HASHET-zh ----
    print("[2] HASHET-zh ...")
    r = subprocess.run([sys.executable, os.path.join(HERE, "hashet_zh", "run.py"),
                        official, "--encoder", args.encoder,
                        "--out", os.path.join(workdir, "hashet")], env=env,
                       capture_output=True, text=True, encoding="utf-8")
    print("\n".join(r.stdout.strip().splitlines()[-8:]))
    if r.returncode != 0:
        print(r.stderr[-2000:]); sys.exit(1)

    if args.quick:
        print("[done] HASHET-zh only")
        return

    # ---- 3. H-ADAPTS-zh ----
    if args.split_date is None:
        split = int(datetime(2023, 5, 15).timestamp())  # 第 20 周漂移开始前
    else:
        try:
            split = int(args.split_date)
        except ValueError:
            split = int(datetime.fromisoformat(args.split_date).timestamp())
    print(f"[3] H-ADAPTS-zh ... (split={split})")
    cmd = [sys.executable, os.path.join(HERE, "hadapts_zh", "run.py"), official,
           "--split-date", str(split), "--encoder", args.encoder,
           "--w2v-mincount", "2", "--mincount", "3",
           "--out", os.path.join(workdir, "hadapts")]
    if args.max_windows:
        cmd += ["--max-windows", str(args.max_windows)]
    r2 = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8")
    print("\n".join(r2.stdout.strip().splitlines()[-12:]))
    if r2.returncode != 0:
        print(r2.stderr[-2000:]); sys.exit(1)

    # ---- 4. bridge ----
    real = load_real_d1_groundtruth(args.d1)
    recs_by_win = build_real_windows(real)
    preds_by_win = OrderedDict()
    pred_file = os.path.join(workdir, "hadapts", "predictions.jsonl")
    for line in open(pred_file, encoding="utf-8"):
        p = json.loads(line)
        # predictions 的 window 索引基于 split 时刻, 桥接窗口基于语料起点:
        # 用 ts 反查桥接窗口号
        t0 = real[0]["ts"]
        wi = int((p["ts"] - t0) // 86400)
        preds_by_win.setdefault(wi, [])
        for t in p["pred_tags"]:
            if t not in [x[0] for x in preds_by_win[wi]]:
                preds_by_win[wi].append((t, 0.0))
    m = window_predictions_to_metrics(preds_by_win, recs_by_win, topk=10,
                                      window_minutes=1440)
    print(f"[4] H-ADAPTS-zh DKNet-style metrics: {json.dumps(m, ensure_ascii=False)}")

    # 漂移关键词级 TtC（合成 D1 上与论文 TtC 定义对应的可观测量）
    stats_path = os.path.join(os.path.dirname(args.d1), "dataset_statistics.json")
    drift_kws = set()
    if os.path.exists(stats_path):
        st = json.load(open(stats_path, encoding="utf-8"))["concept_drift"]
        for mp in ("drift_week_1_new_topics", "drift_week_2_new_topics"):
            drift_kws.update(mp and st[mp].values())
    if drift_kws:
        dt = drift_keyword_ttc(preds_by_win, real, drift_kws, topk=10,
                               window_minutes=1440)
        m["drift_ttc"] = {k: v for k, v in dt.items() if k != "per_keyword"}
        print(f"[4] drift-keyword TtC: captured={dt['captured']} "
              f"ttc={dt['ttc_windows']} windows ({dt['ttc_min']} min)")
        out_rows = dt["per_keyword"]

    shifts = json.load(open(os.path.join(workdir, "hadapts", "shifts.json"),
                            encoding="utf-8"))
    jan1 = datetime(2023, 1, 1).timestamp()
    for s in shifts:
        week = int((s["ts"] - jan1) // (7 * 86400)) + 1
        print(f"    shift @ window {s['window']} -> week {week}, jd={s['jd']}, "
              f"top5={s['topk'][:5]}")
    out = {"hadapts_bridge": m, "n_shifts": len(shifts)}
    with open(os.path.join(workdir, "bridge_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[done]")


if __name__ == "__main__":
    main()
