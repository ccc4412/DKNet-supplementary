# -*- coding: utf-8 -*-
"""microblogPCU proxy 上 HASHET-zh / H-ADAPTS-zh 的 5-seed 正式跑批。

协议（与 COMPARISON_REPORT §1 对齐，便于与 DKNet weak_eval 同层比较）:
  - 数据: proxy_real_data/microblogpcu_dknet_proxy.csv（全量 31,530 行）
  - HASHET-zh: eval-last——前 (N_total - N_eval) 弱标注行训练一次, 评测末 N_eval
  - H-ADAPTS-zh: 同一切分 bootstrap + 流式窗口滚动
  - 编码器: bert-base-chinese；种子 42/1/2/3/4

用法（在 baselines/zh 下）:
  python run_proxy_5seeds.py --encoder bert --refresh 1500
  python run_proxy_5seeds.py --encoder bert --refresh 1500 --seeds 42,4
  python run_proxy_5seeds.py --encoder mock --eval-last 800   # 快速冒烟
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 1, 2, 3, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="bert", choices=["mock", "bert"])
    ap.add_argument("--eval-last", type=int, default=800,
                    help="末段评测行数（含标签行会少于此值，按文件末尾 N 行切）")
    ap.add_argument("--refresh", type=int, default=0,
                    help="0=使用 eval-last 单次训练; >0=as-of chunked 刷新间隔")
    ap.add_argument("--seeds", type=str, default=None,
                    help="逗号分隔种子子集，默认 SEEDS 全量（如 42,4）")
    args = ap.parse_args()
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS)

    # 避免 Windows 上 gensim/numpy 与 torch 的 OpenMP 双重初始化崩溃
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    root = os.path.join(HERE, f"proxy_runs_{args.encoder}")
    os.makedirs(root, exist_ok=True)
    all_summaries = []

    for seed in seeds:
        out = os.path.join(root, f"seed{seed}")
        os.makedirs(out, exist_ok=True)
        env = dict(os.environ)
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["ZH_ENCODER"] = args.encoder
        env["ZH_SEED"] = str(seed)
        cmd = [sys.executable, os.path.join(HERE, "run_proxy_comparison.py"),
               "--encoder", args.encoder, "--seed", str(seed),
               "--out", out]
        if args.refresh > 0:
            cmd += ["--refresh", str(args.refresh)]
        else:
            cmd += ["--eval-last", str(args.eval_last)]
        print(f"\n===== SEED {seed} =====", flush=True)
        r = subprocess.run(cmd, cwd=HERE, env=env)
        if r.returncode != 0:
            print(f"[seed {seed}] FAILED", flush=True)
            continue
        sp = os.path.join(out, "summary.json")
        if os.path.exists(sp):
            all_summaries.append(json.load(open(sp, encoding="utf-8")))

    # 汇总 mean ± std
    def agg(key, sub):
        vals = [s[key][sub] for s in all_summaries if key in s and sub in s[key]]
        if not vals:
            return None
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
        return {"mean": round(m, 4), "std": round(sd, 4), "n": len(vals),
                "per_seed": vals}

    # 合并已有成功的 seed summary（支持只重跑失败子集后更新聚合）
    for seed in SEEDS:
        if seed in seeds:
            continue
        sp = os.path.join(root, f"seed{seed}", "summary.json")
        if os.path.exists(sp):
            s = json.load(open(sp, encoding="utf-8"))
            if s.get("refresh") == (args.refresh or None) and s.get("eval_last") is None:
                all_summaries.append(s)

    summary = {
        "encoder": args.encoder,
        "seeds": SEEDS,
        "eval_last": args.eval_last if args.refresh == 0 else None,
        "refresh": args.refresh or None,
        "n_runs": len(all_summaries),
        "hashet_zh": {k: agg("hashet_zh", k) for k in ("P@10", "R@10", "F1@10", "hit@10")},
        "hadapts_zh": {k: agg("hadapts_zh", k) for k in ("P@10", "R@10", "F1@10", "hit@10")},
        "runs": all_summaries,
    }
    out_path = os.path.join(root, "aggregate_5seeds.json")
    json.dump(summary, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n==== AGGREGATE ====")
    print(json.dumps({k: summary[k] for k in ("encoder", "n_runs", "hashet_zh", "hadapts_zh")},
                     ensure_ascii=False, indent=1))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
