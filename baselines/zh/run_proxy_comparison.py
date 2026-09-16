# -*- coding: utf-8 -*-
"""microblogPCU proxy 三方对比: DKNet(已有 jsonl) vs HASHET-zh vs H-ADAPTS-zh。

协议（对齐 DKNet checkpoint 运行）:
  - 行序 = CSV 文件序（DKNet 的 context 顺序）; 弱标签 = keywords(hashtag)
  - HASHET-zh(as-of): 每 CHUNK=300 行刷新一次模型——用该行之前所有弱标注行
    训练 W2V+MLP（两阶段官方日程）, 预测当前 chunk 内的弱标注行
  - H-ADAPTS-zh(流式): tumbling window=300 行(伪时间, 偏离 D10), K=10, τ=0.9,
    logic window=900 行, FT window=300 行; 切换触发重训, 每窗口预测
  - 预测落盘为 DKNet 同款 jsonl（row_index/labels/predicted）→ 同一评测器
  - 指标: P@10(分母固定10) / R@10 / F1@10 / hit@10, 仅在含标签行上平均

用法:
  python run_proxy_comparison.py --encoder mock --max-rows 600   # 快速验证
  python run_proxy_comparison.py --encoder bert                  # 正式
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eval_proxy import load_proxy_rows, PROXY_DIR
from common.zh_text import clean_line, sentence_for_embedding
from common import zh_w2v as emb
from common.zh_config import seed_all

CHUNK = 300          # DKNet context checkpoint 粒度
W2V_MINCOUNT = 2     # 偏离 D9: 弱标注行数稀少
MINCOUNT = 2
K_HADAPTS = 10
TAU = 0.9
LOGIC_WINDOW = 900
FT_WINDOW = 300


def build_train_pairs(label_rows, w2v):
    """弱标注行 -> (去#句子, hashtag均值向量, hashtag列表)（官方语义）。"""
    X, Y, HT = [], [], []
    for r in label_rows:
        tokens = clean_line(r["asof_text"])
        hts = [t for t in tokens.split() if t.startswith("#")]
        h = emb.tweet_arith_embedding(w2v, " ".join(hts))
        if h is not None:
            Y.append(h)
            X.append(sentence_for_embedding(tokens))
            HT.append(hts)
    return X, Y, HT


def predict_rows(model, w2v, rows, topk=10):
    X = [sentence_for_embedding(clean_line(r["asof_text"])) for r in rows]
    if not X:
        return {}
    preds = model.predict_top_k_hashtags(X, 200, w2v=w2v)
    out = {}
    for r, plist in zip(rows, preds):
        # 官方 retain_hts 后去 '#' 号, 与 DKNet 预测词形对齐
        kws = []
        for t, _ in plist:
            k = t[1:] if t.startswith("#") else t
            if k not in kws:
                kws.append(k)
        out[r["idx"]] = kws[:topk]
    return out


def write_jsonl(path, rows, pred_by_idx, variant):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            if not r["has_label"] or r["idx"] not in pred_by_idx:
                continue
            f.write(json.dumps({"row_index": r["idx"] + 1, "variant": variant,
                                "labels": r["labels"],
                                "status": "ok",
                                "predicted": [[k, 0.0] for k in pred_by_idx[r["idx"]]]},
                               ensure_ascii=False) + "\n")


def m_from_jsonl(path, topk=10):
    P = R = F1 = HIT = 0.0
    n = 0
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        gold = set(d.get("labels") or [])
        if not gold:
            continue
        preds = [x[0] for x in (d.get("predicted") or [])][:topk]
        inter = sum(1 for x in preds if x in gold)
        p = inter / topk
        r = inter / len(gold)
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        P += p; R += r; F1 += f1; HIT += 1 if inter else 0
        n += 1
    return {"n": n, "P@10": round(P / n, 4), "R@10": round(R / n, 4),
            "F1@10": round(F1 / n, 4), "hit@10": round(HIT / n, 4)} if n else {"n": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="mock", choices=["mock", "bert"])
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--refresh", type=int, default=CHUNK,
                    help="HASHET-zh as-of 模型刷新间隔（行数）")
    ap.add_argument("--eval-last", type=int, default=None,
                    help="单点全历史模式: 用前 (total-N) 行训练一次, 评测末 N 行")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.environ["ZH_ENCODER"] = args.encoder
    os.environ["ZH_SEED"] = str(args.seed)
    seed_all(args.seed)

    workdir = args.out or os.path.join(HERE, f"proxy_run_{args.encoder}_s{args.seed}")
    os.makedirs(workdir, exist_ok=True)

    rows = load_proxy_rows()
    if args.max_rows:
        rows = rows[:args.max_rows]
    n_label = sum(1 for r in rows if r["has_label"])
    print(f"rows={len(rows)} label_rows={n_label}")

    # 弱标注 hashtag 以 #词# 嵌入正文（与 DKNet 输入剥离协议互补: 输入仍为裸文本,
    # 训练 target 用嵌入形态; 预测输入 sentence_for_embedding 会剥 #）
    for r in rows:
        r["asof_text"] = r["text"] + (" " + " ".join(f"#{k}#" for k in r["labels"])
                                      if r["has_label"] else "")

    from hashet_zh import model_bert_zh as mz
    from hashet_zh import constants as hc
    from collections import Counter
    from hadapts_zh.trend_shift import TrendShiftDetector

    # 权重/缓存按 seed 隔离，避免并行或连续跑种子时互相覆盖
    hc.TL_WEIGHTS = os.path.join(workdir, "hashet_TLweights.pt")
    hc.FT_WEIGHTS = os.path.join(workdir, "hashet_FTweights.pt")
    hc.W2V_SAVE_FILE_NAME = os.path.join(workdir, "w2v_hashet.model")
    hc.SEED = args.seed
    hc.ENCODER_BACKEND = args.encoder
    # mock 冒烟可加快；bert 保持官方 50
    if args.encoder == "mock":
        hc.MAX_EPOCHS = 12
    else:
        hc.MAX_EPOCHS = 50

    from hadapts_zh import constants as ac
    ac.SEED = args.seed
    ac.ENCODER_BACKEND = args.encoder
    ac.FT_WEIGHTS = os.path.join(workdir, "hadapts_FTweights.pt")
    ac.W2V_SAVE_FILE_NAME = os.path.join(workdir, "w2v_hadapts.model")

    # ---------------- HASHET-zh ----------------
    hashet_path = os.path.join(workdir, "hashet_zh.jsonl")
    open(hashet_path, "w").close()
    w2v_path = os.path.join(workdir, "w2v_hashet.model")

    if args.eval_last:
        # 单点全历史: 一次训练, 末段评测（对标 weak_eval 的 proxy_hybrid_mmr）
        # 官方 HASHET: W2V 在全语料上无监督训练（含测试文本）；MLP 仅用历史弱标注。
        n_eval = args.eval_last
        cut = len(rows) - n_eval
        hist = [r for r in rows[:cut] if r["has_label"]]
        eval_rows = [r for r in rows[cut:] if r["has_label"]]
        print(f"[full-history] train on {len(hist)} label rows, eval {len(eval_rows)}")
        sentences = [clean_line(r["asof_text"]).split() for r in rows]
        w2v = emb.create_and_train_word2vec(sentences, mincount=W2V_MINCOUNT,
                                            save_path=w2v_path)
        X, Y, _ = build_train_pairs(hist, w2v)
        n_val = max(1, int(len(X) * 0.25))
        model = mz.HashetModel()
        model.transfer_and_fine_tune(X[:-n_val], X[-n_val:], Y[:-n_val], Y[-n_val:])
        preds = predict_rows(model, w2v, eval_rows)
        write_jsonl(hashet_path, eval_rows, preds, "hashet_zh_full")
    else:
        print("== HASHET-zh (as-of chunked) ==")
        model = None
        w2v = None
        for start in range(0, len(rows), args.refresh):
            chunk = rows[start:start + args.refresh]
            hist = [r for r in rows[:start] if r["has_label"]]
            eval_rows = [r for r in chunk if r["has_label"]]
            if not eval_rows:
                continue
            if hist:
                print(f" chunk@{start}: train on {len(hist)} label rows, "
                      f"eval {len(eval_rows)}")
                sentences = [clean_line(r["asof_text"]).split() for r in hist]
                w2v = emb.create_and_train_word2vec(sentences, mincount=W2V_MINCOUNT,
                                                    save_path=w2v_path)
                X, Y, HT = build_train_pairs(hist, w2v)
                n_val = max(1, int(len(X) * 0.25))
                model = mz.HashetModel()
                model.transfer_and_fine_tune(X[:-n_val], X[-n_val:],
                                             Y[:-n_val], Y[-n_val:])
                preds = predict_rows(model, w2v, eval_rows)
            else:
                preds = {}  # 冷启动: 无监督信号（与 DKNet ctx10/30=0 对应）
                print(f" chunk@{start}: cold start, no labeled history")
            write_jsonl(hashet_path, eval_rows, preds, "hashet_zh")

    # ---------------- H-ADAPTS-zh streaming ----------------
    print("== H-ADAPTS-zh (streaming windows) ==")
    hadapts_path = os.path.join(workdir, "hadapts_zh.jsonl")
    open(hadapts_path, "w").close()
    w2v2_path = os.path.join(workdir, "w2v_hadapts.model")
    detector = TrendShiftDetector(k=K_HADAPTS, threshold=TAU)
    model2, w2v2 = None, None
    shifts = []

    if args.eval_last:
        # bootstrap: 切分点前全部弱标注行 -> 静态训练; 之后流式滚动
        n_eval = args.eval_last
        cut = len(rows) - n_eval
        boot = [r for r in rows[:cut] if r["has_label"]]
        print(f"[hadapts bootstrap] {len(boot)} label rows")
        # 与官方一致: W2V 用全语料文本（无监督）; 模型只用 bootstrap 弱标注
        w2v2 = emb.create_and_train_word2vec(
            [clean_line(r["asof_text"]).split() for r in rows],
            mincount=W2V_MINCOUNT, save_path=w2v2_path)
        X, Y, _ = build_train_pairs(boot, w2v2)
        n_val = max(1, int(len(X) * 0.25))
        model2 = mz.HashetModel()
        model2.transfer_and_fine_tune(X[:-n_val], X[-n_val:], Y[:-n_val], Y[-n_val:])
        boot_counts = Counter(k for r in boot for k in r["labels"])
        detector = TrendShiftDetector(k=K_HADAPTS, threshold=TAU,
                                      initial_counts=boot_counts)
        stream = rows[cut:]
    else:
        stream = rows

    for start in range(0, len(stream), CHUNK):
        window = stream[start:start + CHUNK]
        eval_rows = [r for r in window if r["has_label"]]
        tags = [k for r in window if r["has_label"] for k in r["labels"]]
        shifted, topk, jd = detector.observe(tags)
        if args.eval_last:
            pre = stream[:start]
        else:
            pre = rows[:start]
        logic_hist = [r for r in pre[-LOGIC_WINDOW:] if r["has_label"]]
        if eval_rows:
            print(f" window@{start}: jd={jd:.3f} shifted={shifted} "
                  f"train={len(logic_hist)} eval={len(eval_rows)}")
        if shifted:
            shifts.append({"start": start, "jd": round(jd, 4), "topk": topk})
        if shifted and logic_hist:
            sentences = [clean_line(r["asof_text"]).split() for r in logic_hist]
            w2v2 = emb.create_and_train_word2vec(sentences, mincount=W2V_MINCOUNT,
                                                 save_path=w2v2_path)
            X, Y, _ = build_train_pairs(logic_hist, w2v2)
            if len(X) >= 4:
                n_val = max(1, int(len(X) * 0.25))
                if model2 is None:
                    model2 = mz.HashetModel()
                    model2.transfer_and_fine_tune(X[:-n_val], X[-n_val:],
                                                  Y[:-n_val], Y[-n_val:])
                else:
                    model2.fine_tune_short(X[:-n_val], X[-n_val:],
                                           Y[:-n_val], Y[-n_val:])
        preds = {}
        if model2 is not None and w2v2 is not None and eval_rows:
            preds = predict_rows(model2, w2v2, eval_rows)
        write_jsonl(hadapts_path, eval_rows, preds, "hadapts_zh")

    json.dump(shifts, open(os.path.join(workdir, "shifts.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---------------- 汇总（始终从最终 jsonl 复算，避免 stale summary）----------------
    print("== metrics (recomputed from jsonl) ==")
    res = {"seed": args.seed, "encoder": args.encoder,
           "hashet_zh": m_from_jsonl(hashet_path),
           "hadapts_zh": m_from_jsonl(hadapts_path),
           "n_shifts": len(shifts),
           "eval_last": args.eval_last,
           "refresh": None if args.eval_last else args.refresh}
    json.dump(res, open(os.path.join(workdir, "summary.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
