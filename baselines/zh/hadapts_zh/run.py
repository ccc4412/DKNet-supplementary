# -*- coding: utf-8 -*-
"""H-ADAPTS-zh 主运行器：官方 run.py ModelRunner 的单进程同步版。

流程对照官方:
  bootstrap: SPLIT_DATE 之前数据静态训练(含 W2V + 初始 top-K 写入检测器)
  流式循环: 每个 tumbling window:
    1) 统计窗口 hashtag 频次 -> TrendShiftDetector.observe
    2) 若切换: 在 logic window 上重训 W2V, 在 logic+FT window 上渐进微调 MLP
       (官方 process(): W2V 全量重训 + rt_training)
    3) 无论是否切换: 对当前窗口做预测 + 官方 global 扩展评测, 记录逐窗指标
偏离: socket/Storm 信号 -> 直接函数调用; 逐窗指标聚合 -> metrics.json。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import zh_w2v as emb
from common.zh_text import clean_line, hashtags_list, sentence_for_embedding
from hadapts_zh import constants as c
from hadapts_zh.trend_shift import TrendShiftDetector
from hashet_zh import evaluation_zh as ev
from hashet_zh import model_bert_zh as mz


def build_targets(records, w2v):
    """官方 prepare_model_inputs_and_targets 语义的窗口内版本。"""
    sentences, targets, ht_lists = [], [], []
    for r in records:
        tokens = clean_line(r["text"])
        hts = [t for t in tokens.split() if t.startswith("#")]
        h = emb.tweet_arith_embedding(w2v, " ".join(hts))
        if h is not None:
            targets.append(h)
            sentences.append(sentence_for_embedding(tokens))
            ht_lists.append(hts)
    return sentences, np.array(targets, dtype=np.float32), ht_lists


def train_w2v(records, mincount):
    sentences = [clean_line(r["text"]).split() for r in records]
    return emb.create_and_train_word2vec(sentences, mincount=mincount,
                                         save_path=c.W2V_SAVE_FILE_NAME)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="官方 schema JSONL（按 date 排序）")
    ap.add_argument("--split-date", type=float, required=True,
                    help="bootstrap/实时 切分时间戳（unix 秒）")
    ap.add_argument("--encoder", default=None, choices=["mock", "bert"])
    ap.add_argument("--max-windows", type=int, default=None,
                    help="冒烟测试: 只跑前 N 个窗口")
    ap.add_argument("--seed", type=int, default=None, help="随机种子")
    ap.add_argument("--out", default=None, help="结果目录覆盖（默认 results/）")
    ap.add_argument("--w2v-mincount", type=int, default=None,
                    help="W2V min_count 覆盖（官方 40 为百万级语料默认; 小语料需调低, 偏离 D9）")
    ap.add_argument("--mincount", type=int, default=None,
                    help="hashtag 入表阈值覆盖（官方 60, 同上）")
    args = ap.parse_args()
    if args.encoder:
        os.environ["ZH_ENCODER"] = args.encoder
    if args.seed is not None:
        c.SEED = args.seed
    if args.w2v_mincount is not None:
        c.W2V_MINCOUNT = args.w2v_mincount
    if args.mincount is not None:
        c.MINCOUNT = args.mincount
    from common.zh_config import seed_all
    seed_all(args.seed)
    if args.out:
        c.RESULTS_FOLDER = args.out
        os.makedirs(c.RESULTS_FOLDER, exist_ok=True)
        c.METRICS_JSON = os.path.join(c.RESULTS_FOLDER, "metrics.json")
        c.SHIFTS_JSON = os.path.join(c.RESULTS_FOLDER, "shifts.json")
        c.PREDICTIONS_JSONL = os.path.join(c.RESULTS_FOLDER, "predictions.jsonl")

    recs = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    recs.sort(key=lambda r: r["date"])
    print(f"> loaded {len(recs)} posts")
    open(c.PREDICTIONS_JSONL, "w").close()  # 重置逐窗口预测导出

    static = [r for r in recs if r["date"] < args.split_date]
    rt = [r for r in recs if r["date"] >= args.split_date]
    print(f"> bootstrap={len(static)} realtime={len(rt)}")

    # ---- bootstrap: 静态训练（官方 static_training + start_trending 写入）----
    print("> bootstrap: static training")
    w2v = train_w2v(static, c.W2V_MINCOUNT)
    s_tr, t_tr, _ = build_targets(static, w2v)
    n_val = max(1, int(len(s_tr) * c.PERC_TEST))
    if len(s_tr) > n_val + 1:
        tr_x, va_x, tr_y, va_y = s_tr[:-n_val], s_tr[-n_val:], t_tr[:-n_val], t_tr[-n_val:]
    else:
        tr_x, va_x, tr_y, va_y = s_tr, s_tr, t_tr, t_tr
    model = mz.HashetModel()
    model.transfer_and_fine_tune(tr_x, va_x, tr_y, va_y)

    from collections import Counter
    from common.zh_text import zh_hashtags_from_text
    boot_counts = Counter(h for r in static for h in zh_hashtags_from_text(r["text"]))
    detector = TrendShiftDetector(k=c.K, threshold=c.THRESHOLD,
                                  initial_counts=boot_counts)

    # ---- 流式循环 ----
    t0 = rt[0]["date"]
    t_end = rt[-1]["date"]
    win_len, logic_len, ft_len = (c.TUMBLING_WINDOW_LENGTH, c.LOGIC_WINDOW_LENGTH,
                                  c.FT_WINDOW_LENGTH)
    shift_log, window_metrics = [], []
    win_start, wi = t0, 0
    while win_start < t_end and (args.max_windows is None or wi < args.max_windows):
        win_end = win_start + win_len
        window = [r for r in rt if win_start <= r["date"] < win_end]
        if window:
            tags = [h for r in window for h in zh_hashtags_from_text(r["text"])]
            shifted, topk, jd = detector.observe(tags)
            print(f"> window {wi} [{win_start:.0f},{win_end:.0f}) n={len(window)} "
                  f"jd={jd:.3f} shifted={shifted} top3={topk[:3]}")
            if shifted:
                shift_log.append({"window": wi, "ts": win_end, "jd": round(jd, 4),
                                  "topk": topk})
                # 官方 process(): logic window 重训 W2V -> rt_training 渐进微调
                train_set = [r for r in rt if win_end - logic_len <= r["date"] < win_end]
                ft_set = [r for r in rt if win_end - ft_len <= r["date"] < win_end]
                if train_set and ft_set:
                    print(f"  retraining: logic={len(train_set)} ft={len(ft_set)}")
                    w2v = train_w2v(train_set, c.W2V_MINCOUNT)
                    s_all, t_all, _ = build_targets(train_set, w2v)
                    s_ft, t_ft, _ = build_targets(ft_set, w2v)
                    if len(s_all) and len(s_ft):
                        n_val = max(1, int(len(s_all) * c.PERC_TEST))
                        model.fine_tune_short(s_all[:-n_val], s_all[-n_val:],
                                              t_all[:-n_val], t_all[-n_val:],
                                              epochs=c.MAX_EPOCHS_FT)
            # 窗口内预测 + 评测（官方 do_test 分支）
            s_te, t_te, hts = build_targets(window, w2v)
            if len(s_te) >= 2:
                rounds = ev.evaluate(hts, s_te, model, max_iterations=1)
                window_metrics.append({"window": wi, "ts": win_end,
                                       "n_test": len(s_te),
                                       **rounds[0]["weighted"]})
                # DKNet 桥接: 导出窗口级 top-K 预测（保序）供 eval_bridge 使用
                preds = model.predict_top_k_hashtags(s_te, 10, w2v=w2v)
                with open(c.PREDICTIONS_JSONL, "a", encoding="utf-8") as pf:
                    for r, p in zip(window, preds):
                        pf.write(json.dumps({"ts": r["date"], "window": wi,
                                             "pred_tags": [t for t, _ in p]},
                                            ensure_ascii=False) + "\n")
        win_start = win_end
        wi += 1

    with open(c.METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump({"windows": window_metrics,
                   "n_shifts": len(shift_log)}, f, ensure_ascii=False, indent=1)
    with open(c.SHIFTS_JSON, "w", encoding="utf-8") as f:
        json.dump(shift_log, f, ensure_ascii=False, indent=1)
    print(f"> {len(window_metrics)} windows evaluated, {len(shift_log)} shifts "
          f"-> {c.METRICS_JSON}")


if __name__ == "__main__":
    main()
