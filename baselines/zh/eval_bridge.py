# -*- coding: utf-8 -*-
"""评测桥接：HASHET-zh / H-ADAPTS-zh 的逐窗口输出 -> DKNet 主表指标。

DKNet 协议（论文 Experiments 节）:
  P@10        : 每窗口输出 top-10 关键词与相关性标注比对
  Topic Coverage : 窗口内预测关键词映射到 ground-truth 主题的比例
  Entity Recall  : 预测关键词(去'#')与实体标注匹配比例
  TtC (min)      : 主题出现 -> 首次被该 baseline top-K 捕获的延迟(窗口数x窗口分钟)

hashtag -> 关键词/主题映射规则（对两个 baseline 与 DKNet 一致）:
  关键词 = hashtag 去掉 '#';
  主题命中 = 该关键词出现在该窗口 ground-truth 主题的关键词/话题集合中。
"""
import json
from collections import defaultdict


def load_d1_native(d1_jsonl):
    """读 D1 原生 schema, 返回逐窗口 ground truth 与记录。"""
    recs = [json.loads(l) for l in open(d1_jsonl, encoding="utf-8") if l.strip()]
    recs.sort(key=lambda r: r["timestamp"])
    return recs


def window_predictions_to_metrics(pred_tags_by_window, d1_rec_by_window,
                                  topk=10, window_minutes=None, eval_from=None):
    """pred_tags_by_window: {win_idx: [('#话题', score), ...]}（模型输出保序）
    d1_rec_by_window:      {win_idx: [记录, ...]}，记录需含
                           topic / hashtags(['#词']) / entities(可选) 字段
                           （真实 D1 由 build_real_windows() 构造）
    eval_from: 评测起始窗口号（基线开始输出的时刻）; TtC 只统计在该时刻之后
               新出现（emerge >= eval_from）的主题, 避免把 bootstrap 前已存在的
               老主题的等待时间计入延迟。None = 全部窗口。
    返回 DKNet 风格四指标（P@10 / Coverage / EntityRecall / TtC）。"""
    if eval_from is None:
        eval_from = min(pred_tags_by_window) if pred_tags_by_window else 0
    prec_list, cov_list, er_list = [], [], []
    first_capture = {}   # topic -> 延迟(窗口数)
    topic_emerge_win = {}
    for wi in sorted(d1_rec_by_window):
        recs = d1_rec_by_window[wi]
        for r in recs:
            topic_emerge_win.setdefault(r["topic"], wi)
        if wi < eval_from or wi not in pred_tags_by_window:
            continue  # 评测窗口之外（bootstrap 期 / 基线未输出）不进入平均
        gold_kw = set()
        gold_ent = set()
        topic_kw = defaultdict(set)
        for r in recs:
            for h in r.get("hashtags", []):
                kw = h.strip("#")
                gold_kw.add(kw)
                topic_kw[r["topic"]].add(kw)
            for e in r.get("entities", []):
                gold_ent.add(e.strip("#"))
        preds = [t for t, _ in (pred_tags_by_window.get(wi) or [])][:topk]
        kws = [p.strip("#") for p in preds]
        if not kws:
            prec_list.append(0.0)
            cov_list.append(0.0)
            er_list.append(0.0)
            continue
        prec_list.append(sum(1 for k in kws if k in gold_kw) / len(kws))
        # 主题命中 = 预测关键词落在该主题自己的 gold 关键词集合内
        kw_set = set(kws)
        cov_topics = {t for t, kws_t in topic_kw.items() if kws_t & kw_set}
        cov_list.append(len(cov_topics) / max(1, len(topic_kw)))
        er_list.append(sum(1 for k in kws if k in gold_ent) / max(1, len(gold_ent))
                       if gold_ent else 0.0)
        for t in cov_topics:
            if topic_emerge_win.get(t, wi) >= eval_from:
                first_capture.setdefault(t, wi - topic_emerge_win[t])
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    ttc_windows = avg(list(first_capture.values())) if first_capture else 0.0
    ttc_min = ttc_windows * window_minutes if window_minutes else None
    return {"P@10": round(avg(prec_list), 3),
            "coverage": round(avg(cov_list), 3),
            "entity_recall": round(avg(er_list), 3),
            "ttc_windows": round(ttc_windows, 2),
            "ttc_min": round(ttc_min, 1) if ttc_min is not None else None,
            "n_windows": len(prec_list),
            "n_topics_ttc": len(first_capture)}


def build_real_windows(real_recs, window_seconds=86400):
    """真实 D1 原生记录 -> 逐窗口桥接记录（keywords 裸词 -> '#词' 形态）。"""
    t0 = real_recs[0]["ts"]
    by_win = defaultdict(list)
    for r in real_recs:
        wi = int((r["ts"] - t0) // window_seconds)
        by_win[wi].append({"topic": r["topic"],
                           "hashtags": ["#" + k for k in r.get("keywords", [])],
                           "entities": [], "ts": r["ts"]})
    return by_win


def drift_keyword_ttc(pred_tags_by_window, real_recs, drift_keywords,
                      window_seconds=86400, topk=10, window_minutes=None):
    """漂移关键词级 TtC: 每个漂移词从首次进入 gold 标注（注入时刻）到首次被
    top-K 预测捕获的窗口延迟。这是合成 D1 上与论文 TtC 定义对应的可观测量
    （主题级涌现均在 bootstrap 之前, 主题级 TtC 不可辨识）。"""
    t0 = real_recs[0]["ts"]
    gold_win, pred_win = {}, {}
    for r in real_recs:
        wi = int((r["ts"] - t0) // window_seconds)
        for kw in drift_keywords:
            if kw in r.get("keywords", []) and kw not in gold_win:
                gold_win[kw] = wi
    for wi in sorted(pred_tags_by_window):
        kws = {t.strip("#") for t, _ in pred_tags_by_window[wi][:topk]}
        for kw in kws & set(drift_keywords):
            pred_win.setdefault(kw, wi)
    rows, delays = [], []
    for kw in sorted(drift_keywords):
        g, p = gold_win.get(kw), pred_win.get(kw)
        d = (p - g) if (g is not None and p is not None and p >= g) else None
        rows.append({"keyword": kw, "gold_win": g, "pred_win": p, "delay_win": d})
        if d is not None:
            delays.append(d)
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {"per_keyword": rows,
            "captured": f"{len(delays)}/{len(drift_keywords)}",
            "ttc_windows": round(avg(delays), 2) if delays else None,
            "ttc_min": round(avg(delays) * window_minutes, 1)
            if delays and window_minutes else None}


def hashet_metrics_to_dknet(metrics_json):
    """直读 hashet_zh metrics.json 的官方指标（加权 P/R/F1@n=0 档）, 供对照。"""
    d = json.load(open(metrics_json, encoding="utf-8"))
    r0 = d["rounds"][0]["weighted"]
    return {"official_weighted_precision": r0["precision"],
            "official_weighted_recall": r0["recall"],
            "official_weighted_f1": r0["f1"],
            "n_test": d.get("n_test")}
