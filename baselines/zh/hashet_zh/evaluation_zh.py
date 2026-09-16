# -*- coding: utf-8 -*-
"""官方 evaluation.py 的等价移植（precision/recall/F1 @ k=1..5, global/local 扩展）。

数值语义逐行对齐官方 compute_scores; 仅新增 dump_metrics() 把结果写为 JSON,
供 eval_bridge 汇总进 DKNet 主表协议。
"""
import numpy as np

from common import zh_w2v as emb
from hashet_zh import constants as c


def recall_score(predicted, original):
    count = sum(1 for p in predicted for o in original if p[0] == o)
    return count / len(original)


def global_nhe(hashtags_test, predicted_list, n):
    """官方 model.global_nhe: 截断到 k+n 个预测。"""
    return [pred[:len(real) + n]
            for real, pred in zip(hashtags_test, predicted_list)]


def local_nhe(hashtags_test, predicted_list, n):
    """官方 model.local_nhe: 对前 k 个预测各取 n 个 W2V 近邻做语义扩展。"""
    emb_model = emb.load_word2vec(c.W2V_SAVE_FILE_NAME)
    result = []
    for real, pred in zip(hashtags_test, predicted_list):
        k = len(real)
        tuples = pred[:k]
        seen = [t[0] for t in tuples]
        if n > 0:
            for ph, _ in tuples:
                try:
                    nns = emb.retain_hts(emb_model.wv.most_similar(ph, topn=1000))[:n]
                except KeyError:
                    continue
                for nn in nns:
                    if nn[0] not in seen:
                        tuples.append(nn)
                        seen.append(nn[0])
        result.append(tuples)
    return result


def compute_scores(hashtags_test, predicted_list, count=5):
    """官方原样: 按每帖真实 hashtag 数 k 分桶(>5 归入桶5), 报告加权 P/R/F1。"""
    n_analyzed = [0] * 6
    rec_sum = [0.0] * 6
    prec_sum = [0.0] * 6
    for real, pred in zip(hashtags_test, predicted_list):
        k = len(real)
        bucket = count if k > count else k - 1
        n_analyzed[bucket] += 1
        r = recall_score(pred, real) if k > 0 else 0.0
        rec_sum[bucket] += r
        prec_sum[bucket] += r * len(real) / len(pred) if pred else 0.0

    per_k = {"n": n_analyzed[:5], "recall": [], "precision": [], "f1": []}
    eps = 1e-14
    for i in range(6):
        d = n_analyzed[i] if n_analyzed[i] else eps
        r = rec_sum[i] / d
        p = prec_sum[i] / d
        fden = r + p if (r + p) > 0 else eps
        if i < 5:
            per_k["recall"].append(round(r, 3))
            per_k["precision"].append(round(p, 3))
            per_k["f1"].append(round(2 * r * p / fden, 3))
    total = sum(n_analyzed[:5])
    w = lambda vals: float(np.sum([v * n for v, n in zip(vals, n_analyzed[:5])]) / total) \
        if total else 0.0
    return {"per_k": per_k,
            "weighted": {"recall": round(w(per_k["recall"]), 3),
                         "precision": round(w(per_k["precision"]), 3),
                         "f1": round(w(per_k["f1"]), 3)}}


def evaluate(hashtags_test, sentences_test, model, max_iterations=None):
    """官方 run.py 的评测入口: 预测 top-50 后逐扩展档位报告。"""
    max_iterations = max_iterations or c.MAX_EXPANSION_ITERATIONS
    predicted = model.predict_top_k_hashtags(sentences_test, 50)
    all_rounds = []
    for n in range(max_iterations):
        if c.EXPANSION_STRATEGY == "local":
            bounded = local_nhe(hashtags_test, predicted, n)
        else:
            bounded = global_nhe(hashtags_test, predicted, n)
        scores = compute_scores(hashtags_test, bounded)
        all_rounds.append({"n": n, **scores})
        print(f"n={n} weighted P={scores['weighted']['precision']} "
              f"R={scores['weighted']['recall']} F1={scores['weighted']['f1']}")
    return all_rounds
