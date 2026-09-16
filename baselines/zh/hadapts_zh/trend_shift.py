# -*- coding: utf-8 -*-
"""H-ADAPTS 趋势切换检测器：官方 CounterBolt.java 的 Python 忠实复刻。

官方逻辑（baselines/H-ADAPTS/src/main/java/bolt/CounterBolt.java:58-88）:
  1. 每个 tumbling window 内统计 hashtag 频次, 取 top-K 为 newTrending
  2. 秩加权交集: intersection = Σ_i min(K-i, K-rank_new(oldTrending[i]))
  3. union = K(K+1)/2; jd = 1 - intersection/union
  4. jd >= THRESHOLD -> 趋势切换, oldTrending <- newTrending, train=True
偏离声明 D8: Storm(Java) 流式外壳 -> 单进程同步循环; 窗口/K/阈值逐参数一致。
"""
from collections import Counter


def top_k(counts, k):
    """官方 topK(): 按计数降序取前 k 个 key。"""
    return [key for key, _ in
            sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def jaccard_distance_weighted(old_topk, new_topk, k):
    """官方秩加权 Jaccard 距离（未出现在 new 中的 old 标签贡献为 0）。"""
    intersection = 0.0
    for i, h in enumerate(old_topk):
        if h in new_topk:
            intersection += min(k - i, k - new_topk.index(h))
    union = k * (k + 1) / 2.0
    return 1 - intersection / union


class TrendShiftDetector:
    """状态机: 喂入每个窗口的 hashtag 频次, 输出 (是否切换, new_topk, jd)。

    初始 old_topk 用 bootstrap 段的 top-K（对齐官方 run.py 把
    start_trending 写入 java_constants.txt 的引导逻辑）。
    """

    def __init__(self, k=10, threshold=0.9, initial_counts=None):
        self.k = k
        self.threshold = threshold
        self.old_topk = top_k(initial_counts or {}, k) if initial_counts else []

    def observe(self, window_hashtags):
        counts = Counter(window_hashtags)
        new_topk = top_k(counts, self.k)
        jd = jaccard_distance_weighted(self.old_topk, new_topk, self.k) \
            if self.old_topk else 1.0
        shifted = jd >= self.threshold
        if shifted:
            self.old_topk = new_topk
        return shifted, new_topk, jd
