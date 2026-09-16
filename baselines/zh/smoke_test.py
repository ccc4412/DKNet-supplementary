# -*- coding: utf-8 -*-
"""端到端冒烟测试: 生成小型 D1 样例 -> 转官方 schema -> HASHET-zh(mock) ->
H-ADAPTS-zh(mock, 前6窗口) -> 趋势切换检测单测。全程 mock 编码器, 不需要权重下载。
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common.d1_adapter import generate_sample_d1, convert_d1_to_official
from hadapts_zh.trend_shift import (TrendShiftDetector, top_k,
                                    jaccard_distance_weighted)


def test_trend_shift_math():
    """对照 CounterBolt 语义的手算用例。"""
    # 相同 top-K 顺序 -> jd = 0, 无切换
    old = ["#a", "#b", "#c"]
    assert abs(jaccard_distance_weighted(old, old[:], 3)) < 1e-9
    # 完全不同的 top-K -> 交为 0, jd = 1 >= 0.9, 切换
    assert jaccard_distance_weighted(old, ["#x", "#y", "#z"], 3) == 1.0
    # 手算: k=3, old=[a,b,c], new=[c,a,b]: a: min(3-0, 3-1)=2; b: min(3-1,3-2)=1;
    # c: min(3-2,3-0)=1; inter=4, union=6, jd=1-4/6=1/3
    assert abs(jaccard_distance_weighted(old, ["#c", "#a", "#b"], 3) - 1 / 3) < 1e-9
    d = TrendShiftDetector(k=2, threshold=0.9)
    s1, topk, jd = d.observe(["#a"] * 5 + ["#b"] * 3)
    assert s1 and jd == 1.0 and topk == ["#a", "#b"]
    s2, _, jd2 = d.observe(["#a"] * 9 + ["#b"] * 1)
    assert not s2 and jd2 < 0.9, f"jd2={jd2}"
    s3, _, jd3 = d.observe(["#zzz"] * 4 + ["#yyy"] * 2)
    assert s3 and jd3 == 1.0
    print("[OK] trend shift math")


def main():
    test_trend_shift_math()
    tmp = tempfile.mkdtemp(prefix="zh_smoke_")
    d1 = os.path.join(tmp, "d1.jsonl")
    official = os.path.join(tmp, "official.jsonl")
    n, drift = generate_sample_d1(d1, n_posts=4000, days=60, drift_days=(30, 45), seed=1)
    print(f"[OK] sample D1 generated: {n} posts, drift days {drift}")
    m = convert_d1_to_official(d1, official)
    print(f"[OK] converted to official schema: {m} records")

    env = dict(os.environ, ZH_ENCODER="mock", PYTHONIOENCODING="utf-8")
    # HASHET-zh 冒烟（限制轮数以便快速结束）
    from hashet_zh import constants as hc
    hc.MAX_EPOCHS = 3
    r = subprocess.run([sys.executable, os.path.join(HERE, "hashet_zh", "run.py"),
                        official, "--encoder", "mock"], env=env,
                       capture_output=True, text=True, encoding="utf-8")
    tail = "\n".join(r.stdout.strip().splitlines()[-8:])
    print(f"[HASHET-zh] rc={r.returncode}\n{tail}")
    if r.returncode != 0:
        print(r.stderr[-2000:]); sys.exit(1)
    assert os.path.exists(hc.METRICS_JSON)
    from eval_bridge import hashet_metrics_to_dknet
    print("[OK] HASHET-zh metrics:", hashet_metrics_to_dknet(hc.METRICS_JSON))

    # H-ADAPTS-zh 冒烟（前 6 个窗口）
    recs = [json.loads(l) for l in open(official, encoding="utf-8") if l.strip()]
    split = recs[len(recs) // 2]["date"]
    from hadapts_zh import constants as ac
    r2 = subprocess.run([sys.executable, os.path.join(HERE, "hadapts_zh", "run.py"),
                         official, "--split-date", str(split), "--encoder", "mock",
                         "--max-windows", "6"], env=env,
                        capture_output=True, text=True, encoding="utf-8")
    tail2 = "\n".join(r2.stdout.strip().splitlines()[-10:])
    print(f"[H-ADAPTS-zh] rc={r2.returncode}\n{tail2}")
    if r2.returncode != 0:
        print(r2.stderr[-2000:]); sys.exit(1)

    # ---- eval_bridge: H-ADAPTS 逐窗口预测 -> DKNet 主表四指标 ----
    # 桥接协议: 窗口级关键词集 = 该窗口各帖预测去重后的前 10 个, 保序
    from eval_bridge import window_predictions_to_metrics, load_d1_native
    from collections import OrderedDict
    d1_recs = load_d1_native(d1)
    preds_by_win, recs_by_win = OrderedDict(), OrderedDict()
    for l in open(ac.PREDICTIONS_JSONL, encoding="utf-8"):
        p = json.loads(l)
        preds_by_win.setdefault(p["window"], [])
        for t in p["pred_tags"]:
            if t not in [x[0] for x in preds_by_win[p["window"]]]:
                preds_by_win[p["window"]].append((t, 0.0))
    t0 = d1_recs[0]["timestamp"]
    for r in d1_recs:
        wi = int((r["timestamp"] - t0) // 86400)
        recs_by_win.setdefault(wi, []).append(r)
    m = window_predictions_to_metrics(preds_by_win, recs_by_win, topk=10,
                                      window_minutes=1440)
    print("[OK] eval_bridge DKNet-style metrics:", json.dumps(m))
    print("[SMOKE TEST PASSED]")


if __name__ == "__main__":
    main()
