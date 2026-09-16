# -*- coding: utf-8 -*-
"""D1 数据适配层。

D1 原生 schema（与论文 Sec. Synthetic Evaluation Corpora 描述一致）:
  {"id": int, "timestamp": unix秒, "text": str, "hashtags": ["#话题#"], 
   "is_repost": bool, "topic": int, "entities": ["#实体#"]}

官方 schema（HASHET/H-ADAPTS 输入）:
  {"text": str, "date": unix秒, "isRetweet": bool, "hashtags": ["#话题"]}

generate_sample_d1(): 按论文 D1 描述生成小型样例语料（10 主题、脚本化漂移、
噪声注入），仅用于流水线冒烟测试与接口联调；真实 D1 到位后直接走
convert_d1_to_official()。
"""
import json
import os
import random

from common.zh_text import normalize_hashtag_syntax

# 10 个 D1 主题: (话题hashtag, 实体hashtag池, 核心词池, 口语变体池)
TOPICS = [
    ("#人工智能大模型#", ["#通用人工智能实验室#", "#智源研究院#"],
     ["大模型", "参数规模", "推理成本", "开源权重", "算力"],
     ["AI大模型", "顶流模型", "模型卷"]),
    ("#新能源汽车#", ["#蔚来汽车#", "#宁德时代#"],
     ["续航", "充电桩", "电池技术", "交付量", "智能驾驶"],
     ["电车", "充电焦虑", "续航虚标"]),
    ("#乡村振兴#", ["#农业农村部#", "#供销社#"],
     ["助农", "直播带货", "特色农产品", "基础设施", "惠农政策"],
     ["三农", "土特产", "家乡带货"]),
    ("#体育赛事#", ["#国家队#", "#市奥体中心#"],
     ["联赛", "夺冠", "淘汰赛", "青训", "主场"],
     ["上分", "卫冕", "绝杀"]),
    ("#房地产政策#", ["#住建部#", "#城市公积金中心#"],
     ["限购", "房贷利率", "保交楼", "刚需", "二手房"],
     ["上车", "接盘", "学区房"]),
    ("#出境旅游#", ["#国际航空公司#", "#边检站#"],
     ["签证", "航线恢复", "免税购物", "行程规划", "汇率"],
     ["特种兵旅游", "攻略", "打卡"]),
    ("#校园就业#", ["#高校毕业生就业中心#", "#人力资源保障局#"],
     ["秋招", "简历", "校招", "offer", "实习"],
     ["上岸", "海投", "卷不动"]),
    ("#医疗健康#", ["#医科大学附属医院#", "#疾控中心#"],
     ["门诊", "流感疫苗", "分级诊疗", "医保", "慢病管理"],
     ["看病难", "排队", "挂号"]),
    ("#影视综艺#", ["#影视集团#", "#视频平台#"],
     ["票房", "收视", "剧集", "综艺", "口碑"],
     ["爆了", "下饭", "追剧"]),
    ("#气象预警#", ["#气象台#", "#应急管理局#"],
     ["暴雨", "台风路径", "高温预警", "防汛", "出行安全"],
     ["桑拿天", "泼水", "降温"]),
]
GENERIC = ["今天", "网友", "讨论", "消息", "报道", "关注", "表示", "官方", "网友热议"]
EMOJIS = ["🔥", "😂", "😱", "👍", "🤔", "💯", "🙏", "😭"]
SLANGS = ["yyds", "绝绝子", "无语住了", "拴Q", "泰裤辣", "破防了"]


def _topic_weight(day, drift_days, rng, base):
    """主题热度分布; drift_days 处发生脚本化 topic-center 迁移(概念漂移)。"""
    w = list(base)
    for i, d in enumerate(drift_days):
        if day >= d:
            j = (i * 3 + 1) % len(w)          # 确定性迁移: 热度在主题间轮转
            k = (i * 7 + 5) % len(w)
            w[j], w[k] = w[k] * 3, w[j]
            w[k - 1] = w[k - 1] * 2 + 1
    return w


def generate_sample_d1(out_jsonl, n_posts=6000, days=90, drift_days=(45, 70), seed=0):
    """生成小型 D1 样例: 10 主题、2 次脚本化漂移、emoji/口语/错字噪声、突发节奏。"""
    rng = random.Random(seed)
    t0 = 1700000000  # 固定起点, 保证可复现
    base = [10.0] * len(TOPICS)
    recs, rid = [], 0
    for day in range(days):
        # 突发节奏: 漂移日附近出现 burst(Fano 系数的简化对应物)
        lam = 40 + 160 * sum(day == d or day == d + 1 for d in drift_days)
        n_today = max(5, int(rng.gauss(lam, lam * 0.3)))
        weights = _topic_weight(day, drift_days, rng, base)
        for _ in range(n_today):
            topic = rng.choices(range(len(TOPICS)), weights=weights)[0]
            tag, ent_pool, vocab, slang_pool = TOPICS[topic]
            words = [rng.choice(GENERIC) for _ in range(rng.randint(1, 3))]
            words += [rng.choice(vocab) for _ in range(rng.randint(2, 4))]
            if rng.random() < 0.35:
                words.append(rng.choice(slang_pool))
            if rng.random() < 0.45:
                words.append(rng.choice(EMOJIS))
            text = "".join(words)
            entities = [rng.choice(ent_pool)] if rng.random() < 0.30 else []
            # 话题/实体以微博 #话题# 语法嵌入正文（72% 帖子带主话题, 实体低频出现）
            in_text_tags = [tag] if rng.random() < 0.72 else []
            in_text_tags += entities
            if in_text_tags:
                pos = rng.randint(0, len(text))
                text = text[:pos] + "".join(t + "#" for t in in_text_tags) + text[pos:]
            hashtags = [t for t in in_text_tags if t == tag or rng.random() < 0.9]
            is_repost = rng.random() < 0.20
            recs.append({"id": rid, "timestamp": t0 + day * 86400 + rng.randrange(86400),
                         "text": text, "hashtags": hashtags, "is_repost": is_repost,
                         "topic": topic, "entities": entities})
            rid += 1
            if rid >= n_posts:
                break
        if rid >= n_posts:
            break
    recs.sort(key=lambda r: r["timestamp"])
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(recs), drift_days


def convert_d1_to_official(d1_jsonl, out_jsonl, no_repost=False):
    """D1 原生 -> 官方 schema。话题语法规范化为 '#话题'（去尾部#）。"""
    n = 0
    with open(d1_jsonl, encoding="utf-8") as fin, open(out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            if no_repost and r.get("is_repost"):
                continue
            tags = [normalize_hashtag_syntax(h) for h in r.get("hashtags", [])]
            fout.write(json.dumps({"text": r["text"], "date": r["timestamp"],
                                   "isRetweet": bool(r.get("is_repost")), "hashtags": tags},
                                  ensure_ascii=False) + "\n")
            n += 1
    return n


def convert_d1_window_to_official(d1_records):
    """流式版本（H-ADAPTS 滚动窗口用）: 记录列表 -> 官方 schema 记录列表。"""
    out = []
    for r in d1_records:
        out.append({"text": r["text"], "date": r["timestamp"],
                    "isRetweet": bool(r.get("is_repost")),
                    "hashtags": [normalize_hashtag_syntax(h) for h in r.get("hashtags", [])]})
    return out


# ---------------------------------------------------------------------------
# 真实 D1 适配（data/generate_datasets.py 产出）
#
# 原生 schema: {"text", "keywords": [裸词], "timestamp": ISO8601, "user_id_hash",
#               "topic", "domain", "engagement"}
# 论文协议（Table 2: Annotation = Hashtag-based）: 关键词以微博 #词# 语法嵌入正文,
# 对所有方法一致。原生模板自带的 #话题# 原样保留。
# 漂移语义: 第 20/36 周后, 漂移主题的记录按 30% 概率获得新词（generate_datasets.py
# 的 apply_concept_drift_d1 已在 keywords 字段完成注入）。
# ---------------------------------------------------------------------------
import json as _json
from datetime import datetime as _dt


def convert_real_d1_to_official(d1_json, out_jsonl):
    """data/datasets/d1_weibo_temporal.json -> 官方 schema JSONL。"""
    n = 0
    with open(d1_json, encoding="utf-8") as fin, open(out_jsonl, "w", encoding="utf-8") as fout:
        recs = _json.load(fin)
        for r in recs:
            ts = int(_dt.fromisoformat(r["timestamp"]).timestamp())
            text = r["text"]
            # 关键词 -> 正文内嵌 #词#（幂等: 已在文中出现的不再重复）
            embedded = []
            for kw in r.get("keywords", []):
                tag = f"#{kw}#"
                if tag not in text and f"#{kw}" not in text:
                    embedded.append(tag)
            if embedded:
                text = text + " " + " ".join(embedded)
            fout.write(_json.dumps({"text": text, "date": ts, "isRetweet": False,
                                    "hashtags": []}, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_real_d1_groundtruth(d1_json):
    """读取真实 D1 原生记录（含 keywords/topic/timestamp），供 eval_bridge
    构造逐窗口 ground truth。返回按时间排序的记录列表（timestamp 转 unix 秒,
    keywords 保留裸词, 另存 topic）。"""
    recs = _json.load(open(d1_json, encoding="utf-8"))
    for r in recs:
        r["ts"] = int(_dt.fromisoformat(r["timestamp"]).timestamp())
    recs.sort(key=lambda r: r["ts"])
    return recs
