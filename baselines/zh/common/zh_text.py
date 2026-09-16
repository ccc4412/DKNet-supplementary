# -*- coding: utf-8 -*-
"""中文文本预处理：官方 HASHET/H-ADAPTS dataset_handler.py 中英文绑定组件的中文等价替换。

对应关系（官方 -> 本文件，偏离处见 zh/README.md 的偏离声明表）:
  spaCy en_core_web_lg cleaning()  -> clean_line()        jieba 分词 + 停用词 + 保留 #话题# 整体
  hashtags_list() 的 '#' 提取      -> zh_hashtags_from_text() / hashtags_list()
  re.sub(r'[^\x00-\x7f]', ' ')     -> KEEP_CJK 正则        官方规则会删除全部中文，必须替换

规范形态约定: 微博话题 #话题# 在词表/语料中统一存为 '#话题'（去掉尾部#），
使官方 word_embedding_model.retain_hts / similar_by_vector 等下游逻辑零改动。
"""
import re

import jieba

# 与官方 URL_REGEX 等价 + 微博转发链 //@昵称
ZH_URL_REGEX = r"//@[\w\u4e00-\u9fff-]+|@[\w\u4e00-\u9fff-]+|https?:?/?/?[\w./\-]*|www\.[\w./\-]*"
# 保留: CJK、拉丁字母数字、'#'、'_'；其余（emoji、全角符号等）替换为空格
KEEP_CJK = r"[^\w#\u4e00-\u9fff]+"
# 微博话题语法: #话题#（1~30 个非#非空白字符）
WEIBO_TOPIC = re.compile(r"#([^#\s]{1,30})#")

# 精简中文停用词表（哈工大停用词表子集，覆盖官方 REMOVE_WORDS 的角色）
ZH_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "这", "那", "啊", "吧", "吗", "呢", "呀", "哦", "嗯", "所以", "因为", "但是",
    "还", "被", "把", "让", "向", "往", "等", "比", "更", "最", "又", "再", "才",
    "rt", "ht", "htt", "https", "http",  # 官方 REMOVE_WORDS 原样保留
    "转发", "微博", "视频", "图片", "全文",
}


def normalize_hashtag_syntax(text: str) -> str:
    """微博 #话题# -> 规范形态 #话题。连续话题 #a##b# 亦被正确切分。"""
    return WEIBO_TOPIC.sub(lambda m: "#" + m.group(1), text)


def zh_hashtags_from_text(text: str) -> list:
    """从原始微博文本提取话题，返回规范形态 ['#话题', ...]（保序去重）。"""
    tags, seen = [], set()
    for m in WEIBO_TOPIC.finditer(text):
        tag = "#" + m.group(1)
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def _is_kept(token: str) -> bool:
    if not token or token in ZH_STOPWORDS:
        return False
    if token.startswith("#"):
        return len(token) > 1
    if len(token) < 2 and not token.isdigit():
        return False  # 对齐官方: 过滤单字 token（'#' token 分支除外）
    return True


def clean_line(text: str) -> str:
    """官方 cleaning() 的中文版：返回空格分隔 token 串，话题以 '#话题' 形态保留。

    官方对英文做 spaCy lemma；中文无形态变化，保留原词（偏离声明 D1）。
    实现: 先把 '#话题#' 暂存为原子占位符（防止 jieba 拆散），其余文本分词后回填。
    """
    text = re.sub(ZH_URL_REGEX, " ", text)
    atoms = []

    def _stash(m):
        atoms.append("#" + m.group(1))
        return f" HTZ{len(atoms)-1}Z "

    text = WEIBO_TOPIC.sub(_stash, text)
    text = re.sub(KEEP_CJK, " ", text)
    parts = []
    for piece in text.split():
        if len(piece) > 4 and piece.startswith("HTZ") and piece.endswith("Z") \
                and piece[3:-1].isdigit():
            parts.append(atoms[int(piece[3:-1])])
        else:
            parts.extend(jieba.lcut(piece))
    return " ".join(t for t in parts if _is_kept(t))


def sentence_for_embedding(tokens_line: str) -> str:
    """官方 remove_hashtags_from_sentences 的 SKIP_HASHTAG_REMOVING 分支:
    '#话题' -> '话题' 后拼接，作为句子编码器输入。"""
    return " ".join(t[1:] if t.startswith("#") else t for t in tokens_line.split())


def hashtags_list(tokens_line: str, w2v_model) -> list:
    """官方 hashtags_list 的中文版: 位于 W2V 词表中的 '#话题' token。"""
    return [t for t in tokens_line.split()
            if t.startswith("#") and t in w2v_model.wv.key_to_index]
