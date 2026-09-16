# -*- coding: utf-8 -*-
"""HASHET dataset_handler.py 的中文等价实现。

保留官方函数语义与调用顺序（run.py 中的五步流程）:
  preprocess_data -> preprocess_data_for_sentence_embedding -> prepare_train_test
  -> prepare_model_inputs_and_targets
偏离点: prepare_train_test 改为按时间先后划分（官方为随机 shuffle, 偏离 D2）。
"""
import json
import os

import numpy as np

from common import zh_w2v as emb
from common import zh_config as cfg
from common.zh_text import clean_line, sentence_for_embedding, hashtags_list
from hashet_zh import constants as c


def store(lines, file):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def load(file):
    with open(file, encoding="utf-8") as f:
        return [l.rstrip("\n").split() for l in f if l.strip()]


def read_official_jsonl(input_file):
    """读官方 schema JSONL, 返回按 date 排序的记录列表。"""
    recs = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r["date"])
    return recs


def preprocess_data(input_file, save_file=None):
    """官方语义: 清洗语料并落盘, 供 Word2Vec 训练（话题以 '#话题' 形态保留）。"""
    recs = read_official_jsonl(input_file)
    lines = [clean_line(r["text"]) for r in recs if len(r["text"]) > 0]
    store(lines, save_file or c.W2V_INPUT)
    return lines


def preprocess_data_for_sentence_embedding(input_file, save_file=None):
    """官方语义: 同样清洗, 落盘到 TRAIN_TEST_INPUT（hashtag 仍在 token 中,
    后续由 remove_hashtags_from_sentences / sentence_for_embedding 决定是否剥离）。"""
    recs = read_official_jsonl(input_file)
    lines = [clean_line(r["text"]) for r in recs if len(r["text"]) > 0]
    store(lines, save_file or c.TRAIN_TEST_INPUT)
    return lines


def load_without_not_relevant_hts(input_file):
    """官方原样: 过滤出现次数 <= MINCOUNT 的 hashtag 行。"""
    counter = {}
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            for w in line.split():
                if "#" in w:
                    counter[w] = counter.get(w, 0) + 1
    result = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            res_line = [w for w in line.split()
                        if ("#" not in w or counter[w] > c.MINCOUNT)]
            if res_line:
                result.append(res_line)
    return result


def prepare_train_test(input_file=None, perc_test=None):
    """按时间先后划分 train/test（偏离 D2: 官方为 random.shuffle 后随机划分;
    合成基准需与 DKNet 的时序协议对齐, 避免时间泄漏）。"""
    corpus = load_without_not_relevant_hts(input_file or c.TRAIN_TEST_INPUT)
    # 保持文件内的时序顺序（preprocess 阶段已按 date 排序落盘）
    kept = []
    for tweet in corpus:
        has_ht = any("#" in w for w in tweet)
        has_w = any("#" not in w for w in tweet)
        if has_ht and has_w:
            kept.append(tweet)
    cut = int(len(kept) * (1 - (perc_test or c.PERC_TEST)))
    store([" ".join(t) for t in kept[:cut]], c.TRAIN_CORPUS)
    store([" ".join(t) for t in kept[cut:]], c.TEST_CORPUS)


def remove_hashtags_from_sentences(lines, targets):
    """官方 SKIP_HASHTAG_REMOVING=True 分支: '#话题' -> '话题'。"""
    out = [sentence_for_embedding(l) for l in lines]
    return out, targets


def prepare_model_inputs_and_targets(w2v):
    """官方语义不变: hashtag 向量均值作 target, 剥离 '#' 后的句子作输入。"""
    train = [" ".join(t) for t in load(c.TRAIN_CORPUS)]
    test = [" ".join(t) for t in load(c.TEST_CORPUS)]

    sentences_train, targets_train, ht_lists_train = [], [], []
    for tweet in train:
        ht_list = hashtags_list(tweet, w2v)
        h = emb.tweet_arith_embedding(w2v, " ".join(ht_list))
        if h is not None:
            targets_train.append(h)
            sentences_train.append(tweet)
            ht_lists_train.append(ht_list)

    sentences_test, targets_test, ht_lists = [], [], []
    for tweet in test:
        ht_list = hashtags_list(tweet, w2v)
        h = emb.tweet_arith_embedding(w2v, " ".join(ht_list))
        if h is not None:
            targets_test.append(h)
            sentences_test.append(tweet)
            ht_lists.append(ht_list)

    sentences_train, targets_train = remove_hashtags_from_sentences(sentences_train, targets_train)
    sentences_test, targets_test = remove_hashtags_from_sentences(sentences_test, targets_test)

    return (np.array(sentences_train), np.array(sentences_test),
            np.array(targets_train, dtype=np.float32),
            np.array(targets_test, dtype=np.float32), ht_lists)
