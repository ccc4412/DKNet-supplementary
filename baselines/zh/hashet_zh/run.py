# -*- coding: utf-8 -*-
"""HASHET-zh 主流程（官方 run.py 的中文适配版）。

用法:
  python hashet_zh/run.py <official_schema.jsonl> [--encoder mock|bert]

流程与官方一致: W2V 语料预处理 -> W2V 训练 -> 句子预处理 -> 时序 train/test 划分
-> 构造 (句子, hashtag 均值向量) -> 两阶段训练 -> top-50 预测 -> global 扩展评测
-> metrics.json + results.txt 落盘。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import zh_w2v as emb
from common.zh_config import seed_all
from hashet_zh import constants as c
from hashet_zh import dataset_handler_zh as dh
from hashet_zh import evaluation_zh as ev
from hashet_zh import model_bert_zh as mz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="官方 schema 的 JSONL 语料")
    ap.add_argument("--encoder", default=None, choices=["mock", "bert"])
    ap.add_argument("--seed", type=int, default=None, help="随机种子（默认 constants.SEED）")
    ap.add_argument("--out", default=None, help="输出目录覆盖（默认 save_files/）")
    args = ap.parse_args()
    if args.encoder:
        os.environ["ZH_ENCODER"] = args.encoder
        c.ENCODER_BACKEND = args.encoder
    if args.seed is not None:
        c.SEED = args.seed
    seed_all(args.seed)
    if args.out:
        c.SAVE_FOLDER = args.out
        for k in ("W2V_INPUT", "W2V_SAVE_FILE_NAME", "TRAIN_TEST_INPUT",
                  "TRAIN_CORPUS", "TEST_CORPUS", "TL_WEIGHTS", "FT_WEIGHTS",
                  "RESULTS_TXT", "METRICS_JSON"):
            setattr(c, k, os.path.join(c.SAVE_FOLDER, os.path.basename(getattr(c, k))))
    os.makedirs(c.SAVE_FOLDER, exist_ok=True)

    print("> Preprocessing data for Word2Vec model")
    dh.preprocess_data(args.input)

    print("> Training Word2Vec model")
    sentences = dh.load(c.W2V_INPUT)
    emb.create_and_train_word2vec(sentences, mincount=c.W2V_MINCOUNT,
                                  save_path=c.W2V_SAVE_FILE_NAME)
    print("W2V MODEL TRAINING SUCCEEDED")

    print("> Preprocessing data for sentence embeddings")
    dh.preprocess_data_for_sentence_embedding(args.input)

    print("> Preparing chronological train/test split")
    dh.prepare_train_test()

    print("> Loading Word2Vec model for calculating targets")
    w2v = emb.load_word2vec(c.W2V_SAVE_FILE_NAME)

    print("> Preparing data for neural network training")
    s_tr, s_te, t_tr, t_te, ht_lists = dh.prepare_model_inputs_and_targets(w2v)
    print(f"  train={len(s_tr)} test={len(s_te)}")

    print("> Training MLP model (transfer + fine-tune)")
    model = mz.HashetModel()
    model.transfer_and_fine_tune(s_tr, s_te, t_tr, t_te)

    print("> Predicting and evaluating")
    rounds = ev.evaluate(ht_lists, s_te, model)

    with open(c.METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump({"expansion": c.EXPANSION_STRATEGY, "rounds": rounds,
                   "n_test": len(s_te)}, f, ensure_ascii=False, indent=1)
    print(f"> metrics -> {c.METRICS_JSON}")


if __name__ == "__main__":
    main()
