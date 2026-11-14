# -*- coding: utf-8 -*-
"""
快速实验验证脚本
使用 --quick 模式生成的数据验证所有基线方法
"""
import os
import sys
import json
import csv
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "version2"))

def compute_precision_at_k(predictions, ground_truth, k=10):
    """计算 P@K"""
    if not predictions or not ground_truth:
        return 0.0
    pred_set = set(predictions[:k])
    gt_set = set(ground_truth)
    return len(pred_set & gt_set) / min(k, len(pred_set))

def compute_recall_at_k(predictions, ground_truth, k=10):
    """计算 R@K"""
    if not predictions or not ground_truth:
        return 0.0
    pred_set = set(predictions[:k])
    gt_set = set(ground_truth)
    if not gt_set:
        return 0.0
    return len(pred_set & gt_set) / len(gt_set)

def compute_f1_at_k(predictions, ground_truth, k=10):
    """计算 F1@K"""
    p = compute_precision_at_k(predictions, ground_truth, k)
    r = compute_recall_at_k(predictions, ground_truth, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def compute_diversity(predictions, model=None):
    """计算多样性 (基于唯一词比例)"""
    if not predictions:
        return 0.0
    return len(set(predictions)) / len(predictions)

def compute_coverage(all_predictions, all_ground_truths):
    """计算主题覆盖率"""
    all_gt_keywords = set()
    for gt in all_ground_truths:
        all_gt_keywords.update(gt)
    all_pred_keywords = set()
    for pred in all_predictions:
        all_pred_keywords.update(pred)
    if not all_gt_keywords:
        return 0.0
    return len(all_pred_keywords & all_gt_keywords) / len(all_gt_keywords)

def load_csv_data(csv_path):
    """加载 CSV 数据"""
    texts = []
    keywords_list = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
            kw_str = row.get('keywords', '')
            keywords_list.append([k.strip() for k in kw_str.split('|') if k.strip()])
    return texts, keywords_list

def run_baseline_experiment(method_name, method, texts, ground_truths, top_k=10):
    """运行单个基线方法的实验"""
    print(f"\n{'='*60}")
    print(f"运行基线方法: {method_name}")
    print(f"{'='*60}")

    # 如果方法需要训练，先训练
    if hasattr(method, 'fit_corpus'):
        print(f"  训练 {method_name} (fit_corpus)...")
        t0 = time.time()
        method.fit_corpus(texts)
        print(f"  训练完成 ({time.time()-t0:.1f}s)")

    # 提取关键词
    all_predictions = []
    t0 = time.time()
    for i, text in enumerate(texts):
        try:
            keywords = method.extract_keywords(text, top_k)
            predictions = [word for word, _ in keywords]
            all_predictions.append(predictions)
        except Exception as e:
            print(f"  [警告] 样本 {i} 提取失败: {e}")
            all_predictions.append([])
    elapsed = time.time() - t0
    print(f"  提取完成 ({elapsed:.1f}s, {len(texts)/elapsed:.1f} docs/s)")

    # 计算指标
    precisions = [compute_precision_at_k(pred, gt, top_k) for pred, gt in zip(all_predictions, ground_truths)]
    recalls = [compute_recall_at_k(pred, gt, top_k) for pred, gt in zip(all_predictions, ground_truths)]
    f1s = [compute_f1_at_k(pred, gt, top_k) for pred, gt in zip(all_predictions, ground_truths)]
    diversities = [compute_diversity(pred) for pred in all_predictions]

    results = {
        'method': method_name,
        'precision_at_10': np.mean(precisions),
        'recall_at_10': np.mean(recalls),
        'f1_at_10': np.mean(f1s),
        'diversity': np.mean(diversities),
        'coverage': compute_coverage(all_predictions, ground_truths),
        'elapsed_seconds': elapsed,
        'num_samples': len(texts)
    }

    print(f"  P@10: {results['precision_at_10']:.4f}")
    print(f"  R@10: {results['recall_at_10']:.4f}")
    print(f"  F1@10: {results['f1_at_10']:.4f}")
    print(f"  Diversity: {results['diversity']:.4f}")
    print(f"  Coverage: {results['coverage']:.4f}")

    return results

def main():
    print("=" * 70)
    print("DKNet 基线方法快速验证实验")
    print("=" * 70)

    # 加载数据
    data_dir = project_root / "version2" / "data" / "datasets"
    d1_csv = data_dir / "d1_weibo_temporal.csv"
    d2_csv = data_dir / "d2_news_portal.csv"

    if not d1_csv.exists():
        print(f"[错误] 数据文件不存在: {d1_csv}")
        print("请先运行: python generate_datasets.py --quick")
        return

    # 加载 D1
    print(f"\n加载 D1 微博数据: {d1_csv}")
    d1_texts, d1_keywords = load_csv_data(str(d1_csv))
    print(f"  样本数: {len(d1_texts)}")

    # 加载 D2
    d2_texts, d2_keywords = [], []
    if d2_csv.exists():
        print(f"加载 D2 新闻数据: {d2_csv}")
        d2_texts, d2_keywords = load_csv_data(str(d2_csv))
        print(f"  样本数: {len(d2_texts)}")

    # 导入基线方法
    print("\n导入基线方法模块...")
    from src.baseline_methods import (
        TFIDFMethod, TextRankMethod, PositionRankMethod, RAKEMethod,
        YAKEMethod, KeyBERTMethod, SIFRankMethod, LDAMethod,
        BERTopicMethod, DTMMethod, PromptLLMMethod,
        OnlyKGMethod, StaticKGMethod
    )
    print("  所有基线方法导入成功")

    # 配置
    config = {
        'model': {
            'llm': {
                'model_name': 'bert-base-chinese',
                'max_length': 128,
                'device': 'cuda'
            }
        },
        'baseline': {
            'use_tfidf': True,
            'use_textrank': True,
            'use_positionrank': True,
            'use_rake': True,
            'use_yake': True,
            'use_keybert': True,
            'use_sifrank': True,
            'use_lda': True,
            'use_bertopic': True,
            'use_dtm': True,
            'use_prompt_llm': True,
            'use_only_kg': True,
            'use_static_kg': True,
        }
    }

    # 定义基线方法列表
    baseline_methods = [
        ("Static Keywords", None),  # 特殊处理
        ("TF-IDF", TFIDFMethod(config)),
        ("TextRank", TextRankMethod(config)),
        ("PositionRank", PositionRankMethod(config)),
        ("RAKE", RAKEMethod(config)),
        ("YAKE", YAKEMethod(config)),
        ("KeyBERT", KeyBERTMethod(config)),
        ("SIFRank", SIFRankMethod(config)),
        ("LDA", LDAMethod(config)),
        ("BERTopic", BERTopicMethod(config)),
        ("DTM", DTMMethod(config)),
        ("Prompt-LLM", PromptLLMMethod(config)),
        ("Only-KG", OnlyKGMethod(config)),
        ("Static-KG", StaticKGMethod(config)),
    ]

    # 运行 D1 实验
    print("\n" + "=" * 70)
    print("D1 微博数据集实验")
    print("=" * 70)

    d1_results = []
    for method_name, method in baseline_methods:
        if method is None:
            # Static Keywords: 直接用第一个关键词
            preds = [[gt[0]] if gt else [] for gt in d1_keywords]
            p = np.mean([compute_precision_at_k(p, g, 10) for p, g in zip(preds, d1_keywords)])
            r = np.mean([compute_recall_at_k(p, g, 10) for p, g in zip(preds, d1_keywords)])
            f1 = np.mean([compute_f1_at_k(p, g, 10) for p, g in zip(preds, d1_keywords)])
            div = np.mean([compute_diversity(p) for p in preds])
            cov = compute_coverage(preds, d1_keywords)
            result = {'method': method_name, 'precision_at_10': p, 'recall_at_10': r, 'f1_at_10': f1, 'diversity': div, 'coverage': cov}
            print(f"\n  {method_name}: P@10={p:.4f}, R@10={r:.4f}, F1={f1:.4f}, Div={div:.4f}, Cov={cov:.4f}")
        else:
            result = run_baseline_experiment(method_name, method, d1_texts, d1_keywords, top_k=10)
        d1_results.append(result)

    # 运行 D2 实验
    d2_results = []
    if d2_texts:
        print("\n" + "=" * 70)
        print("D2 新闻数据集实验")
        print("=" * 70)

        for method_name, method in baseline_methods:
            if method is None:
                preds = [[gt[0]] if gt else [] for gt in d2_keywords]
                p = np.mean([compute_precision_at_k(p, g, 10) for p, g in zip(preds, d2_keywords)])
                r = np.mean([compute_recall_at_k(p, g, 10) for p, g in zip(preds, d2_keywords)])
                f1 = np.mean([compute_f1_at_k(p, g, 10) for p, g in zip(preds, d2_keywords)])
                div = np.mean([compute_diversity(p) for p in preds])
                cov = compute_coverage(preds, d2_keywords)
                result = {'method': method_name, 'precision_at_10': p, 'recall_at_10': r, 'f1_at_10': f1, 'diversity': div, 'coverage': cov}
                print(f"\n  {method_name}: P@10={p:.4f}, R@10={r:.4f}, F1={f1:.4f}, Div={div:.4f}, Cov={cov:.4f}")
            else:
                result = run_baseline_experiment(method_name, method, d2_texts, d2_keywords, top_k=10)
            d2_results.append(result)

    # 保存结果
    output_dir = project_root / "version2" / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 D1 结果
    d1_csv_path = output_dir / "baseline_results_d1.csv"
    fields = ['method', 'precision_at_10', 'recall_at_10', 'f1_at_10', 'diversity', 'coverage']
    with open(d1_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(d1_results)
    print(f"\nD1 结果已保存: {d1_csv_path}")

    # 保存 D2 结果
    if d2_results:
        d2_csv_path = output_dir / "baseline_results_d2.csv"
        with open(d2_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(d2_results)
        print(f"D2 结果已保存: {d2_csv_path}")

    # 打印汇总表
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    print(f"\n{'Method':<15} {'P@10':>8} {'R@10':>8} {'F1@10':>8} {'Div':>8} {'Cov':>8}")
    print("-" * 60)
    for r in d1_results:
        print(f"{r['method']:<15} {r['precision_at_10']:>8.4f} {r['recall_at_10']:>8.4f} {r['f1_at_10']:>8.4f} {r['diversity']:>8.4f} {r['coverage']:>8.4f}")

    if d2_results:
        print(f"\n{'Method':<15} {'P@10':>8} {'R@10':>8} {'F1@10':>8} {'Div':>8} {'Cov':>8}")
        print("-" * 60)
        print("  (D2 新闻数据集)")
        print("-" * 60)
        for r in d2_results:
            print(f"{r['method']:<15} {r['precision_at_10']:>8.4f} {r['recall_at_10']:>8.4f} {r['f1_at_10']:>8.4f} {r['diversity']:>8.4f} {r['coverage']:>8.4f}")

    print("\n实验完成！")

if __name__ == "__main__":
    main()
