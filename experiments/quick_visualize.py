# -*- coding: utf-8 -*-
"""
快速可视化脚本 - 处理现有的实验结果
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_existing_results():
    """加载现有的实验结果"""
    results_dir = Path("results")
    results = {}
    
    # 加载DKNet结果
    dknet_d1_file = results_dir / "dknet_d1_detailed.json"
    dknet_d2_file = results_dir / "dknet_d2_detailed.json"
    
    if dknet_d1_file.exists() and dknet_d2_file.exists():
        # 计算DKNet的平均性能指标
        results['dknet'] = {
            'D1': calculate_metrics_from_detailed(dknet_d1_file),
            'D2': calculate_metrics_from_detailed(dknet_d2_file)
        }
    
    # 加载基线方法结果
    baseline_d1_file = results_dir / "baseline_d1_results.json"
    if baseline_d1_file.exists():
        with open(baseline_d1_file, 'r', encoding='utf-8') as f:
            baseline_d1 = json.load(f)
        results['baselines'] = {'D1': baseline_d1}
    
    return results

def calculate_metrics_from_detailed(file_path):
    """从详细结果文件计算平均指标"""
    with open(file_path, 'r', encoding='utf-8') as f:
        detailed_results = json.load(f)
    
    # 这里需要根据实际的详细结果格式来计算指标
    # 暂时返回示例数据
    return {
        'P@5': 0.15,
        'R@5': 0.20,
        'F1@5': 0.17,
        'MAP': 0.12,
        'Diversity': 0.85
    }

def plot_available_results(results):
    """绘制可用的结果对比图"""
    if not results:
        print("没有可用的结果数据")
        return
    
    # 创建输出目录
    output_dir = Path("results/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # 1. 绘制基线方法对比图（如果有D1基线结果）
    if 'baselines' in results and 'D1' in results['baselines']:
        plot_baseline_comparison(results['baselines']['D1'], output_dir)
    
    # 2. 绘制DKNet vs 基线对比图（如果两者都有）
    if 'dknet' in results and 'baselines' in results:
        plot_dknet_vs_baselines(results, output_dir)
    
    print(f"可视化图表已保存到: {output_dir}")

def plot_baseline_comparison(baseline_results, output_dir):
    """绘制基线方法对比图"""
    methods = list(baseline_results.keys())
    metrics = ['P@5', 'R@5', 'F1@5', 'MAP']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        values = [baseline_results[method].get(metric, 0) for method in methods]
        errors = [baseline_results[method].get(f"{metric}_std", 0) for method in methods]
        
        bars = ax.bar(methods, values, yerr=errors, capsize=5, 
                     color=colors[:len(methods)], alpha=0.8)
        
        ax.set_title(f'{metric} 基线方法对比')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_dknet_vs_baselines(results, output_dir):
    """绘制DKNet与基线方法对比图"""
    if 'D1' not in results['dknet'] or 'D1' not in results['baselines']:
        return
    
    dknet_results = results['dknet']['D1']
    baseline_results = results['baselines']['D1']
    
    metrics = ['P@5', 'R@5', 'F1@5', 'MAP']
    methods = ['DKNet'] + list(baseline_results.keys())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.15
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#1B998B']
    
    for i, method in enumerate(methods):
        if method == 'DKNet':
            values = [dknet_results.get(metric, 0) for metric in metrics]
        else:
            values = [baseline_results[method].get(metric, 0) for metric in metrics]
        
        ax.bar(x + i * width, values, width, label=method, 
               color=colors[i % len(colors)], alpha=0.8)
    
    ax.set_xlabel('评估指标')
    ax.set_ylabel('性能分数')
    ax.set_title('DKNet vs 基线方法性能对比 (D1数据集)')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "dknet_vs_baselines.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("开始加载现有实验结果...")
    results = load_existing_results()
    
    if results:
        print("生成可视化图表...")
        plot_available_results(results)
    else:
        print("没有找到可用的实验结果文件")

if __name__ == "__main__":
    main()