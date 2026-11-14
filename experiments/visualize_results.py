# -*- coding: utf-8 -*-
"""
实验结果可视化脚本
生成各种图表和分析报告
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
sns.set_style("whitegrid")
sns.set_palette("husl")

class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载实验结果
        self.results = self._load_results()
        
        # 颜色配置
        self.colors = {
            'DKNet': '#2E86AB',
            'TF-IDF': '#A23B72',
            'TextRank': '#F18F01',
            'KeyBERT': '#C73E1D',
            'LDA': '#592E83',
            'YAKE': '#1B998B'
        }
    
    def _load_results(self) -> Dict:
        """加载实验结果"""
        results_file = self.results_dir / "all_results.json"
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"结果文件不存在: {results_file}")
            return {}
    
    def plot_performance_comparison(self, save_path: Optional[str] = None):
        """绘制性能对比图"""
        if not self.results:
            return
        
        # 准备数据
        datasets = list(self.results['dknet'].keys())
        metrics = ['P@5', 'R@5', 'F1@5', 'MAP', 'Diversity']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # 收集所有方法的数据
            methods = ['DKNet']
            values = {dataset: [] for dataset in datasets}
            
            # DKNet结果
            for dataset in datasets:
                dknet_value = self.results['dknet'][dataset]['results'].get(metric, 0)
                values[dataset].append(dknet_value)
            
            # 基线方法结果
            if 'baselines' in self.results:
                baseline_methods = list(self.results['baselines'][datasets[0]].keys())
                methods.extend(baseline_methods)
                
                for method in baseline_methods:
                    for dataset in datasets:
                        baseline_value = self.results['baselines'][dataset][method].get(metric, 0)
                        values[dataset].append(baseline_value)
            
            # 绘制柱状图
            x = np.arange(len(datasets))
            width = 0.15
            
            for j, method in enumerate(methods):
                method_values = [values[dataset][j] for dataset in datasets]
                color = self.colors.get(method, f'C{j}')
                ax.bar(x + j * width, method_values, width, label=method, color=color)
            
            ax.set_xlabel('数据集')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} 性能对比')
            ax.set_xticks(x + width * (len(methods) - 1) / 2)
            ax.set_xticklabels(datasets)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 删除多余的子图
        if len(metrics) < len(axes):
            for i in range(len(metrics), len(axes)):
                fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_ablation_study(self, save_path: Optional[str] = None):
        """绘制消融研究结果"""
        if 'ablation' not in self.results or not self.results['ablation']:
            print("没有消融研究结果")
            return
        
        datasets = list(self.results['ablation'].keys())
        fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 6))
        
        if len(datasets) == 1:
            axes = [axes]
        
        for i, dataset in enumerate(datasets):
            ax = axes[i]
            
            # 获取完整模型和消融版本的F1@5分数
            full_f1 = self.results['dknet'][dataset]['results'].get('F1@5', 0)
            ablation_results = self.results['ablation'][dataset]
            
            # 准备数据
            components = ['完整模型']
            f1_scores = [full_f1]
            
            component_names = {
                'no_temporal_kg': '无时间KG',
                'no_adaptive_fusion': '无自适应融合',
                'no_mmr': '无MMR',
                'no_llm_candidates': '无LLM候选'
            }
            
            for ablation_name, results in ablation_results.items():
                components.append(component_names.get(ablation_name, ablation_name))
                f1_scores.append(results.get('F1@5', 0))
            
            # 绘制柱状图
            bars = ax.bar(components, f1_scores, color=['#2E86AB'] + ['#FF6B6B'] * (len(components) - 1))
            
            # 添加数值标签
            for bar, score in zip(bars, f1_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{score:.3f}', ha='center', va='bottom')
            
            ax.set_ylabel('F1@5 分数')
            ax.set_title(f'{dataset} 数据集消融研究')
            ax.set_ylim(0, max(f1_scores) * 1.1)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "ablation_study.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_parameter_sensitivity(self, save_path: Optional[str] = None):
        """绘制参数敏感性分析"""
        if 'sensitivity' not in self.results or not self.results['sensitivity']:
            print("没有参数敏感性分析结果")
            return
        
        sensitivity_results = self.results['sensitivity']
        parameters = list(sensitivity_results.keys())
        
        fig, axes = plt.subplots(1, len(parameters), figsize=(6 * len(parameters), 6))
        
        if len(parameters) == 1:
            axes = [axes]
        
        for i, param_name in enumerate(parameters):
            ax = axes[i]
            
            param_data = sensitivity_results[param_name]
            param_values = [item['parameter_value'] for item in param_data]
            f1_scores = [item['f1_5'] for item in param_data]
            map_scores = [item['map'] for item in param_data]
            diversity_scores = [item['diversity'] for item in param_data]
            
            # 绘制多条线
            ax.plot(param_values, f1_scores, 'o-', label='F1@5', linewidth=2, markersize=6)
            ax.plot(param_values, map_scores, 's-', label='MAP', linewidth=2, markersize=6)
            ax.plot(param_values, diversity_scores, '^-', label='Diversity', linewidth=2, markersize=6)
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('分数')
            ax.set_title(f'{param_name} 敏感性分析')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "parameter_sensitivity.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metric_correlation(self, save_path: Optional[str] = None):
        """绘制指标相关性热力图"""
        if not self.results:
            return
        
        # 收集所有方法在所有数据集上的指标
        all_metrics = []
        
        for dataset in self.results['dknet'].keys():
            # DKNet结果
            dknet_metrics = self.results['dknet'][dataset]['results']
            all_metrics.append(dknet_metrics)
            
            # 基线方法结果
            if 'baselines' in self.results:
                for method_results in self.results['baselines'][dataset].values():
                    all_metrics.append(method_results)
        
        # 转换为DataFrame
        df = pd.DataFrame(all_metrics)
        
        # 只保留数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols]
        
        # 计算相关性矩阵
        correlation_matrix = df_numeric.corr()
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('评估指标相关性热力图')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "metric_correlation.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_radar(self, save_path: Optional[str] = None):
        """绘制性能雷达图"""
        if not self.results:
            return
        
        datasets = list(self.results['dknet'].keys())
        
        for dataset in datasets:
            # 准备数据
            metrics = ['P@5', 'R@5', 'F1@5', 'MAP', 'Diversity']
            
            # 收集方法和分数
            methods_data = {}
            
            # DKNet
            dknet_scores = [self.results['dknet'][dataset]['results'].get(m, 0) for m in metrics]
            methods_data['DKNet'] = dknet_scores
            
            # 基线方法
            if 'baselines' in self.results:
                for method_name, method_results in self.results['baselines'][dataset].items():
                    method_scores = [method_results.get(m, 0) for m in metrics]
                    methods_data[method_name] = method_scores
            
            # 绘制雷达图
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            for method_name, scores in methods_data.items():
                scores += scores[:1]  # 闭合图形
                color = self.colors.get(method_name, None)
                ax.plot(angles, scores, 'o-', linewidth=2, label=method_name, color=color)
                ax.fill(angles, scores, alpha=0.1, color=color)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title(f'{dataset} 数据集性能雷达图', size=16, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}_{dataset.lower()}_radar.png", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(self.output_dir / f"{dataset.lower()}_performance_radar.png", dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def plot_improvement_analysis(self, save_path: Optional[str] = None):
        """绘制改进分析图"""
        if 'baselines' not in self.results:
            print("没有基线方法结果")
            return
        
        datasets = list(self.results['dknet'].keys())
        baseline_methods = list(self.results['baselines'][datasets[0]].keys())
        
        # 计算改进百分比
        improvements = {}
        
        for dataset in datasets:
            dataset_improvements = {}
            dknet_f1 = self.results['dknet'][dataset]['results'].get('F1@5', 0)
            
            for method in baseline_methods:
                baseline_f1 = self.results['baselines'][dataset][method].get('F1@5', 0)
                if baseline_f1 > 0:
                    improvement = ((dknet_f1 - baseline_f1) / baseline_f1) * 100
                    dataset_improvements[method] = improvement
                else:
                    dataset_improvements[method] = 0
            
            improvements[dataset] = dataset_improvements
        
        # 绘制改进图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(baseline_methods))
        width = 0.35
        
        for i, dataset in enumerate(datasets):
            improvement_values = [improvements[dataset][method] for method in baseline_methods]
            ax.bar(x + i * width, improvement_values, width, label=dataset)
        
        ax.set_xlabel('基线方法')
        ax.set_ylabel('F1@5 改进百分比 (%)')
        ax.set_title('DKNet相对于基线方法的改进')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(baseline_methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 添加数值标签
        for i, dataset in enumerate(datasets):
            improvement_values = [improvements[dataset][method] for method in baseline_methods]
            for j, value in enumerate(improvement_values):
                ax.text(j + i * width, value + (1 if value >= 0 else -3), 
                       f'{value:.1f}%', ha='center', va='bottom' if value >= 0 else 'top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "improvement_analysis.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_summary_table(self, save_path: Optional[str] = None):
        """生成结果摘要表格"""
        if not self.results:
            return
        
        summary_data = []
        
        for dataset in self.results['dknet'].keys():
            # DKNet结果
            dknet_metrics = self.results['dknet'][dataset]['results']
            row = {'Dataset': dataset, 'Method': 'DKNet'}
            row.update({k: f"{v:.4f}" for k, v in dknet_metrics.items() if isinstance(v, (int, float))})
            summary_data.append(row)
            
            # 基线方法结果
            if 'baselines' in self.results:
                for method_name, method_results in self.results['baselines'][dataset].items():
                    row = {'Dataset': dataset, 'Method': method_name}
                    row.update({k: f"{v:.4f}" for k, v in method_results.items() if isinstance(v, (int, float))})
                    summary_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(summary_data)
        
        # 保存为CSV
        if save_path:
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(self.output_dir / "summary_table.csv", index=False, encoding='utf-8-sig')
        
        print("结果摘要表格:")
        print(df.to_string(index=False))
        
        return df
    
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("开始生成可视化图表...")
        
        try:
            # 1. 性能对比图
            print("生成性能对比图...")
            self.plot_performance_comparison()
            
            # 2. 消融研究图
            print("生成消融研究图...")
            self.plot_ablation_study()
            
            # 3. 参数敏感性分析图
            print("生成参数敏感性分析图...")
            self.plot_parameter_sensitivity()
            
            # 4. 指标相关性热力图
            print("生成指标相关性热力图...")
            self.plot_metric_correlation()
            
            # 5. 性能雷达图
            print("生成性能雷达图...")
            self.plot_performance_radar()
            
            # 6. 改进分析图
            print("生成改进分析图...")
            self.plot_improvement_analysis()
            
            # 7. 摘要表格
            print("生成摘要表格...")
            self.generate_summary_table()
            
            print(f"所有可视化图表已保存到: {self.output_dir}")
            
        except Exception as e:
            print(f"生成可视化图表时出现错误: {str(e)}")
            raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化实验结果')
    parser.add_argument('--results_dir', type=str, default='outputs', help='结果目录路径')
    parser.add_argument('--output_dir', type=str, help='可视化输出目录（可选）')
    parser.add_argument('--plots', nargs='+', 
                       choices=['comparison', 'ablation', 'sensitivity', 'correlation', 'radar', 'improvement', 'all'],
                       default=['all'], help='指定生成的图表类型')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = ResultVisualizer(args.results_dir)
    
    if args.output_dir:
        visualizer.output_dir = Path(args.output_dir)
        visualizer.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成指定的图表
    if 'all' in args.plots:
        visualizer.generate_all_visualizations()
    else:
        if 'comparison' in args.plots:
            visualizer.plot_performance_comparison()
        if 'ablation' in args.plots:
            visualizer.plot_ablation_study()
        if 'sensitivity' in args.plots:
            visualizer.plot_parameter_sensitivity()
        if 'correlation' in args.plots:
            visualizer.plot_metric_correlation()
        if 'radar' in args.plots:
            visualizer.plot_performance_radar()
        if 'improvement' in args.plots:
            visualizer.plot_improvement_analysis()
        
        # 总是生成摘要表格
        visualizer.generate_summary_table()


if __name__ == "__main__":
    main()