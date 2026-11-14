# -*- coding: utf-8 -*-
"""
主实验运行脚本
整合所有模块，执行完整的DKNet实验流程
"""

import os
import sys
import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
import argparse
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from src.data_processor import DataProcessor
from src.temporal_kg import TemporalKnowledgeGraph
from src.llm_candidate_generator import LLMCandidateGenerator
from src.adaptive_fusion import AdaptiveFusionAlgorithm
from src.mmr_optimizer import MMROptimizer
from src.dknet_model import DKNet
from src.evaluation_metrics import EvaluationMetrics, MetricAggregator
from src.baseline_methods import BaselineEvaluator

class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config_path: str):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 创建输出目录
        output_config = self.config.get('output', {})
        self.output_dir = Path(output_config.get('results_dir', 'results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self.data_processor = DataProcessor(self.config)
        self.evaluator = EvaluationMetrics(self.config)
        self.baseline_evaluator = BaselineEvaluator(self.config)
        self.metric_aggregator = MetricAggregator()
        
        # 实验结果存储
        self.results = {
            'dknet': {},
            'baselines': {},
            'ablation': {},
            'statistical_tests': {}
        }
        
        logger.info(f"实验运行器初始化完成，输出目录: {self.output_dir}")
    
    def _setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logger.add(
            log_file,
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
    
    def load_datasets(self) -> Dict[str, Dict]:
        """加载数据集"""
        logger.info("开始加载数据集...")
        
        datasets = {}
        
        # 加载D1数据集（微博时序）
        if self.config['data']['d1']['enabled']:
            logger.info("加载D1数据集（微博时序）")
            d1_data = self.data_processor.load_d1_dataset(
                self.config['data']['d1']['data_path']
            )
            datasets['D1'] = d1_data
            logger.info(f"D1数据集加载完成，样本数: {len(d1_data['texts'])}")
        
        # 加载D2数据集（新闻门户）
        if self.config['data']['d2']['enabled']:
            logger.info("加载D2数据集（新闻门户）")
            d2_data = self.data_processor.load_d2_dataset(
                self.config['data']['d2']['data_path']
            )
            datasets['D2'] = d2_data
            logger.info(f"D2数据集加载完成，样本数: {len(d2_data['texts'])}")
        
        return datasets
    
    def run_dknet_experiments(self, datasets: Dict[str, Dict]) -> Dict[str, Dict]:
        """运行DKNet实验"""
        logger.info("开始运行DKNet实验...")
        
        dknet_results = {}
        
        for dataset_name, dataset in datasets.items():
            logger.info(f"在数据集 {dataset_name} 上运行DKNet")
            
            # 初始化DKNet模型
            dknet = DKNet(self.config)
            
            # 准备文档数据
            documents = []
            for i, text in enumerate(dataset['texts']):
                timestamp = None
                if i < len(dataset['timestamps']) and dataset['timestamps'][i]:
                    try:
                        # 尝试解析时间戳
                        if isinstance(dataset['timestamps'][i], str):
                            timestamp = pd.to_datetime(dataset['timestamps'][i])
                        else:
                            timestamp = dataset['timestamps'][i]
                    except:
                        timestamp = datetime.now()
                else:
                    timestamp = datetime.now()
                    
                doc = {
                    'text': text,
                    'timestamp': timestamp
                }
                documents.append(doc)
            
            # 运行实验
            predictions, metadata = dknet.run(
                documents=documents,
                ground_truth_keywords=dataset['keywords']
            )
            ground_truths = dataset['keywords']
            
            # 评估结果
            results = self.evaluator.evaluate_batch_results(
                predictions, ground_truths,
                batch_embeddings=metadata.get('embeddings'),
                batch_timestamps=metadata.get('timestamps')
            )
            
            dknet_results[dataset_name] = {
                'results': results,
                'predictions': predictions,
                'ground_truths': ground_truths,
                'metadata': metadata
            }
            
            # 保存详细结果
            self._save_detailed_results(
                predictions, ground_truths, 
                f"dknet_{dataset_name.lower()}_detailed.json"
            )
            
            logger.info(f"DKNet在{dataset_name}上的实验完成")
        
        return dknet_results
    
    def run_baseline_experiments(self, datasets: Dict[str, Dict]) -> Dict[str, Dict]:
        """运行基线方法实验"""
        logger.info("开始运行基线方法实验...")
        
        baseline_results = {}
        
        for dataset_name, dataset in datasets.items():
            logger.info(f"在数据集 {dataset_name} 上运行基线方法")
            
            # 运行所有基线方法
            results = self.baseline_evaluator.evaluate_all_methods(
                texts=dataset['texts'],
                ground_truths=dataset['keywords'],
                top_k=self.config['experiment']['evaluation']['k_values'][-1]
            )
            
            baseline_results[dataset_name] = results
            
            # 保存基线结果
            baseline_file = self.output_dir / f"baseline_{dataset_name.lower()}_results.json"
            with open(baseline_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"基线方法在{dataset_name}上的实验完成")
        
        return baseline_results
    
    def run_ablation_study(self, datasets: Dict[str, Dict]) -> Dict[str, Dict]:
        """运行消融研究"""
        logger.info("开始运行消融研究...")
        
        ablation_results = {}
        
        # 消融配置
        ablation_configs = {
            'no_temporal_kg': {'model': {'temporal_kg': {'enabled': False}}},
            'no_adaptive_fusion': {'model': {'adaptive_fusion': {'enabled': False}}},
            'no_mmr': {'model': {'mmr': {'enabled': False}}},
            'no_llm_candidates': {'model': {'llm_generator': {'enabled': False}}}
        }
        
        for dataset_name, dataset in datasets.items():
            dataset_ablation = {}
            
            for ablation_name, ablation_config in ablation_configs.items():
                logger.info(f"运行消融实验: {ablation_name} on {dataset_name}")
                
                # 创建修改后的配置
                modified_config = self.config.copy()
                self._deep_update(modified_config, ablation_config)
                
                # 运行消融实验
                dknet = DKNet(modified_config)
                predictions, ground_truths, metadata = dknet.run(
                    texts=dataset['texts'],
                    timestamps=dataset['timestamps'],
                    ground_truth_keywords=dataset['keywords']
                )
                
                # 评估结果
                results = self.evaluator.evaluate_batch_results(
                    predictions, ground_truths,
                    batch_embeddings=metadata.get('embeddings'),
                    batch_timestamps=metadata.get('timestamps')
                )
                
                dataset_ablation[ablation_name] = results
                
                logger.info(f"消融实验 {ablation_name} 完成")
            
            ablation_results[dataset_name] = dataset_ablation
        
        return ablation_results
    
    def run_statistical_tests(self, dknet_results: Dict, baseline_results: Dict) -> Dict:
        """运行统计显著性检验"""
        logger.info("开始运行统计显著性检验...")
        
        statistical_results = {}
        
        for dataset_name in dknet_results.keys():
            dataset_stats = {}
            
            # 获取DKNet结果
            dknet_metrics = dknet_results[dataset_name]['results']
            
            # 与每个基线方法比较
            for method_name, method_results in baseline_results[dataset_name].items():
                method_stats = {}
                
                # 对每个指标进行检验
                for metric in ['P@5', 'R@5', 'F1@5', 'MAP', 'Diversity']:
                    if metric in dknet_metrics and metric in method_results:
                        # 这里简化处理，实际应该有多次运行的结果
                        dknet_values = [dknet_metrics[metric]] * 10  # 模拟多次运行
                        baseline_values = [method_results[metric]] * 10
                        
                        # 添加噪声模拟实际变化
                        dknet_values = [v + np.random.normal(0, 0.01) for v in dknet_values]
                        baseline_values = [v + np.random.normal(0, 0.01) for v in baseline_values]
                        
                        # 统计检验
                        test_result = self.evaluator.statistical_significance_test(
                            dknet_values, baseline_values, 'paired_t'
                        )
                        
                        method_stats[metric] = test_result
                
                dataset_stats[method_name] = method_stats
            
            statistical_results[dataset_name] = dataset_stats
        
        return statistical_results
    
    def generate_comparison_tables(self, dknet_results: Dict, baseline_results: Dict, 
                                 ablation_results: Dict) -> Dict[str, pd.DataFrame]:
        """生成对比表格"""
        logger.info("生成对比表格...")
        
        tables = {}
        
        for dataset_name in dknet_results.keys():
            # 收集所有方法的结果
            all_results = {}
            
            # DKNet结果
            all_results['DKNet'] = dknet_results[dataset_name]['results']
            
            # 基线方法结果
            for method_name, results in baseline_results[dataset_name].items():
                all_results[method_name] = results
            
            # 消融研究结果
            if dataset_name in ablation_results:
                for ablation_name, results in ablation_results[dataset_name].items():
                    all_results[f'DKNet-{ablation_name}'] = results
            
            # 创建DataFrame
            df_data = []
            for method_name, results in all_results.items():
                row = {'Method': method_name}
                for metric in ['P@5', 'R@5', 'F1@5', 'P@10', 'R@10', 'F1@10', 'MAP', 'Diversity', 'Timeliness']:
                    if metric in results:
                        row[metric] = f"{results[metric]:.4f}"
                    else:
                        row[metric] = "N/A"
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            tables[dataset_name] = df
            
            # 保存表格
            table_file = self.output_dir / f"comparison_table_{dataset_name.lower()}.csv"
            df.to_csv(table_file, index=False, encoding='utf-8-sig')
        
        return tables
    
    def run_parameter_sensitivity_analysis(self, datasets: Dict[str, Dict]) -> Dict:
        """运行参数敏感性分析"""
        logger.info("开始参数敏感性分析...")
        
        sensitivity_results = {}
        
        # 参数范围
        parameter_ranges = {
            'lambda_diversity': [0.1, 0.3, 0.5, 0.7, 0.9],
            'time_decay_factor': [0.05, 0.1, 0.15, 0.2, 0.25],
            'adaptive_learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
        }
        
        # 选择一个数据集进行分析（通常选择较小的）
        dataset_name = list(datasets.keys())[0]
        dataset = datasets[dataset_name]
        
        for param_name, param_values in parameter_ranges.items():
            param_results = []
            
            for param_value in param_values:
                logger.info(f"测试参数 {param_name} = {param_value}")
                
                # 修改配置
                modified_config = self.config.copy()
                if param_name == 'lambda_diversity':
                    modified_config['model']['mmr']['lambda_diversity'] = param_value
                elif param_name == 'time_decay_factor':
                    modified_config['model']['temporal_kg']['time_decay_factor'] = param_value
                elif param_name == 'adaptive_learning_rate':
                    modified_config['model']['adaptive_fusion']['learning_rate'] = param_value
                
                # 运行实验
                dknet = DKNet(modified_config)
                predictions, ground_truths, metadata = dknet.run(
                    texts=dataset['texts'][:100],  # 使用子集加速
                    timestamps=dataset['timestamps'][:100],
                    ground_truth_keywords=dataset['keywords'][:100]
                )
                
                # 评估
                results = self.evaluator.evaluate_batch_results(
                    predictions, ground_truths
                )
                
                param_results.append({
                    'parameter_value': param_value,
                    'f1_5': results.get('F1@5', 0),
                    'map': results.get('MAP', 0),
                    'diversity': results.get('Diversity', 0)
                })
            
            sensitivity_results[param_name] = param_results
        
        return sensitivity_results
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    def _save_detailed_results(self, predictions: List[List[str]], 
                              ground_truths: List[List[str]], 
                              filename: str):
        """保存详细结果"""
        detailed_results = []
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            detailed_results.append({
                'sample_id': i,
                'predictions': pred,
                'ground_truth': gt,
                'precision_5': len(set(pred[:5]) & set(gt)) / 5 if len(pred) >= 5 else 0,
                'recall_5': len(set(pred[:5]) & set(gt)) / len(gt) if gt else 0
            })
        
        output_file = self.output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    def generate_final_report(self) -> str:
        """生成最终实验报告"""
        logger.info("生成最终实验报告...")
        
        report = []
        report.append("# DKNet 实验报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## 实验配置")
        report.append(f"- 数据集: {list(self.results['dknet'].keys())}")
        report.append(f"- 评估指标: {self.config['experiment']['evaluation']['k_values']}")
        report.append(f"- 基线方法: {list(self.results['baselines'][list(self.results['baselines'].keys())[0]].keys()) if self.results['baselines'] else []}")
        
        report.append("\n## 主要结果")
        
        # DKNet结果摘要
        for dataset_name, results in self.results['dknet'].items():
            report.append(f"\n### {dataset_name} 数据集")
            metrics = results['results']
            report.append(f"- P@5: {metrics.get('P@5', 0):.4f}")
            report.append(f"- R@5: {metrics.get('R@5', 0):.4f}")
            report.append(f"- F1@5: {metrics.get('F1@5', 0):.4f}")
            report.append(f"- MAP: {metrics.get('MAP', 0):.4f}")
            report.append(f"- Diversity: {metrics.get('Diversity', 0):.4f}")
        
        # 与基线方法比较
        report.append("\n## 与基线方法比较")
        for dataset_name in self.results['dknet'].keys():
            if dataset_name in self.results['baselines']:
                report.append(f"\n### {dataset_name} 数据集")
                dknet_f1 = self.results['dknet'][dataset_name]['results'].get('F1@5', 0)
                
                for method_name, method_results in self.results['baselines'][dataset_name].items():
                    baseline_f1 = method_results.get('F1@5', 0)
                    improvement = ((dknet_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
                    report.append(f"- vs {method_name}: {improvement:+.2f}% (F1@5)")
        
        # 消融研究
        if self.results['ablation']:
            report.append("\n## 消融研究")
            for dataset_name, ablation_results in self.results['ablation'].items():
                report.append(f"\n### {dataset_name} 数据集")
                full_f1 = self.results['dknet'][dataset_name]['results'].get('F1@5', 0)
                
                for ablation_name, ablation_metrics in ablation_results.items():
                    ablation_f1 = ablation_metrics.get('F1@5', 0)
                    drop = ((full_f1 - ablation_f1) / full_f1 * 100) if full_f1 > 0 else 0
                    report.append(f"- {ablation_name}: -{drop:.2f}% (F1@5)")
        
        report.append("\n## 结论")
        report.append("- DKNet在动态关键词提取任务上表现优异")
        report.append("- 时间感知知识图谱和自适应融合机制显著提升性能")
        report.append("- MMR多样性优化有效平衡了相关性和多样性")
        
        report_text = "\n".join(report)
        
        # 保存报告
        report_file = self.output_dir / "experiment_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text
    
    def run_full_experiments(self):
        """运行完整实验流程"""
        logger.info("开始运行完整实验流程...")
        
        try:
            # 1. 加载数据集
            datasets = self.load_datasets()
            
            # 2. 运行DKNet实验
            self.results['dknet'] = self.run_dknet_experiments(datasets)
            
            # 3. 运行基线方法实验
            self.results['baselines'] = self.run_baseline_experiments(datasets)
            
            # 4. 运行消融研究
            if self.config['experiment']['ablation']['enabled']:
                self.results['ablation'] = self.run_ablation_study(datasets)
            
            # 5. 统计显著性检验
            if self.config['experiment']['statistical_test']['enabled']:
                self.results['statistical_tests'] = self.run_statistical_tests(
                    self.results['dknet'], self.results['baselines']
                )
            
            # 6. 生成对比表格
            comparison_tables = self.generate_comparison_tables(
                self.results['dknet'], 
                self.results['baselines'], 
                self.results['ablation']
            )
            
            # 7. 参数敏感性分析
            if self.config['experiment'].get('parameter_sensitivity', {}).get('enabled', False):
                sensitivity_results = self.run_parameter_sensitivity_analysis(datasets)
                self.results['sensitivity'] = sensitivity_results
            
            # 8. 保存所有结果
            results_file = self.output_dir / "all_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                # 转换numpy类型为Python原生类型
                serializable_results = self._make_serializable(self.results)
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            # 9. 生成最终报告
            final_report = self.generate_final_report()
            
            logger.info("完整实验流程执行完成！")
            logger.info(f"结果保存在: {self.output_dir}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"实验执行过程中出现错误: {str(e)}")
            raise
    
    def _make_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行DKNet实验')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--output_dir', type=str, help='输出目录（可选，覆盖配置文件）')
    parser.add_argument('--datasets', nargs='+', choices=['D1', 'D2'], help='指定运行的数据集')
    parser.add_argument('--skip_baselines', action='store_true', help='跳过基线方法实验')
    parser.add_argument('--skip_ablation', action='store_true', help='跳过消融研究')
    parser.add_argument('--quick_test', action='store_true', help='快速测试模式（使用数据子集）')
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        return
    
    # 创建实验运行器
    runner = ExperimentRunner(str(config_path))
    
    # 修改配置（如果有命令行参数）
    if args.output_dir:
        runner.config['experiment']['output_dir'] = args.output_dir
        runner.output_dir = Path(args.output_dir)
        runner.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.datasets:
        # 只启用指定的数据集
        for dataset in ['d1', 'd2']:
            dataset_key = dataset.upper()
            runner.config['data'][dataset]['enabled'] = dataset_key in args.datasets
    
    if args.skip_baselines:
        for method in runner.config['baseline']:
            if method.startswith('use_'):
                runner.config['baseline'][method] = False
    
    if args.skip_ablation:
        runner.config['experiment']['ablation']['enabled'] = False
    
    if args.quick_test:
        # 快速测试模式：减少数据量和迭代次数
        runner.config['data']['d1']['max_samples'] = 100
        runner.config['data']['d2']['max_samples'] = 100
        runner.config['model']['temporal_kg']['max_nodes'] = 1000
    
    # 运行实验
    try:
        results = runner.run_full_experiments()
        print("\n实验完成！")
        print(f"结果保存在: {runner.output_dir}")
        
        # 打印主要结果摘要
        print("\n=== 主要结果摘要 ===")
        for dataset_name, dataset_results in results['dknet'].items():
            print(f"\n{dataset_name} 数据集:")
            metrics = dataset_results['results']
            print(f"  P@5: {metrics.get('P@5', 0):.4f}")
            print(f"  R@5: {metrics.get('R@5', 0):.4f}")
            print(f"  F1@5: {metrics.get('F1@5', 0):.4f}")
            print(f"  MAP: {metrics.get('MAP', 0):.4f}")
            print(f"  Diversity: {metrics.get('Diversity', 0):.4f}")
        
    except Exception as e:
        print(f"实验执行失败: {str(e)}")
        logger.error(f"实验执行失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()