"""
XGFT模型K折交叉验证主执行脚本
文件名: main.py
"""

import sys
import os
import logging
import json
from typing import Optional, Dict, List
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config, setup_experiment, print_feature_summary
from utils import set_seed, setup_logging, print_system_info
from data_loader import load_knowledge_graph, validate_data_quality, print_data_quality_report
from models import create_model, analyze_model_capacity
from trainer import train_model
from prediction import PredictionService

logger = logging.getLogger(__name__)

# 运行配置
RUN_MODE = 'train'  # 'train', 'predict', 'experiment'
EXPERIMENT_MODE = 'full_biological'  # 'baseline', 'expression_only', 'llm_only', 'full_biological'
DATA_FILE = str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'data/processed/kg_v2_xgft_coding_nosim.csv')
MODEL_PATH = 'checkpoints/bestmodel.pth'
USE_KFOLD = True # 是否使用K折交叉验证
N_FOLDS = 5  # K折的折数


def setup_environment():
    """设置运行环境"""
    set_seed(config.seed)
    setup_logging(config.paths.logs_dir, config.log_level)
    print_system_info()
    print_feature_summary()
    config.print_config()


def train_mode():
    """训练模式 - 支持K折交叉验证"""
    logger.info(f"🚀 启动XGFT模型训练 (K折: {'是' if USE_KFOLD else '否'})")

    try:
        # 加载数据
        logger.info("📊 加载增强知识图谱数据...")
        data = load_knowledge_graph(kg_file=DATA_FILE)

        # 提取原始特征维度信息
        node_feature_dims = {}
        for node_type in data.node_types:
            if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                node_feature_dims[node_type] = data[node_type].x.shape[1]
            else:
                node_feature_dims[node_type] = config.model.embedding_dim

        logger.info(f"🔧 提取到的节点原始特征维度: {node_feature_dims}")

        # 数据质量验证
        logger.info("🔍 执行数据质量验证...")
        quality_report = validate_data_quality(data)
        print_data_quality_report(quality_report)

        # 分析特征使用效果
        feature_analysis = analyze_feature_usage(quality_report, node_feature_dims)
        print_feature_analysis(feature_analysis)

        if USE_KFOLD:
            # K折交叉验证训练
            logger.info(f"🔄 开始{N_FOLDS}折交叉验证训练...")
            trainer, results = train_model(None, data, use_kfold=True)

            # 保存K折实验结果
            experiment_results = {
                'model_type': 'XGFT',
                'training_mode': 'k-fold',
                'n_folds': N_FOLDS,
                'experiment_mode': EXPERIMENT_MODE,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'use_biological_features': config.data.use_biological_features,
                    'feature_sources': config.data.feature_sources,
                    'model_params': {
                        'embedding_dim': config.model.embedding_dim,
                        'hidden_dim': config.model.hidden_dim,
                        'num_layers': config.model.num_layers,
                        'num_heads': config.model.num_heads
                    },
                    'node_feature_dims': node_feature_dims
                },
                'data_quality': quality_report,
                'feature_analysis': feature_analysis,
                'kfold_results': results
            }

        else:
            # 单次训练
            logger.info("🧠 创建XGFT模型...")
            model = create_model(
                metadata=data.metadata(),
                node_feature_dims=node_feature_dims,
                model_type='xgft'
            )

            # 分析模型容量
            logger.info("🔬 分析模型处理能力...")
            model_analysis = analyze_model_capacity(model, data)
            print_model_analysis(model_analysis)

            # 训练模型
            logger.info("🎯 开始XGFT训练...")
            trainer, results = train_model(model, data, use_kfold=False)

            # 保存实验结果
            experiment_results = {
                'model_type': 'XGFT',
                'training_mode': 'single',
                'experiment_mode': EXPERIMENT_MODE,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'use_biological_features': config.data.use_biological_features,
                    'feature_sources': config.data.feature_sources,
                    'model_params': {
                        'embedding_dim': config.model.embedding_dim,
                        'hidden_dim': config.model.hidden_dim,
                        'num_layers': config.model.num_layers,
                        'num_heads': config.model.num_heads
                    },
                    'node_feature_dims': node_feature_dims
                },
                'data_quality': quality_report,
                'feature_analysis': feature_analysis,
                'model_analysis': model_analysis,
                'performance': results
            }

        # 保存结果
        save_experiment_results(experiment_results)

        # 打印增强效果分析
        print_enhancement_analysis(feature_analysis, results, node_feature_dims)

        logger.info("✅ XGFT训练完成!")
        return trainer, results

    except Exception as e:
        logger.error(f"❌ XGFT训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def experiment_mode():
    """实验模式 - 对比不同特征配置"""
    logger.info("🧪 启动XGFT实验模式 - 特征配置对比")

    experiment_modes = ['baseline', 'expression_only', 'llm_only', 'full_biological']
    results_comparison = {}

    for mode in experiment_modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔬 开始实验: {mode}")
        logger.info(f"{'='*60}")

        try:
            # 设置实验配置
            setup_experiment(mode)

            # 重新配置环境
            print_feature_summary()

            # 加载数据
            data = load_knowledge_graph(kg_file=DATA_FILE)

            # 提取特征维度
            node_feature_dims = {}
            for node_type in data.node_types:
                if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                    node_feature_dims[node_type] = data[node_type].x.shape[1]
                else:
                    node_feature_dims[node_type] = config.model.embedding_dim

            # 验证数据
            quality_report = validate_data_quality(data)

            if USE_KFOLD:
                # K折训练
                trainer, results = train_model(None, data, use_kfold=True)
            else:
                # 单次训练
                model = create_model(
                    metadata=data.metadata(),
                    node_feature_dims=node_feature_dims,
                    model_type='xgft'
                )
                trainer, results = train_model(model, data, use_kfold=False)

            # 保存实验结果
            results_comparison[mode] = {
                'model_type': 'XGFT',
                'performance': results,
                'data_quality': quality_report,
                'node_feature_dims': node_feature_dims,
                'timestamp': datetime.now().isoformat()
            }

            if USE_KFOLD:
                logger.info(f"✅ 实验 {mode} 完成 (K折平均)")
                logger.info(f"   AUC: {results['fold_metrics']['AUC']['mean']:.4f} ± {results['fold_metrics']['AUC']['std']:.4f}")
                logger.info(f"   AUPR: {results['fold_metrics']['AUPR']['mean']:.4f} ± {results['fold_metrics']['AUPR']['std']:.4f}")
            else:
                logger.info(f"✅ 实验 {mode} 完成")
                logger.info(f"   AUC: {results.get('AUC', 'N/A'):.4f}")
                logger.info(f"   AUPR: {results.get('AUPR', 'N/A'):.4f}")

        except Exception as e:
            logger.error(f"❌ 实验 {mode} 失败: {e}")
            results_comparison[mode] = {'error': str(e)}

    # 打印实验对比
    print_experiment_comparison(results_comparison)

    # 保存对比结果
    comparison_file = config.paths.experiments_dir / f"xgft_experiment_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(results_comparison, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"📊 实验对比结果已保存: {comparison_file}")
    return results_comparison


def predict_mode():
    """预测模式"""
    logger.info("🔮 启动XGFT预测模式")

    try:
        # 创建预测服务
        predictor = PredictionService(
            model_path=MODEL_PATH,
            data_path=DATA_FILE
        )

        # 测试预测
        test_pairs = [
            ('hsa-miR-21-5p', 'PTEN'),
            ('hsa-miR-200c-3p', 'ZEB1'),
            ('hsa-miR-34a-5p', 'MDM2'),
            ('hsa-miR-155-5p', 'SOCS1'),
            ('hsa-miR-125b-5p', 'TP53'),
        ]

        results = predictor.predict_pairs(test_pairs, return_details=True)

        # 打印结果
        print("\n🔮 XGFT模型预测结果:")
        print("=" * 60)
        for result in results:
            if 'error' not in result:
                print(f"miRNA: {result['mirna']:<15} Gene: {result['gene']:<10}")
                print(f"  预测分数: {result['prediction_score']:.4f}")
                print(f"  预测标签: {'相互作用' if result['prediction_label'] == 1 else '无相互作用'}")
                print(f"  置信度: {result['confidence']:.4f}")
                print("-" * 50)
            else:
                print(f"预测失败 {result['mirna']}-{result['gene']}: {result['error']}")

        return results

    except Exception as e:
        logger.error(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_feature_usage(quality_report: Dict, node_feature_dims: Dict[str, int]) -> Dict:
    """分析特征使用情况"""
    feature_analysis = {
        'biological_features_enabled': config.data.use_biological_features,
        'feature_sources': config.data.feature_sources.copy(),
        'feature_quality': quality_report.get('feature_quality', {}),
        'feature_coverage': {},
        'original_feature_dims': node_feature_dims,
        'dimension_compression_ratios': {}
    }

    # 计算特征覆盖率和维度压缩比
    target_dim = config.model.hidden_dim

    for node_type, orig_dim in node_feature_dims.items():
        if orig_dim > 0:
            compression_ratio = target_dim / orig_dim
            feature_analysis['dimension_compression_ratios'][node_type] = {
                'original_dim': orig_dim,
                'target_dim': target_dim,
                'compression_ratio': compression_ratio,
                'information_preservation_expected': min(1.0, compression_ratio)
            }

    # 计算特征覆盖率
    if 'feature_quality' in quality_report:
        for node_type, quality in quality_report['feature_quality'].items():
            mean_norm = quality.get('mean_norm', 0)

            if mean_norm > 0.01:
                coverage = 'High'
            elif mean_norm > 0.001:
                coverage = 'Medium'
            else:
                coverage = 'Low'

            feature_analysis['feature_coverage'][node_type] = {
                'coverage_level': coverage,
                'mean_feature_norm': mean_norm,
                'has_meaningful_features': quality.get('has_nonzero', False) and mean_norm > 0.001
            }

    return feature_analysis


def print_feature_analysis(feature_analysis: Dict):
    """打印特征分析结果"""
    print("\n" + "=" * 70)
    print("🎯 增强特征使用分析")
    print("=" * 70)

    print(f"生物学特征启用: {'✅' if feature_analysis['biological_features_enabled'] else '❌'}")

    if feature_analysis['biological_features_enabled']:
        print("\n📊 特征源配置:")
        feature_names = {
            'gene_expression': '基因表达谱',
            'mirna_expression': 'miRNA表达谱',
            'llm_embeddings': 'LLM嵌入',
            'biowordvec_embeddings': 'BioWordVec嵌入'
        }

        for source, enabled in feature_analysis['feature_sources'].items():
            status = "✅" if enabled else "❌"
            name = feature_names.get(source, source)
            print(f"  {status} {name}")

        print("\n🔧 原始特征维度:")
        for node_type, dim in feature_analysis['original_feature_dims'].items():
            print(f"  {node_type}: {dim} 维")

        print("\n📐 维度投影分析:")
        for node_type, info in feature_analysis['dimension_compression_ratios'].items():
            orig_dim = info['original_dim']
            target_dim = info['target_dim']
            ratio = info['compression_ratio']

            if ratio > 1:
                trend = "📈 扩展"
            elif ratio < 1:
                trend = "📉 压缩"
            else:
                trend = "➡️ 保持"

            print(f"  {node_type}: {orig_dim} → {target_dim} ({ratio:.2f}x) {trend}")

    print("=" * 70)


def print_model_analysis(model_analysis: Dict):
    """打印模型分析结果"""
    print("\n" + "=" * 70)
    print("🧠 XGFT模型容量分析")
    print("=" * 70)

    # XGFT融合权重分析
    if 'xgft_fusion_weights' in model_analysis:
        print("✨ XGFT层融合权重分析:")
        for layer_name, weights in model_analysis['xgft_fusion_weights'].items():
            print(f"  {layer_name}:")
            print(f"    α (全局→局部): {weights['alpha (global->local)']:.4f}")
            print(f"    β (局部→全局): {weights['beta (local->global)']:.4f}")
            print(f"    双向强度: {weights['bidirectional_strength']:.4f}")

    # 投影质量分析
    if 'projection_quality' in model_analysis:
        print("\n🔧 投影层质量:")
        for node_type, quality in model_analysis['projection_quality'].items():
            preservation = quality.get('projection_info_preservation', 0)
            print(f"  {node_type}:")
            print(f"    信息保留度: {preservation:.4f}")

            if preservation > 0.8:
                print(f"    质量评估: 🟢 优秀")
            elif preservation > 0.6:
                print(f"    质量评估: 🟡 良好")
            else:
                print(f"    质量评估: 🔴 需要优化")

    print("=" * 70)


def print_enhancement_analysis(feature_analysis: Dict, results: Optional[Dict],
                             node_feature_dims: Dict):
    """打印增强效果分析"""
    print("\n" + "=" * 70)
    print("🚀 XGFT模型 + 生物学特征增强效果分析")
    print("=" * 70)

    print("📈 技术优势:")
    print("  🎯 XGFT核心创新:")
    print("     ✅ 持续的局部-全局信息交互")
    print("     ✅ 可学习的双向融合门控 (α, β)")
    print("     ✅ 层次化特征表示学习")
    print("     ✅ 动态信息流控制")
    print("     ✅ 原始维度特征保留 - 避免信息损失")
    print("     ✅ 神经网络学习投影 - 自适应特征变换")
    print("     ✅ 多模态特征融合 - 表达谱 + LLM语义")

    print("\n  ✨ 特征增强优势:")
    enabled_features = [k for k, v in feature_analysis['feature_sources'].items() if v]

    if 'gene_expression' in enabled_features:
        gene_dim = node_feature_dims.get('Gene', 0)
        print(f"     ✅ 基因表达谱 ({gene_dim}维): 癌症特异性表达模式")
    if 'mirna_expression' in enabled_features:
        mirna_dim = node_feature_dims.get('miRNA', 0)
        print(f"     ✅ miRNA表达谱 ({mirna_dim}维): 调控活性信息")
    if 'llm_embeddings' in enabled_features:
        pathway_dim = node_feature_dims.get('Pathway', 0)
        disease_dim = node_feature_dims.get('Disease', 0)
        print(f"     ✅ LLM嵌入 (通路:{pathway_dim}维, 疾病:{disease_dim}维): 深度语义理解")

    if results:
        print(f"\n📊 实际性能结果:")

        if isinstance(results, dict) and 'fold_metrics' in results:
            # K折结果
            auc_mean = results['fold_metrics']['AUC']['mean']
            auc_std = results['fold_metrics']['AUC']['std']
            aupr_mean = results['fold_metrics']['AUPR']['mean']
            aupr_std = results['fold_metrics']['AUPR']['std']
            f1_mean = results['fold_metrics']['F1']['mean']
            f1_std = results['fold_metrics']['F1']['std']

            print(f"     AUC: {auc_mean:.4f} ± {auc_std:.4f}")
            print(f"     AUPR: {aupr_mean:.4f} ± {aupr_std:.4f}")
            print(f"     F1: {f1_mean:.4f} ± {f1_std:.4f}")

            # 集成结果
            if 'ensemble_results' in results:
                print(f"\n     集成预测:")
                print(f"       平均集成 AUC: {results['ensemble_results']['average']['AUC']:.4f}")
                print(f"       投票集成 AUC: {results['ensemble_results']['voting']['AUC']:.4f}")

            auc = auc_mean  # 用于性能评估
        else:
            # 单次训练结果
            auc = results.get('AUC', 0)
            aupr = results.get('AUPR', 0)
            f1 = results.get('F1', 0)

            print(f"     AUC: {auc:.4f}")
            print(f"     AUPR: {aupr:.4f}")
            print(f"     F1: {f1:.4f}")

        # 性能评估
        target_auc = config.experiment.performance_benchmarks['target_auc']
        acceptable_auc = config.experiment.performance_benchmarks['acceptable_auc']

        if auc >= target_auc:
            print("     🎉 优秀！达到目标性能")
        elif auc >= acceptable_auc:
            print("     📈 良好！超过可接受阈值")
        else:
            print("     ⚠️ 需要进一步优化")

    print("\n💡 XGFT技术创新点:")
    print("     🔬 局部HGT + 全局Transformer的交互式融合")
    print("     🌍 持续的双向信息交换机制")
    print("     🔀 动态的融合权重学习")
    print("     🧬 生物学先验知识的深度整合")
    print("     🤖 端到端特征学习与图神经网络的统一优化")

    print("=" * 70)


def print_experiment_comparison(results_comparison: Dict):
    """打印实验对比结果"""
    print("\n" + "=" * 70)
    print("🧪 XGFT模型实验对比分析")
    print("=" * 70)

    # 提取性能指标
    performance_data = {}
    for mode, result in results_comparison.items():
        if 'performance' in result and 'error' not in result:
            performance_data[mode] = result['performance']

    if not performance_data:
        print("❌ 没有可用的性能数据进行对比")
        return

    # 根据是否K折显示不同格式
    use_kfold = any('fold_metrics' in perf for perf in performance_data.values())

    if use_kfold:
        # K折结果表格
        print(f"{'模式':<20} {'AUC (均值±标准差)':<20} {'AUPR (均值±标准差)':<20} {'F1 (均值±标准差)':<20}")
        print("-" * 80)

        mode_names = {
            'baseline': '基线（随机特征）',
            'expression_only': '仅表达谱',
            'llm_only': '仅LLM嵌入',
            'full_biological': '完整生物学特征'
        }

        for mode, perf in performance_data.items():
            name = mode_names.get(mode, mode)
            auc_mean = perf['fold_metrics']['AUC']['mean']
            auc_std = perf['fold_metrics']['AUC']['std']
            aupr_mean = perf['fold_metrics']['AUPR']['mean']
            aupr_std = perf['fold_metrics']['AUPR']['std']
            f1_mean = perf['fold_metrics']['F1']['mean']
            f1_std = perf['fold_metrics']['F1']['std']

            print(f"{name:<20} {f'{auc_mean:.4f}±{auc_std:.4f}':<20} "
                  f"{f'{aupr_mean:.4f}±{aupr_std:.4f}':<20} "
                  f"{f'{f1_mean:.4f}±{f1_std:.4f}':<20}")

    else:
        # 单次训练结果表格
        print(f"{'模式':<20} {'AUC':<8} {'AUPR':<8} {'F1':<8} {'Accuracy':<10}")
        print("-" * 60)

        mode_names = {
            'baseline': '基线（随机特征）',
            'expression_only': '仅表达谱',
            'llm_only': '仅LLM嵌入',
            'full_biological': '完整生物学特征'
        }

        for mode, perf in performance_data.items():
            name = mode_names.get(mode, mode)
            auc = perf.get('AUC', 0)
            aupr = perf.get('AUPR', 0)
            f1 = perf.get('F1', 0)
            acc = perf.get('Accuracy', 0)

            print(f"{name:<20} {auc:<8.4f} {aupr:<8.4f} {f1:<8.4f} {acc:<10.4f}")

    print("=" * 70)


def save_experiment_results(experiment_results: Dict):
    """保存实验结果"""
    # 根据训练模式选择文件名
    training_mode = experiment_results.get('training_mode', 'single')
    results_file = config.paths.experiments_dir / f"xgft_{EXPERIMENT_MODE}_{training_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"📊 实验结果已保存: {results_file}")

    # 保存配置
    config_file = config.paths.experiments_dir / f"config_xgft_{EXPERIMENT_MODE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    config.save_config(str(config_file))


def main():
    """XGFT模型主函数"""
    print("🧬 XGFT miRNA-mRNA 相互作用预测模型")
    print("✨ 交互式局部-全局融合变换器 + 原始维度生物学特征")
    print("=" * 60)
    print(f"运行模式: {RUN_MODE}")
    print(f"实验模式: {EXPERIMENT_MODE}")
    print(f"K折训练: {'是' if USE_KFOLD else '否'}")
    if USE_KFOLD:
        print(f"折数: {N_FOLDS}")
    print(f"数据文件: {DATA_FILE}")
    print("=" * 60)

    # 更新配置
    config.data.use_kfold = USE_KFOLD
    config.data.n_folds = N_FOLDS

    # 设置实验配置
    if RUN_MODE in ['train', 'predict', 'experiment']:
        setup_experiment(EXPERIMENT_MODE)

    # 设置环境
    setup_environment()

    # 根据模式执行相应功能
    try:
        if RUN_MODE == 'train':
            result = train_mode()
        elif RUN_MODE == 'predict':
            result = predict_mode()
        elif RUN_MODE == 'experiment':
            result = experiment_mode()
        else:
            logger.error(f"❌ 未知模式: {RUN_MODE}")
            return

        # train_mode 失败时返回 (None, None) 这个元组而不是 None，
        # 只判断 is not None 会把崩溃当成功，批量脚本就会一路跑下去。
        failed = result is None or (isinstance(result, tuple) and all(r is None for r in result))
        if not failed:
            logger.info("🎉 程序执行成功完成!")
        else:
            logger.error("❌ 程序执行失败!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("⏹️ 用户中断程序执行")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 程序执行出现未预期错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()