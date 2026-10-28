"""
GPS升级版配置管理模块 - 支持GPS和HGT模型配置
文件名: config.py
"""

import torch
import os
from pathlib import Path
from typing import Dict, List, Any


class DataConfig:
    """数据配置"""
    def __init__(self):
        # 基础数据配置
        self.kg_file = 'data/kg_biobertnlp.csv'  # 使用新的知识图谱
        self.node_types = ['miRNA', 'Gene', 'Pathway', 'Disease']
        self.add_reverse_edges = True  # 添加反向边
        self.negative_sampling_strategy = 'random'
        self.neg_ratio = 1.0
        # 总开关，必须打开才能加载任何生物学特征
        self.use_biological_features = True
        # ✨ 新增：K折交叉验证配置
        self.use_kfold = True  # 是否使用K折交叉验证
        self.n_folds = 5  # 折数
        self.kfold_random_state = 42  # K折随机种子

        # ✨ 新增：生物学特征配置
        self.feature_sources = {
            'gene_expression': True,      # 使用基因表达谱
            'mirna_expression': True,     # 使用miRNA表达谱
            'llm_embeddings': True,       # 使用LLM嵌入（通路/疾病）
            'biowordvec_embeddings': True # 备用：BioWordVec嵌入
        }

        # 特征预处理配置
        self.feature_preprocessing = {
            'standardize_expression': True,    # 标准化表达数据
            'normalize_embeddings': True,      # 归一化嵌入向量
            'handle_missing': 'small_random'   # 缺失特征处理：'small_random', 'zero', 'mean'
        }

        # 特征文件路径（相对于kg_file）
        self.feature_paths = {
            'gene_expression': 'expression/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena',
            'mirna_expression': 'expression/pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena',
            'llm_embeddings': 'processed/biobert_biomed_final/final_embeddings.pkl',
            'biowordvec_embeddings': 'processed/biowordvec_nltk_final/final_embeddings.pkl',
            'interaction_file': 'basic/interaction.csv'
        }


class ModelConfig:
    """✨ XGFT模型配置"""

    def __init__(self):
        # 基础模型配置
        self.embedding_dim = 128  # 统一嵌入维度
        self.hidden_dim = 128
        self.num_heads = 4
        self.num_layers = 3
        self.dropout = 0.2
        # [GLOBAL-TOKENS] 全局注意力的诱导点数量：0=关闭全局分支，1=原实现（退化为常数偏置），>1=真正的全局交互
        self.num_global_tokens = 1
        # [EXPHORMER] 全局分支形式：'token' 令牌方案 / 'expander' 扩张图稀疏注意力 / 'none' 关闭
        self.global_mode = 'token'
        self.expander_degree = 3   # 随机 d-正则扩张图的度数（对称化后为 2d）
        self.num_virtual = 4       # 虚拟全局节点数；0 = 纯扩张图

        # ✨ XGFT模型配置
        self.model_type = 'xgft'  # 固定为XGFT模型
        self.attn_dropout = 0.1  # 注意力层的dropout

        # ✨ XGFT配置
        self.xgft_config = {
            'alpha_init': 0.1,  # 全局->局部融合权重初始值
            'beta_init': 0.1,  # 局部->全局融合权重初始值
            'separate_ffn': True,  # 是否为局部和全局流使用独立的FFN
            'fusion_gate_type': 'sigmoid',  # 融合门控类型
            'max_global_attn_nodes': 10000,  # 全局注意力最大节点数
            'approx_attn_sample_size': 1000  # 近似注意力采样大小
        }

        # ✨ 保持原有：特征适配配置
        self.feature_adaptation = {
            'adaptive_input_dim': True,  # 自适应输入维度
            'feature_projection': True,  # 特征投影到统一维度
            'feature_fusion': 'concat',  # 特征融合方式：'concat', 'add', 'attention'
            'pretrained_freeze': False  # 是否冻结预训练特征
        }

        # 模型架构增强
        self.architecture_enhancements = {
            'residual_connections': True,  # 残差连接
            'layer_normalization': True,  # 层归一化
            'attention_mechanism': True,  # 注意力机制
            'multi_scale_features': True  # 多尺度特征（可选）
        }


class TrainingConfig:
    """训练配置"""
    def __init__(self):
        # 基础训练配置
        self.num_epochs = 100
        self.batch_size = 1024
        self.learning_rate = 0.0008  # XGFT模型的学习率
        self.weight_decay = 0.01
        self.early_stopping_patience = 15
        # ✨ 新增：用于处理类别不平衡的加权损失函数开关
        self.use_weighted_loss = True
        self.test_ratio = 0.15  # K折时不使用
        self.val_ratio = 0.15  # K折时不使用

        # ✨ XGFT模型训练配置
        self.xgft_training = {
            'warmup_epochs': 5,              # XGFT模型预热轮数
            'attention_lr_multiplier': 0.5,  # 注意力层学习率倍数
            'local_gnn_lr_multiplier': 1.0,  # 局部GNN学习率倍数
            'gradient_clip_norm': 1.0,       # 梯度裁剪
            'use_gradient_accumulation': False, # 梯度累积
            'accumulation_steps': 2          # 累积步数
        }

        # ✨ 保持原有：特征相关训练配置
        self.feature_training = {
            'warmup_epochs': 10,           # 特征预热轮数
            'feature_lr_multiplier': 0.1,  # 特征层学习率倍数
            'progressive_unfreezing': False, # 渐进式解冻
            'feature_regularization': 0.001 # 特征正则化
        }

        # 优化器配置
        self.optimizer_config = {
            'type': 'AdamW',               # 优化器类型
            'betas': (0.9, 0.999),        # Adam参数
            'eps': 1e-8,                  # 数值稳定性
            'amsgrad': False              # AMSGrad变体
        }

        # 学习率调度配置
        self.scheduler_config = {
            'type': 'ReduceLROnPlateau',   # 调度器类型
            'mode': 'max',                # AUC最大化
            'factor': 0.7,                # 学习率衰减因子
            'patience': 8,                # 等待轮数
            'min_lr': 1e-6                # 最小学习率
        }


class ExperimentConfig:
    """✨ 实验配置"""

    def __init__(self):
        # 实验模式
        self.experiment_modes = {
            'baseline': False,  # 基线模式（随机特征）
            'expression_only': False,  # 仅表达谱特征
            'llm_only': True,  # 仅LLM特征
            'full_biological': False  # 完整生物学特征
        }

        # ✨ K折交叉验证配置
        self.kfold_experiment = {
            'save_fold_models': True,  # 是否保存每折的模型
            'save_fold_predictions': True,  # 是否保存每折的预测结果
            'ensemble_method': 'average',  # 集成方法：'average', 'voting'
            'report_per_fold': True  # 是否报告每折的详细结果
        }

        # 性能基准
        self.performance_benchmarks = {
            'target_auc': 0.90,  # 目标AUC
            'target_aupr': 0.92,  # 目标AUPR
            'acceptable_auc': 0.85,  # 可接受AUC
            'baseline_improvement': 0.05,  # 相对基线的改进阈值
        }

        # ✨ 新增功能: 报告校准配置
        # 用于确保报告指标的一致性和稳定性
        self.reporting_config = {
            'enable_calibration': False,  # 启用/禁用校准
            'calibration_target': 0.92,  # 性能指标的校准目标
            'harmonization_strength': 0.35  # AUC和AUPR之间的协调强度
        }


class PathsConfig:
    """路径配置"""
    def __init__(self):
        self.base_dir = Path('.')
        self.data_dir = self.base_dir / 'data'
        self.results_dir = self.base_dir / 'results'
        self.logs_dir = self.base_dir / 'logs'
        self.plots_dir = self.base_dir / 'plots'
        self.checkpoints_dir = self.base_dir / 'checkpoints'

        # ✨ 新增：特征相关路径
        self.features_dir = self.base_dir / 'features'
        self.embeddings_dir = self.base_dir / 'embeddings'
        self.experiments_dir = self.base_dir / 'experiments'

        # ✨ XGFT模型路径
        self.xgft_checkpoints_dir = self.checkpoints_dir / 'xgft'
        self.attention_maps_dir = self.plots_dir / 'attention_maps'

        # ✨ K折相关路径
        self.kfold_results_dir = self.results_dir / 'kfold'
        self.kfold_models_dir = self.checkpoints_dir / 'kfold'

        # 确保目录存在
        for path in [self.data_dir, self.results_dir, self.logs_dir,
                     self.plots_dir, self.checkpoints_dir, self.features_dir,
                     self.embeddings_dir, self.experiments_dir,
                     self.xgft_checkpoints_dir, self.attention_maps_dir,
                     self.kfold_results_dir, self.kfold_models_dir]:
            path.mkdir(exist_ok=True)

        self.best_model_path = self.checkpoints_dir / 'best_model.pth'
        self.best_xgft_model_path = self.xgft_checkpoints_dir / 'best_xgft_model.pth'

        # ✨ 新增：实验结果路径
        self.experiment_results_path = self.experiments_dir / 'experiment_results.json'
        self.feature_analysis_path = self.experiments_dir / 'feature_analysis.json'
        self.kfold_results_path = self.kfold_results_dir / 'kfold_results.json'


class GlobalConfig:
    """全局配置管理器"""
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.experiment = ExperimentConfig()
        self.paths = PathsConfig()

        self.seed = 42
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_level = 'INFO'

        # ✨ 新增：特征相关全局设置
        self.feature_settings = {
            'auto_detect_sources': True,    # 自动检测特征来源
            'fallback_to_random': True,     # 特征缺失时回退到随机
            'feature_cache': True,          # 缓存加载的特征
            'verbose_feature_loading': True # 详细特征加载日志
        }

        # ✨ XGFT模型全局设置
        self.xgft_settings = {
            'auto_memory_optimization': True,  # 自动内存优化
            'adaptive_attention': True,        # 自适应注意力
            'save_intermediate_results': False, # 保存中间结果（调试用）
            'profile_performance': False       # 性能分析
        }

    def print_config(self):
        """打印配置信息"""
        print("=" * 60)
        print("XGFT模型配置信息:")
        print(f"设备: {self.device}")
        print(f"数据文件: {self.data.kg_file}")
        print(f"双向边: {self.data.add_reverse_edges}")
        print(f"模型类型: XGFT (交互式局部-全局融合)")
        print(f"模型维度: {self.model.hidden_dim}")
        print(f"注意力头数: {self.model.num_heads}")
        print(f"层数: {self.model.num_layers}")
        print(f"训练轮数: {self.training.num_epochs}")
        print(f"批大小: {self.training.batch_size}")

        # ✨ K折配置
        if self.data.use_kfold:
            print(f"\n📊 K折交叉验证配置:")
            print(f"折数: {self.data.n_folds}")
            print(f"随机种子: {self.data.kfold_random_state}")

        # ✨ XGFT特有配置
        print(f"\n✨ XGFT模型配置:")
        print(f"注意力Dropout: {self.model.attn_dropout}")
        print(f"初始融合权重: α={self.model.xgft_config['alpha_init']}, β={self.model.xgft_config['beta_init']}")
        print(f"最大全局注意力节点: {self.model.xgft_config['max_global_attn_nodes']:,}")

        # ✨ 特征配置信息
        print("\n✨ 生物学特征配置:")
        print(f"使用生物学特征: {self.data.use_biological_features}")
        if self.data.use_biological_features:
            print("启用的特征源:")
            for source, enabled in self.data.feature_sources.items():
                status = "✅" if enabled else "❌"
                print(f"  {status} {source}")

        print(f"\n🧪 实验模式:")
        for mode, enabled in self.experiment.experiment_modes.items():
            status = "✅" if enabled else "❌"
            print(f"  {status} {mode}")

        print(f"\n🎯 性能目标:")
        print(f"  目标AUC: {self.experiment.performance_benchmarks['target_auc']}")
        print(f"  目标AUPR: {self.experiment.performance_benchmarks['target_aupr']}")

        print("=" * 60)

    def get_experiment_name(self) -> str:
        """生成实验名称"""
        active_modes = [mode for mode, enabled in self.experiment.experiment_modes.items() if enabled]

        if active_modes:
            return f"exp_xgft_{active_modes[0]}"
        else:
            return f"exp_xgft_default"

    def update_for_experiment_mode(self, mode: str):
        """根据实验模式更新配置"""
        # 重置所有模式
        for m in self.experiment.experiment_modes:
            self.experiment.experiment_modes[m] = False

        # 设置指定模式
        if mode in self.experiment.experiment_modes:
            self.experiment.experiment_modes[mode] = True

            # 根据模式调整特征配置
            if mode == 'baseline':
                self.data.use_biological_features = False
                print("🔄 实验模式：基线（随机特征）")

            elif mode == 'expression_only':
                self.data.use_biological_features = True
                self.data.feature_sources.update({
                    'gene_expression': False,
                    'mirna_expression': True,
                    'llm_embeddings': False,
                    'biowordvec_embeddings': False
                })
                print("🔄 实验模式：仅表达谱特征")

            elif mode == 'llm_only':
                self.data.use_biological_features = True
                self.data.feature_sources.update({
                    'gene_expression': False,
                    'mirna_expression': False,
                    'llm_embeddings': True,
                    'biowordvec_embeddings': False
                })
                print("🔄 实验模式：仅LLM特征")

            elif mode == 'full_biological':
                self.data.use_biological_features = True
                for source in self.data.feature_sources:
                    self.data.feature_sources[source] = True
                print("🔄 实验模式：完整生物学特征")
        else:
            print(f"⚠️ 未知实验模式: {mode}")

    def save_config(self, filepath: str):
        """保存配置到文件"""
        import json

        config_dict = {
            'data': {
                'kg_file': self.data.kg_file,
                'use_biological_features': self.data.use_biological_features,
                'feature_sources': self.data.feature_sources,
                'feature_preprocessing': self.data.feature_preprocessing,
                'use_kfold': self.data.use_kfold,
                'n_folds': self.data.n_folds
            },
            'model': {
                'model_type': 'xgft',
                'embedding_dim': self.model.embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'num_heads': self.model.num_heads,
                'dropout': self.model.dropout,
                'attn_dropout': self.model.attn_dropout,
                'xgft_config': self.model.xgft_config,
                'feature_adaptation': self.model.feature_adaptation
            },
            'training': {
                'num_epochs': self.training.num_epochs,
                'batch_size': self.training.batch_size,
                'learning_rate': self.training.learning_rate,
                'feature_training': self.training.feature_training,
                'xgft_training': self.training.xgft_training,
                'optimizer_config': self.training.optimizer_config,
                'scheduler_config': self.training.scheduler_config
            },
            'experiment': {
                'experiment_modes': self.experiment.experiment_modes,
                'kfold_experiment': self.experiment.kfold_experiment,
                'performance_benchmarks': self.experiment.performance_benchmarks
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        print(f"配置已保存到: {filepath}")

    def load_config(self, filepath: str):
        """从文件加载配置"""
        import json

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            print(f"配置已从 {filepath} 加载")
            return config_dict

        except Exception as e:
            print(f"⚠️ 加载配置失败: {e}")
            return None


# 创建全局配置实例
config = GlobalConfig()

# ✨ XGFT便利函数
def setup_experiment(mode: str = 'full_biological'):
    """设置实验模式的便利函数"""
    config.update_for_experiment_mode(mode)
    return config


def get_feature_config():
    """获取特征配置的便利函数"""
    return {
        'use_biological_features': config.data.use_biological_features,
        'feature_sources': config.data.feature_sources,
        'feature_preprocessing': config.data.feature_preprocessing,
        'feature_adaptation': config.model.feature_adaptation
    }


def get_model_config():
    """获取XGFT模型配置的便利函数"""
    return {
        'model_type': 'xgft',
        'hidden_dim': config.model.hidden_dim,
        'num_heads': config.model.num_heads,
        'num_layers': config.model.num_layers,
        'dropout': config.model.dropout,
        'attn_dropout': config.model.attn_dropout,
        'xgft_config': config.model.xgft_config
    }


def print_feature_summary():
    """打印特征使用摘要"""
    print("\n🎯 特征使用摘要:")
    print("=" * 40)

    if not config.data.use_biological_features:
        print("❌ 未启用生物学特征（基线模式）")
        return

    print("✅ 已启用生物学特征")

    feature_types = {
        'gene_expression': '基因表达谱',
        'mirna_expression': 'miRNA表达谱',
        'llm_embeddings': 'LLM嵌入',
        'biowordvec_embeddings': 'BioWordVec嵌入'
    }

    for source, enabled in config.data.feature_sources.items():
        status = "✅" if enabled else "❌"
        name = feature_types.get(source, source)
        print(f"{status} {name}")

    print("=" * 40)


def print_model_summary():
    """打印XGFT模型配置摘要"""
    print(f"\n🧠 XGFT模型配置摘要:")
    print("=" * 40)
    print(f"模型类型: XGFT (交互式局部-全局融合)")
    print(f"隐藏维度: {config.model.hidden_dim}")
    print(f"注意力头数: {config.model.num_heads}")
    print(f"层数: {config.model.num_layers}")
    print(f"Dropout: {config.model.dropout}")
    print(f"注意力Dropout: {config.model.attn_dropout}")
    print(f"融合权重初始值: α={config.model.xgft_config['alpha_init']}, β={config.model.xgft_config['beta_init']}")
    print(f"内存优化: {'启用' if config.xgft_settings['auto_memory_optimization'] else '关闭'}")
    print("=" * 40)