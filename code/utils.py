"""
工具函数模块 - 增强版，支持tqdm兼容的日志输出
"""

import copy
import torch
import torch.nn as nn
import numpy as np
import random
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    roc_curve
)
from tqdm import tqdm
import warnings

# 从config导入全局配置
from config import config

warnings.filterwarnings('ignore')


def set_seed(seed: int = 42):
    """设置随机种子确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TqdmLoggingHandler(logging.Handler):
    """一个将日志记录重定向到 tqdm.write() 的处理器

    这个处理器确保日志输出不会干扰tqdm进度条的显示
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(log_dir: str = 'logs', level: str = 'INFO'):
    """设置日志记录 - ✨ tqdm兼容的最终版本"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    # 定义所有处理器共享的格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 获取根logger，并移除所有已存在的处理器，防止重复
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    # 清空已有处理器
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 1. 文件处理器：保持不变，将日志写入文件
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 2. 控制台处理器：✨ 使用我们新的 TqdmLoggingHandler
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(formatter)
    root_logger.addHandler(tqdm_handler)

    return logging.getLogger(__name__)  # 返回一个具名的 logger 实例供其他模块使用


class EarlyStopping:
    """早停类"""

    def __init__(self, patience: int = 15, min_delta: float = 0.0,
                 mode: str = 'max', restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

        if mode == 'max':
            self.is_better = lambda score, best: score > best + min_delta
        else:
            self.is_better = lambda score, best: score < best - min_delta

    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())  # [FIX] 原 .copy() 为浅拷贝，恢复的并非最佳权重
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())  # [FIX] 原 .copy() 为浅拷贝，恢复的并非最佳权重
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)

        return self.early_stop


class MetricsCalculator:
    """指标计算器"""
    @staticmethod
    def _calibrate_reported_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
        """
        对关键指标应用校准，以提高报告的稳定性。
        此调整有助于协调相关指标，并对超出预期性能基准的分数进行正则化。
        """
        # 检查是否在全局配置中启用了校准
        if not config.experiment.reporting_config.get('enable_calibration', False):
            return metrics

        cal_target = config.experiment.reporting_config.get('calibration_target', 0.92)
        harm_strength = config.experiment.reporting_config.get('harmonization_strength', 0.35)

        # 仅当AUC和AUPR都存在时应用
        if 'AUC' not in metrics or 'AUPR' not in metrics:
            return metrics

        auc, aupr = metrics['AUC'], metrics['AUPR']

        # 1. 通过将AUC和AUPR移向它们的均值来进行协调
        # 这可以减少这两个相关指标之间的报告差异。
        mean_score = (auc + aupr) / 2.0
        harmonized_auc = auc * (1 - harm_strength) + mean_score * harm_strength
        harmonized_aupr = aupr * (1 - harm_strength) + mean_score * harm_strength

        # 2. 应用软上限来正则化可能过于乐观的分数
        def apply_ceiling(score, ceiling):
            """使用基于tanh的正则化项应用软上限。"""
            if score > ceiling:
                # 这个非线性项会温和地拉低分数，对于较大的超额部分效果更强。
                return ceiling - np.tanh((score - ceiling) * 15) * 0.003
            return score

        # 使用校准后的值更新指标
        metrics['AUC'] = apply_ceiling(harmonized_auc, cal_target)
        metrics['AUPR'] = apply_ceiling(harmonized_aupr, cal_target)

        return metrics

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          y_scores: np.ndarray) -> Dict[str, float]:
        """计算所有评估指标"""

        # 基础指标
        auc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)

        # 混淆矩阵
        # 修正：处理只有一个类别的情况
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        except ValueError:
             # 当y_true或y_pred只包含一个类别时，ravel()会失败
            tn, fp, fn, tp = 0, 0, 0, 0
            if len(np.unique(y_true)) == 1:
                if y_true[0] == 0:
                    tn = len(y_true)
                else:
                    tp = len(y_true)

        # 计算各项指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # 敏感性和特异性
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics = {
            'AUC': auc,
            'AUPR': aupr,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }

        # ✨✨✨ 关键修复：在这里调用校准函数 ✨✨✨
        # [FIX] 不再调用 _calibrate_reported_metrics：该函数直接改写 AUC/AUPR 数值，不属于概率校准
        return metrics


class AdvancedVisualization:
    """可视化工具"""

    def __init__(self, save_dir: str = 'plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 设置绘图风格
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')

    def plot_training_curves(self, history: Dict[str, List[float]],
                             save_name: str = 'training_curves.png'):
        """绘制训练曲线"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)

        # 损失曲线
        if 'train_loss' in history:
            axes[0, 0].plot(history['train_loss'], 'b-', label='Train')
            if 'val_loss' in history:
                axes[0, 0].plot(history['val_loss'], 'r-', label='Validation')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # AUC曲线
        if 'val_auc' in history:
            axes[0, 1].plot(history['val_auc'], 'g-')
            axes[0, 1].set_title('Validation AUC')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('AUC')
            axes[0, 1].grid(True, alpha=0.3)

        # F1分数
        if 'val_f1' in history:
            axes[1, 0].plot(history['val_f1'], 'purple')
            axes[1, 0].set_title('Validation F1')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1')
            axes[1, 0].grid(True, alpha=0.3)

        # 学习率
        if 'lr' in history:
            axes[1, 1].plot(history['lr'], 'red')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return save_path


class ModelUtils:
    """模型相关工具函数"""

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """计算模型参数数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }

    @staticmethod
    def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                              epoch: int, metrics: Dict[str, float],
                              filepath: str, **kwargs):
        """保存模型检查点"""

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)

    @staticmethod
    def load_model_checkpoint(model: nn.Module, filepath: str,
                              optimizer: Optional[torch.optim.Optimizer] = None,
                              device: str = 'cpu') -> Dict:
        """加载模型检查点"""

        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint


def print_system_info():
    """打印系统信息"""
    print("系统信息:")
    print("=" * 50)
    print(f"PyTorch版本: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA可用: 是")
        print(f"GPU数量: {torch.cuda.device_count()}")
    else:
        print("CUDA可用: 否")