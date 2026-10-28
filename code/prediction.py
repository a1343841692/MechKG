"""
XGFT模型预测服务模块
"""

import torch
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import json
from pathlib import Path

from config import config
from utils import ModelUtils
from data_loader import load_knowledge_graph
from models import create_model

logger = logging.getLogger(__name__)


class PredictionService:
    """XGFT模型预测服务类"""

    def __init__(self, model_path: str = None, data_path: Optional[str] = None,
                 use_ensemble: bool = False, kfold_models_dir: str = None):
        """
        初始化预测服务

        Args:
            model_path: 单个模型路径（用于非K折模型）
            data_path: 数据路径
            use_ensemble: 是否使用K折集成预测
            kfold_models_dir: K折模型目录
        """
        self.model_path = model_path
        self.data_path = data_path or config.data.kg_file
        self.use_ensemble = use_ensemble
        self.kfold_models_dir = kfold_models_dir or config.paths.kfold_models_dir

        # 初始化组件
        self.models = []  # 存储模型（单个或多个K折模型）
        self.data = None
        self.node_feature_dims = {}
        self.embeddings = None

        # 加载模型和数据
        self._load_models_and_data()

        logger.info(f"✅ 预测服务初始化完成 (集成模式: {'是' if use_ensemble else '否'})")

    def _load_models_and_data(self):
        """加载模型和数据"""
        logger.info("加载模型和数据...")

        try:
            # 加载数据
            self.data = load_knowledge_graph(self.data_path)
            self.data = self.data.to(config.device)

            # 提取特征维度
            for node_type in self.data.node_types:
                if hasattr(self.data[node_type], 'x') and self.data[node_type].x is not None:
                    self.node_feature_dims[node_type] = self.data[node_type].x.shape[1]
                else:
                    self.node_feature_dims[node_type] = config.model.embedding_dim

            if self.use_ensemble:
                # 加载K折模型
                self._load_kfold_models()
            else:
                # 加载单个模型
                self._load_single_model()

            # 预计算嵌入
            self._precompute_embeddings()

            logger.info("✅ 模型和数据加载成功")

        except Exception as e:
            logger.error(f"加载失败: {e}")
            raise

    def _load_single_model(self):
        """加载单个模型"""
        # 创建XGFT模型实例
        model = create_model(
            metadata=self.data.metadata(),
            node_feature_dims=self.node_feature_dims,
            model_type='xgft'
        )

        # 加载训练好的权重
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = ModelUtils.load_model_checkpoint(
                model, self.model_path, device=str(config.device)
            )
            logger.info(f"✅ 成功加载模型权重: {self.model_path}")
        else:
            logger.warning("⚠️ 未找到模型权重文件，使用随机初始化的模型")

        model = model.to(config.device)
        model.eval()
        self.models = [model]

    def _load_kfold_models(self):
        """加载K折模型"""
        logger.info(f"加载K折模型集合: {self.kfold_models_dir}")

        # 查找所有折的模型
        model_files = list(Path(self.kfold_models_dir).glob("fold_*_model.pth"))

        if not model_files:
            logger.error(f"未找到K折模型文件在: {self.kfold_models_dir}")
            raise FileNotFoundError(f"No fold models found in {self.kfold_models_dir}")

        # 按折数排序
        model_files.sort(key=lambda x: int(x.stem.split('_')[1]))

        for model_file in model_files:
            model = create_model(
                metadata=self.data.metadata(),
                node_feature_dims=self.node_feature_dims,
                model_type='xgft'
            )

            checkpoint = ModelUtils.load_model_checkpoint(
                model, str(model_file), device=str(config.device)
            )

            model = model.to(config.device)
            model.eval()
            self.models.append(model)

            logger.info(f"  ✅ 加载折模型: {model_file.name}")

        logger.info(f"✅ 成功加载 {len(self.models)} 个K折模型")

    def _precompute_embeddings(self):
        """预计算节点嵌入"""
        logger.info("预计算节点嵌入...")

        self.embeddings = []

        with torch.no_grad():
            for i, model in enumerate(self.models):
                x_local, x_global = model(self.data.x_dict, self.data.edge_index_dict)
                self.embeddings.append((x_local, x_global))

                if self.use_ensemble:
                    logger.info(f"  计算第 {i+1}/{len(self.models)} 个模型的嵌入")

        logger.info("✅ 嵌入预计算完成")

    def predict_pairs(self, pairs: List[Tuple[str, str]],
                      return_details: bool = False,
                      confidence_threshold: float = 0.5) -> List[Dict]:
        """
        预测miRNA-Gene对的相互作用

        Args:
            pairs: (miRNA, Gene) 名称对列表
            return_details: 是否返回详细信息
            confidence_threshold: 置信度阈值

        Returns:
            预测结果列表
        """
        results = []

        for mirna_name, gene_name in pairs:
            try:
                prediction = self._predict_single_pair(mirna_name, gene_name)

                result = {
                    'mirna': mirna_name,
                    'gene': gene_name,
                    'prediction_score': prediction['score'],
                    'prediction_label': prediction['label'],
                    'confidence': prediction['confidence']
                }

                if return_details:
                    result.update(prediction.get('details', {}))

                results.append(result)

            except Exception as e:
                logger.warning(f"预测失败 {mirna_name}-{gene_name}: {e}")
                results.append({
                    'mirna': mirna_name,
                    'gene': gene_name,
                    'error': str(e),
                    'prediction_score': None,
                    'prediction_label': None,
                    'confidence': None
                })

        return results

    def _predict_single_pair(self, mirna_name: str, gene_name: str) -> Dict:
        """预测单个miRNA-Gene对"""

        # 获取节点索引
        mirna_idx = self._get_node_index(mirna_name, 'miRNA')
        gene_idx = self._get_node_index(gene_name, 'Gene')

        if mirna_idx is None:
            raise ValueError(f"未找到miRNA: {mirna_name}")
        if gene_idx is None:
            raise ValueError(f"未找到Gene: {gene_name}")

        # 预测
        with torch.no_grad():
            mirna_indices = torch.tensor([mirna_idx], device=config.device)
            gene_indices = torch.tensor([gene_idx], device=config.device)

            if self.use_ensemble:
                # 集成预测
                all_scores = []

                for i, (model, (x_local, x_global)) in enumerate(zip(self.models, self.embeddings)):
                    score = model.predict_links(
                        x_local, x_global, mirna_indices, gene_indices
                    ).item()
                    all_scores.append(score)

                # 平均集成
                score = np.mean(all_scores)
                score_std = np.std(all_scores)

                details = {
                    'ensemble_scores': all_scores,
                    'score_mean': score,
                    'score_std': score_std,
                    'n_models': len(self.models)
                }

            else:
                # 单模型预测
                x_local, x_global = self.embeddings[0]
                score = self.models[0].predict_links(
                    x_local, x_global, mirna_indices, gene_indices
                ).item()

                details = {
                    'model_type': 'XGFT',
                    'single_model': True
                }

        # 计算置信度和标签
        threshold = 0.5
        label = 1 if score >= threshold else 0

        # 置信度：距离0.5的距离（归一化到0-1）
        confidence = abs(score - 0.5) * 2

        result = {
            'score': score,
            'label': label,
            'confidence': confidence,
            'details': details
        }

        return result

    def _get_node_index(self, node_name: str, node_type: str) -> Optional[int]:
        """获取节点索引"""
        # 检查是否有节点映射
        if hasattr(self.data, 'node_mapping'):
            if (node_type in self.data.node_mapping and
                    node_name in self.data.node_mapping[node_type]):
                return self.data.node_mapping[node_type][node_name]

        # 备用方法：遍历查找
        logger.warning(f"未找到节点映射，尝试遍历查找: {node_name}")
        return None

    def batch_predict(self, mirna_list: List[str], gene_list: List[str]) -> pd.DataFrame:
        """
        批量预测所有miRNA-Gene组合

        Args:
            mirna_list: miRNA列表
            gene_list: 基因列表

        Returns:
            预测结果DataFrame
        """
        logger.info(f"批量预测: {len(mirna_list)} miRNAs × {len(gene_list)} Genes")

        pairs = [(mirna, gene) for mirna in mirna_list for gene in gene_list]
        results = self.predict_pairs(pairs, return_details=True)

        # 转换为DataFrame
        df = pd.DataFrame(results)

        # 按分数排序
        if 'prediction_score' in df.columns:
            df = df.sort_values('prediction_score', ascending=False)

        return df

    def predict_top_interactions(self, mirna: str = None, gene: str = None,
                               top_k: int = 10) -> pd.DataFrame:
        """
        预测给定miRNA或基因的Top-K相互作用

        Args:
            mirna: miRNA名称（如果指定，预测该miRNA的所有靶基因）
            gene: 基因名称（如果指定，预测调控该基因的所有miRNA）
            top_k: 返回Top-K个结果

        Returns:
            Top-K预测结果
        """
        if mirna and gene:
            raise ValueError("不能同时指定miRNA和基因")

        if not mirna and not gene:
            raise ValueError("必须指定miRNA或基因中的一个")

        if mirna:
            # 获取所有基因
            all_genes = list(self.data.node_mapping.get('Gene', {}).keys())
            pairs = [(mirna, g) for g in all_genes]
            logger.info(f"预测 {mirna} 的Top-{top_k}个靶基因")
        else:
            # 获取所有miRNA
            all_mirnas = list(self.data.node_mapping.get('miRNA', {}).keys())
            pairs = [(m, gene) for m in all_mirnas]
            logger.info(f"预测调控 {gene} 的Top-{top_k}个miRNA")

        # 批量预测
        results = self.predict_pairs(pairs)

        # 转换为DataFrame并排序
        df = pd.DataFrame(results)
        df = df[df['prediction_score'].notna()]  # 移除失败的预测
        df = df.sort_values('prediction_score', ascending=False).head(top_k)

        return df

    def save_predictions(self, predictions: pd.DataFrame, save_path: str):
        """保存预测结果"""
        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存CSV
        csv_path = save_path.replace('.json', '.csv')
        predictions.to_csv(csv_path, index=False)
        logger.info(f"📁 预测结果已保存为CSV: {csv_path}")

        # 保存JSON（包含更多细节）
        if save_path.endswith('.json'):
            predictions_dict = predictions.to_dict('records')

            # 添加元信息
            result_data = {
                'metadata': {
                    'model_type': 'XGFT',
                    'use_ensemble': self.use_ensemble,
                    'n_models': len(self.models) if self.use_ensemble else 1,
                    'timestamp': datetime.now().isoformat(),
                    'n_predictions': len(predictions)
                },
                'predictions': predictions_dict
            }

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            logger.info(f"📁 详细预测结果已保存为JSON: {save_path}")