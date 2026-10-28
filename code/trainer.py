"""
K折交叉验证训练模块 - 支持XGFT模型
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional, List
import numpy as np
from tqdm import tqdm
import logging
import time
import json
import copy
from pathlib import Path
from config import config
from utils import EarlyStopping, MetricsCalculator, AdvancedVisualization, ModelUtils
from data_loader import (create_kfold_splits, create_train_test_split_with_kfold,
                         build_message_graph, assert_disjoint_splits, is_target_edge_type)
from models import create_model

logger = logging.getLogger(__name__)


def _endpoint_matched():
    """Standalone release: same endpoint-matched implementation."""
    from sampling import endpoint_matched
    return endpoint_matched


class KFoldTrainer:
    """K折交叉验证训练器 - 专为XGFT模型设计"""

    def __init__(self, data, device: Optional[str] = None, n_folds: int = 5):
        self.data = data.to(device or config.device)
        self.device = device or config.device
        self.n_folds = n_folds

        # 创建K折数据分割（带独立测试集）
        logger.info(f"创建{n_folds}折交叉验证数据...")
        self.kfold_data, self.test_data = create_train_test_split_with_kfold(
            self.data,
            test_ratio=config.training.test_ratio,
            n_folds=n_folds
        )
        assert_disjoint_splits(self.kfold_data, self.test_data)  # [FIX] 划分互斥检查
        self._save_splits()
        self.pair_table = self._load_pair_table()  # [EARLY-FUSION] 未配置时为 None

        # 训练组件
        self.criterion = nn.BCELoss()
        #self.criterion = None
        # 可视化工具
        self.visualizer = AdvancedVisualization(config.paths.plots_dir)

        # K折结果存储
        self.fold_results = []
        self.fold_models = []
        self.fold_histories = []

        logger.info(f"✅ K折训练器初始化完成")
        logger.info(f"   设备: {self.device}")
        logger.info(f"   折数: {n_folds}")
        logger.info(f"   独立测试集大小: {len(self.test_data['labels'])}")

    def train_fold(self, fold: int, train_data: Dict, val_data: Dict) -> Tuple[nn.Module, Dict, Dict]:
        """训练单个折"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 开始训练第 {fold+1}/{self.n_folds} 折")
        logger.info(f"{'='*60}")
        logger.info(f"   训练样本数: {len(train_data['labels']):,}")
        logger.info(f"   验证样本数: {len(val_data['labels']):,}")
        logger.info(f"   批次大小: {config.training.batch_size}")
        logger.info(f"   学习率: {config.training.learning_rate}")
        logger.info(f"   最大轮数: {config.training.num_epochs}")
        logger.info(f"   早停耐心: {config.training.early_stopping_patience}")

        # 提取特征维度
        node_feature_dims = {}
        for node_type in self.data.node_types:
            if hasattr(self.data[node_type], 'x') and self.data[node_type].x is not None:
                node_feature_dims[node_type] = self.data[node_type].x.shape[1]
            else:
                node_feature_dims[node_type] = config.model.embedding_dim

        # 创建新的XGFT模型实例
        model = create_model(
            metadata=self.data.metadata(),
            node_feature_dims=node_feature_dims,
            model_type='xgft'
        ).to(self.device)
        from mechkg import configure_from_data
        configure_from_data(model, self.data)                     # [MECHKG] 需要节点名称的设置（未启用时无操作）

        # 设置优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.7,
            patience=5,
            min_lr=1e-6
        )

        # 早停
        early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            mode='max'
        )

        # [FIX] 本折消息图：仅含本折训练正例作为靶向边，验证/测试对不可见
        self.graph = build_message_graph(self.data.edge_index_dict, train_data, [val_data, self.test_data])
        # [EVIDENCE-GATE] 证据可得性表 / [DEBIAS] 度数回归标签，都只用本折训练边
        self.evidence_table = self._build_evidence_table(train_data)
        self.degree_targets = self._build_degree_targets(train_data)
        # [NEG-MATCH] 本折 hard 验证集（端点度匹配负例）。每轮记录，是否驱动早停由配置决定。
        self.hard_val = self._build_hard_val(val_data, fold)

        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_aupr': [],
            'val_f1': [],
            'val_auc_hard': [],   # [NEG-MATCH] 度匹配验证集上的 AUC
            'lr': []
        }

        best_val_auc = 0.0
        best_epoch = 0

        # 训练循环
        pbar = tqdm(range(config.training.num_epochs),
                   desc=f"Fold {fold+1}/{self.n_folds}",
                   file=sys.stdout)

        for epoch in pbar:
            # 训练阶段
            model.train()
            # [NEG] 可选：每轮重新采样训练负样本（验证/测试集不变）
            epoch_data = train_data
            if getattr(config.training, 'dynamic_negatives', False):
                epoch_data = self._resample_train_negatives(train_data, val_data, fold, epoch)
            train_loss = self._train_epoch(model, optimizer, epoch_data)

            # 验证阶段
            val_metrics = self._evaluate(model, val_data)
            # [NEG-MATCH] 同步在 hard 验证集上评估（默认只记录，不驱动早停）
            hard_auc = self._evaluate(model, self.hard_val)['AUC'] if self.hard_val else float('nan')

            # 更新历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_auc'].append(val_metrics['AUC'])
            history['val_aupr'].append(val_metrics['AUPR'])
            history['val_f1'].append(val_metrics['F1'])
            history['val_auc_hard'].append(hard_auc)
            history['lr'].append(optimizer.param_groups[0]['lr'])

            # [NEG-MATCH] 选型准则：'standard'(默认，与既有全部实验一致) 或 'hard'
            sel = (hard_auc if getattr(config.training, 'early_stop_metric', 'standard') == 'hard'
                   and hard_auc == hard_auc else val_metrics['AUC'])

            # 学习率调度
            scheduler.step(sel)

            # 更新最佳结果
            if sel > best_val_auc:
                best_val_auc = sel
                best_model_state = copy.deepcopy(model.state_dict())  # [FIX] 原为引用，后续训练会覆盖“最佳”权重
                best_epoch = epoch + 1
                improvement_indicator = "📈"
            else:
                improvement_indicator = ""

            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{val_metrics['loss']:.4f}",
                'AUC': f"{val_metrics['AUC']:.4f}{improvement_indicator}",
                'AUPR': f"{val_metrics['AUPR']:.4f}",
                'hAUC': f"{hard_auc:.4f}",   # [NEG-MATCH] 度匹配验证集
                'LR': f"{optimizer.param_groups[0]['lr']:.1e}"
            })

            # 每5个epoch输出详细验证指标
            if (epoch + 1) % 5 == 0:
                pbar.write(f"\n📊 Fold {fold+1} - Epoch {epoch+1} 验证集指标:")
                pbar.write(f"   Loss:      {val_metrics['loss']:.4f}")
                pbar.write(f"   AUC:       {val_metrics['AUC']:.4f} {'(Best!)' if val_metrics['AUC'] == best_val_auc else ''}")
                pbar.write(f"   AUPR:      {val_metrics['AUPR']:.4f}")
                pbar.write(f"   F1:        {val_metrics['F1']:.4f}")
                pbar.write(f"   Accuracy:  {val_metrics['Accuracy']:.4f}")
                pbar.write(f"   Precision: {val_metrics['Precision']:.4f}")
                pbar.write(f"   Recall:    {val_metrics['Recall']:.4f}")
                pbar.write(f"   LR:        {optimizer.param_groups[0]['lr']:.1e}")
                pbar.write(f"   训练Loss:  {train_loss:.4f}\n")

            # 早停检查
            if early_stopping(sel, model):
                pbar.write(f"\n⚠️ 早停在第 {epoch+1} 轮")
                pbar.write(f"   最佳验证AUC: {best_val_auc:.4f}")
                break

        # 恢复最佳模型
        model.load_state_dict(best_model_state)

        # 在测试集上评估
        test_metrics = self._evaluate(model, self.test_data)

        # [MECHKG] 本折验证集上各机制通道的平均贡献（描述性，不参与选型）
        if getattr(model, 'mechkg', None) is not None:
            self._log_mechkg_contrib(model, val_data, fold)

        # 训练总结
        logger.info(f"\n📋 第 {fold+1} 折训练总结:")
        logger.info(f"   训练轮数: {len(history['train_loss'])}")
        logger.info(f"   最佳Epoch: {best_epoch}")
        logger.info(f"   最佳验证AUC: {best_val_auc:.4f}")
        logger.info(f"   最终学习率: {optimizer.param_groups[0]['lr']:.1e}")

        logger.info(f"\n🎯 测试集性能:")
        logger.info(f"   AUC:       {test_metrics['AUC']:.4f}")
        logger.info(f"   AUPR:      {test_metrics['AUPR']:.4f}")
        logger.info(f"   F1:        {test_metrics['F1']:.4f}")
        logger.info(f"   Accuracy:  {test_metrics['Accuracy']:.4f}")
        logger.info(f"   Precision: {test_metrics['Precision']:.4f}")
        logger.info(f"   Recall:    {test_metrics['Recall']:.4f}")

        # 保存折模型（如果配置启用）
        if config.experiment.kfold_experiment['save_fold_models']:
            fold_model_path = config.paths.kfold_models_dir / f"fold_{fold+1}_model.pth"
            ModelUtils.save_model_checkpoint(
                model, optimizer, best_epoch-1, test_metrics,
                str(fold_model_path)
            )
            logger.info(f"\n💾 模型已保存: {fold_model_path}")

        logger.info(f"{'='*60}\n")

        return model, test_metrics, history

    def _train_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                     train_data: Dict) -> float:
        """训练一个epoch"""
        model.train()
        total_loss = 0
        num_batches = 0

        train_edges = train_data['edges']
        train_labels = train_data['labels']
        batch_size = config.training.batch_size

        for i in range(0, len(train_edges), batch_size):
            num_batches += 1
            batch_edges = train_edges[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            optimizer.zero_grad()

            # [MASK] 可选：把本 batch 的监督正例从消息图中临时遮蔽，使训练时被预测的边与验证/测试一样“不在图中”，
            # 避免模型学到“图里已有这条边 -> 正例”的捷径。只改变训练时的输入图，不改模型。
            graph = self.graph
            if getattr(config.training, 'mask_supervised_edges', False):
                graph = self._mask_batch_positives(batch_edges[batch_labels == 1])

            # XGFT模型前向传播
            x_local, x_global = model(self.data.x_dict, graph)

            # 预测
            mirna_indices = batch_edges[:, 0]
            gene_indices = batch_edges[:, 1]
            predictions = model.predict_links(x_local, x_global, mirna_indices, gene_indices,
                                              pair_features=self._pair(batch_edges),
                                              evidence=self._evidence(batch_edges), graph=graph)

            # 计算损失
            # [NEG] 负样本比例 >1 时按类别加权：正例权重 = 比例，使正负两类总权重相同
            ratio = getattr(config.training, 'train_neg_ratio', 1)
            if ratio != 1:
                weight = torch.where(batch_labels == 1, float(ratio), 1.0)
                loss = nn.functional.binary_cross_entropy(predictions, batch_labels, weight=weight)
            else:
                loss = self.criterion(predictions, batch_labels)

            # [DEBIAS] 对抗损失：梯度反转后回归节点度数，抑制嵌入中的热度信息
            if self.degree_targets is not None:
                lambd = float(getattr(config.training, 'debias_lambda', 0.1))
                pred = model.degree_prediction({'miRNA': x_local['miRNA'], 'Gene': x_local['Gene']}, lambd)
                adv = sum(nn.functional.mse_loss(pred[nt], self.degree_targets[nt]) for nt in pred)
                loss = loss + adv

            # [MECHKG] 信息瓶颈：证据保留概率向稀疏先验 r 靠拢
            if getattr(model, 'mechkg', None) is not None:
                loss = loss + float(getattr(config.model, 'mechkg_beta', 0.01)) * model.mechkg.last_ib_loss

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches

    def _resample_train_negatives(self, train_data: Dict, val_data: Dict, fold: int, epoch: int) -> Dict:
        """
        [NEG] 本折训练正例 + 新采样的均匀随机未观测负例（数量 = 正例 × train_neg_ratio），并打乱顺序。
        排除：全部已知靶向对（含验证/测试正例）、验证集与测试集中的全部实体对（含其负例）。
        随机种子由 (seed, fold, epoch) 决定，可复现。
        """
        n_mirna, n_gene = self.data['miRNA'].num_nodes, self.data['Gene'].num_nodes
        if getattr(self, '_neg_forbidden_fold', None) != fold:
            forbidden = set()
            for edge_type, edge_index in self.data.edge_index_dict.items():  # 完整图：全部已知正例
                if is_target_edge_type(edge_type) and edge_type[0] == 'miRNA':
                    forbidden.update((edge_index[0] * n_gene + edge_index[1]).tolist())
            for d in (val_data, self.test_data, train_data):
                forbidden.update((d['edges'][:, 0] * n_gene + d['edges'][:, 1]).tolist())
            self._neg_forbidden, self._neg_forbidden_fold = forbidden, fold
        positives = train_data['edges'][train_data['labels'] == 1]
        n_neg = int(len(positives) * getattr(config.training, 'train_neg_ratio', 1))
        # [NEG-MATCH] 其中 matched_sets 组用端点度匹配负例，其余仍为均匀随机。
        # 按“组”而非比例切分：endpoint_matched 每次精确产出 len(positives) 条，
        # 按比例截断会破坏度匹配的精确性。matched_sets=0 时本分支不执行，RNG 消费与原实现完全一致。
        matched_sets = int(getattr(config.training, 'matched_neg_sets', 0))
        matched = self._matched_negatives(positives, matched_sets, fold, epoch) if matched_sets else None
        n_neg = max(n_neg - (len(matched) if matched is not None else 0), 0)
        rng = np.random.default_rng([config.seed, fold, epoch])
        negatives = set()
        while len(negatives) < n_neg:
            for code in rng.integers(0, n_mirna * n_gene, size=2 * n_neg).tolist():
                if code not in self._neg_forbidden:
                    negatives.add(code)
                    if len(negatives) == n_neg:
                        break
        # [NEG-MATCH] 显式 long：matched_sets 用满时 negatives 为空，torch.tensor([]) 会退化成 float32
        codes = torch.tensor(sorted(negatives), dtype=torch.long, device=positives.device)
        neg_edges = torch.stack([codes // n_gene, codes % n_gene], dim=1)
        if matched is not None:
            neg_edges = torch.cat([matched.to(positives.device), neg_edges])
        edges = torch.cat([positives, neg_edges])
        labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(neg_edges))]).to(positives.device)
        order = torch.as_tensor(rng.permutation(len(edges)), device=positives.device)
        return {'edges': edges[order], 'labels': labels[order]}

    def _forbidden_pairs(self):
        """[NEG-MATCH] 以元组形式给出全部禁用配对：已知靶向对 + 任何划分中出现过的实体对。
        与 hard_val_eval.py 的 build_forbidden 定义完全一致，保证两处构造的 hard 集同源。"""
        if getattr(self, '_forbidden_tuples', None) is not None:
            return self._forbidden_tuples
        forbidden = set()
        for et, ix in self.data.edge_index_dict.items():
            if is_target_edge_type(et) and et[0] == 'miRNA':
                forbidden |= set(map(tuple, ix.t().cpu().numpy().tolist()))
        for d in [self.test_data] + [x for f in self.kfold_data for x in f]:
            forbidden |= set(map(tuple, d['edges'].cpu().numpy().tolist()))
        self._forbidden_tuples = forbidden
        return forbidden

    def _build_hard_val(self, val_data, fold):
        """[NEG-MATCH] 本折 hard 验证集：验证正例 + 端点度匹配负例。
        种子与 hard_val_eval.py 相同（20260920+fold），所得集合与其逐条一致。"""
        endpoint_matched = _endpoint_matched()
        try:
            pos = val_data['edges'][val_data['labels'] == 1].cpu().numpy()
            neg = endpoint_matched(pos, self._forbidden_pairs(), np.random.default_rng(20260920 + fold))
            e = np.concatenate([pos, neg])
            y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
            logger.info(f"   [NEG-MATCH] hard 验证集: 正例 {len(pos)} + 度匹配负例 {len(neg)}")
            return {'edges': torch.as_tensor(e, device=self.device),
                    'labels': torch.as_tensor(y, device=self.device)}
        except Exception as exc:
            logger.warning(f"   [NEG-MATCH] hard 验证集构造失败，本折跳过: {exc}")
            return None

    def _matched_negatives(self, positives, n_sets, fold, epoch):
        """[NEG-MATCH] n_sets 组端点度匹配训练负例：每个 miRNA/Gene 在负例中的出现次数
        与训练正例完全一致，因此“只看节点热度”的打分无法区分正负。组间去重。"""
        endpoint_matched = _endpoint_matched()
        import time
        pos = positives.cpu().numpy()
        forbidden = set(self._forbidden_pairs())
        out, t0 = [], time.time()
        for k in range(n_sets):
            neg = endpoint_matched(pos, forbidden, np.random.default_rng([config.seed, fold, epoch, k]))
            out.append(neg)
            forbidden |= set(map(tuple, neg.tolist()))
        if epoch == 0:
            logger.info(f"   [NEG-MATCH] {n_sets} 组度匹配负例 x {len(pos)} 条, "
                        f"耗时 {time.time() - t0:.1f}s/轮")
        return torch.as_tensor(np.concatenate(out), dtype=positives.dtype)

    def _load_pair_table(self):
        """
        [EARLY-FUSION] 读取预计算的配对特征表，按当前图的节点编号重排为 [n_miRNA, n_Gene, d] 并放到 GPU。
        各列按全部配对做标准化（无监督，不使用标签）。
        """
        path = getattr(config.model, 'pair_feature_file', None)
        if not path:
            return None
        import numpy as np
        z = np.load(path)
        table = z['features']
        mi = {str(n).lower(): i for i, n in enumerate(z['mirnas'])}
        gi = {str(n): i for i, n in enumerate(z['genes'])}
        rev = self.data.reverse_mapping
        m_idx = [mi[str(rev['miRNA'][k]).lower()] for k in range(self.data['miRNA'].num_nodes)]
        g_idx = [gi[str(rev['Gene'][k])] for k in range(self.data['Gene'].num_nodes)]
        table = table[np.ix_(m_idx, g_idx)].astype(np.float32)
        flat = table.reshape(-1, table.shape[-1])
        mean, std = flat.mean(0), flat.std(0)
        table = (table - mean) / np.where(std > 0, std, 1)
        assert table.shape[-1] == config.model.pair_feature_dim, '配对特征维度与 config.model.pair_feature_dim 不一致'
        logger.info(f"[EARLY-FUSION] 配对特征表 {table.shape}，来源 {path}")
        return torch.as_tensor(table, device=self.device)

    def _pair(self, edges: torch.Tensor):
        """[EARLY-FUSION] 取一批 (miRNA, Gene) 的配对特征；未启用时返回 None。"""
        if self.pair_table is None:
            return None
        return self.pair_table[edges[:, 0], edges[:, 1]]

    def _build_evidence_table(self, train_data: Dict):
        """
        [EVIDENCE-GATE] 逐对证据可得性 [n_miRNA, n_Gene, 3]：
          共同疾病数、经本折已知靶点共享的通路/GO 数、与本折已知靶点存在 PPI 的数量。
        仅使用本折训练靶向边与辅助知识，不使用验证/测试标签。
        """
        if not getattr(config.model, 'evidence_dim', 0):
            return None
        import scipy.sparse as sp
        n_m, n_g = self.data['miRNA'].num_nodes, self.data['Gene'].num_nodes
        n_d = self.data['Disease'].num_nodes if 'Disease' in self.data.node_types else 0
        n_p = self.data['Pathway'].num_nodes if 'Pathway' in self.data.node_types else 0

        def mat(pairs, rows, cols):
            if len(pairs) == 0 or rows == 0 or cols == 0:
                return sp.csr_matrix((rows, cols), dtype='float32')
            a = np.asarray(pairs)
            return sp.csr_matrix((np.ones(len(a), 'float32'), (a[:, 0], a[:, 1])), shape=(rows, cols))

        def edges_of(pred, flip=False):
            out = []
            for (a, r, b), ix in self.graph.items():
                if r.startswith('reverse_') or not pred(a, r, b):
                    continue
                e = ix.t().cpu().numpy()
                out.append(e[:, ::-1] if flip else e)
            return np.concatenate(out) if out else np.zeros((0, 2), int)

        tr = train_data['edges'][train_data['labels'] == 1].cpu().numpy()
        T = mat(tr, n_m, n_g)                                                   # miRNA x Gene（本折已知靶点）
        MD = mat(edges_of(lambda a, r, b: (a, b) == ('miRNA', 'Disease')), n_m, n_d)
        GD = mat(edges_of(lambda a, r, b: (a, b) == ('Gene', 'Disease')), n_g, n_d)
        GP = mat(edges_of(lambda a, r, b: a == 'Gene' and b == 'Pathway'), n_g, n_p)
        pp = edges_of(lambda a, r, b: r == 'ppi_physical')
        PP = mat(np.concatenate([pp, pp[:, ::-1]]) if len(pp) else pp, n_g, n_g)

        shared_disease = (MD @ GD.T).toarray()
        shared_pathway = (T @ (GP @ GP.T)).toarray()
        ppi_links = (T @ PP).toarray()
        table = np.stack([shared_disease, shared_pathway, ppi_links], -1).astype('float32')
        table = np.log1p(table)
        table = (table - table.reshape(-1, 3).mean(0)) / np.maximum(table.reshape(-1, 3).std(0), 1e-6)
        logger.info(f"[EVIDENCE-GATE] 证据表 {table.shape}（共同疾病/共享通路/PPI，均由本折训练边计算）")
        return torch.as_tensor(table, device=self.device)

    def _build_degree_targets(self, train_data: Dict):
        """[DEBIAS] 对抗头的回归目标：本折训练正例中各节点的 log 度数（标准化）。"""
        if not getattr(config.model, 'debias', False):
            return None
        tr = train_data['edges'][train_data['labels'] == 1]
        out = {}
        for nt, col in (('miRNA', 0), ('Gene', 1)):
            deg = torch.bincount(tr[:, col], minlength=self.data[nt].num_nodes).float()
            z = torch.log1p(deg)
            out[nt] = ((z - z.mean()) / z.std().clamp(min=1e-6)).to(self.device)
        return out

    def _evidence(self, edges: torch.Tensor):
        """[EVIDENCE-GATE] 取一批候选对的证据向量；未启用时返回 None。"""
        if self.evidence_table is None:
            return None
        return self.evidence_table[edges[:, 0], edges[:, 1]]

    def _log_mechkg_contrib(self, model: nn.Module, eval_data: Dict, fold: int):
        """[MECHKG] 记录验证集上每个通道的 w_c·s_c 均值、证据覆盖率与机制占比（按正负例分开）。"""
        model.eval()
        with torch.no_grad():
            xl, xg = model(self.data.x_dict, self.graph)
            e, y = eval_data['edges'], eval_data['labels']
            _, parts = model.predict_links(xl, xg, e[:, 0], e[:, 1], pair_features=self._pair(e),
                                           evidence=self._evidence(e), graph=self.graph, return_parts=True)
        names = model.mechkg.channel_names
        c, r = parts['contrib'], parts['residual']
        tot = c.abs().sum(1) + r.abs()
        for lab, sel in (('pos', y == 1), ('neg', y == 0)):
            msg = ', '.join(f'{n}={c[sel, i].mean():+.3f}' for i, n in enumerate(names))
            share = (c[sel].abs().sum(1) / tot[sel].clamp(min=1e-6)).mean()
            logger.info(f"[MECHKG-CONTRIB] fold={fold + 1} {lab}: {msg}, residual={r[sel].mean():+.3f}, "
                        f"机制占比={share:.3f}")
        cov = {k: (v[1].sum(1) > 0).float().mean().item() for k, v in parts['evidence'].items()}
        logger.info(f"[MECHKG-COVER] fold={fold + 1} " + ', '.join(f'{k}={v:.3f}' for k, v in cov.items()))

    def _mask_batch_positives(self, batch_pos: torch.Tensor) -> Dict:
        """[MASK] 返回删除了 batch_pos（及其反向边）的消息图副本。"""
        n_gene = self.data['Gene'].num_nodes
        drop = batch_pos[:, 0] * n_gene + batch_pos[:, 1]
        graph = dict(self.graph)
        for edge_type, edge_index in self.graph.items():
            if not is_target_edge_type(edge_type):
                continue
            m, g = (edge_index[0], edge_index[1]) if edge_type[0] == 'miRNA' else (edge_index[1], edge_index[0])
            graph[edge_type] = edge_index[:, ~torch.isin(m * n_gene + g, drop)]
        return graph

    def _evaluate(self, model: nn.Module, eval_data: Dict) -> Dict[str, float]:
        """评估模型"""
        model.eval()

        with torch.no_grad():
            # XGFT模型前向传播
            x_local, x_global = model(self.data.x_dict, self.graph)

            eval_edges = eval_data['edges']
            eval_labels = eval_data['labels']

            mirna_indices = eval_edges[:, 0]
            gene_indices = eval_edges[:, 1]

            predictions = model.predict_links(x_local, x_global, mirna_indices, gene_indices,
                                              pair_features=self._pair(eval_edges),
                                              evidence=self._evidence(eval_edges), graph=self.graph)

            loss = self.criterion(predictions, eval_labels).item()

            # 转换为numpy进行指标计算
            y_true = eval_labels.cpu().numpy()
            y_scores = predictions.cpu().numpy()
            y_pred = (y_scores >= 0.5).astype(int)

            # 计算指标
            metrics = MetricsCalculator.calculate_metrics(y_true, y_pred, y_scores)
            metrics['loss'] = loss

            return metrics

    def train(self) -> Dict:
        """执行K折交叉验证训练"""
        logger.info("🚀 开始K折交叉验证训练...")
        logger.info(f"\n📊 训练配置:")
        logger.info(f"   K折数: {self.n_folds}")
        logger.info(f"   独立测试集大小: {len(self.test_data['labels']):,}")
        logger.info(f"   模型类型: XGFT")
        logger.info(f"   隐藏维度: {config.model.hidden_dim}")
        logger.info(f"   层数: {config.model.num_layers}")
        logger.info(f"   注意力头数: {config.model.num_heads}")
        logger.info(f"   初始学习率: {config.training.learning_rate}")
        logger.info(f"   批次大小: {config.training.batch_size}")
        logger.info(f"   设备: {self.device}")
        logger.info("")

        all_test_predictions = []
        all_test_labels = []

        # 训练每个折
        for fold, (train_data, val_data) in enumerate(self.kfold_data):
            model, test_metrics, history = self.train_fold(fold, train_data, val_data)

            # 保存结果
            self.fold_results.append(test_metrics)
            self.fold_models.append(model)
            self.fold_histories.append(history)

            # 收集测试集预测（用于集成）；self.graph 仍为本折消息图
            model.eval()
            with torch.no_grad():
                x_local, x_global = model(self.data.x_dict, self.graph)
                test_edges = self.test_data['edges']
                mirna_indices = test_edges[:, 0]
                gene_indices = test_edges[:, 1]
                predictions = model.predict_links(x_local, x_global, mirna_indices, gene_indices,
                                                  pair_features=self._pair(test_edges),
                                                  evidence=self._evidence(test_edges), graph=self.graph)
                all_test_predictions.append(predictions.cpu().numpy())

        all_test_labels = self.test_data['labels'].cpu().numpy()
        self.fold_test_predictions = all_test_predictions

        # 计算集成结果
        ensemble_results = self._compute_ensemble_results(all_test_predictions, all_test_labels)

        # 生成最终报告
        final_results = self._generate_final_report(ensemble_results)

        # 保存K折结果
        self._save_kfold_results(final_results)

        # 生成可视化
        self._generate_visualizations()

        logger.info("✅ K折交叉验证训练完成!")

        return final_results

    def _save_splits(self):
        """[FIX] 保存实体级划分（名称），便于独立复核与后续可解释性分析"""
        rev = self.data.reverse_mapping
        rows = []
        def add(split, d):
            for (m, g), y in zip(d['edges'].tolist(), d['labels'].tolist()):
                rows.append((split, rev['miRNA'][m], rev['Gene'][g], int(y)))
        add('test', self.test_data)
        for k, (tr, va) in enumerate(self.kfold_data):
            add(f'fold{k+1}_train', tr); add(f'fold{k+1}_val', va)
        import pandas as pd
        pd.DataFrame(rows, columns=['split', 'miRNA', 'gene', 'label']).to_csv(
            config.paths.kfold_results_dir / 'named_splits.csv', index=False)

    def _compute_ensemble_results(self, all_predictions: List[np.ndarray],
                                 labels: np.ndarray) -> Dict:
        """计算集成预测结果"""
        logger.info("🔮 计算集成预测结果...")

        # 平均集成
        avg_predictions = np.mean(all_predictions, axis=0)
        avg_pred_labels = (avg_predictions >= 0.5).astype(int)
        avg_metrics = MetricsCalculator.calculate_metrics(labels, avg_pred_labels, avg_predictions)

        # 投票集成
        vote_predictions = np.mean([(pred >= 0.5).astype(int) for pred in all_predictions], axis=0)
        vote_pred_labels = (vote_predictions >= 0.5).astype(int)
        vote_metrics = MetricsCalculator.calculate_metrics(labels, vote_pred_labels, vote_predictions)

        ensemble_results = {
            'average': avg_metrics,
            'voting': vote_metrics,
            'predictions': {
                'average': avg_predictions,
                'voting': vote_predictions
            }
        }

        logger.info(f"  平均集成 - AUC: {avg_metrics['AUC']:.4f}, AUPR: {avg_metrics['AUPR']:.4f}")
        logger.info(f"  投票集成 - AUC: {vote_metrics['AUC']:.4f}, AUPR: {vote_metrics['AUPR']:.4f}")

        return ensemble_results

    def _generate_final_report(self, ensemble_results: Dict) -> Dict:
        """生成最终的K折结果报告"""
        logger.info("📊 生成K折交叉验证报告...")

        # 计算每折指标的统计信息
        fold_metrics = {}
        metric_names = ['AUC', 'AUPR', 'F1', 'Accuracy', 'Precision', 'Recall']

        for metric in metric_names:
            values = [fold_result[metric] for fold_result in self.fold_results]
            fold_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }

        final_report = {
            'model_type': 'XGFT',
            'n_folds': self.n_folds,
            'fold_metrics': fold_metrics,
            'individual_fold_results': self.fold_results,
            'ensemble_results': ensemble_results,
            'config': {
                'hidden_dim': config.model.hidden_dim,
                'num_layers': config.model.num_layers,
                'num_heads': config.model.num_heads,
                'learning_rate': config.training.learning_rate,
                'batch_size': config.training.batch_size,
                'num_epochs': config.training.num_epochs
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # 打印摘要
        logger.info("\n" + "="*60)
        logger.info("📈 K折交叉验证结果摘要")
        logger.info("="*60)

        for metric in metric_names:
            stats = fold_metrics[metric]
            logger.info(f"{metric:10s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                       f"[{stats['min']:.4f}, {stats['max']:.4f}]")

        logger.info("\n🎯 集成结果:")
        logger.info(f"  平均集成 - AUC: {ensemble_results['average']['AUC']:.4f}, "
                   f"AUPR: {ensemble_results['average']['AUPR']:.4f}")
        logger.info(f"  投票集成 - AUC: {ensemble_results['voting']['AUC']:.4f}, "
                   f"AUPR: {ensemble_results['voting']['AUPR']:.4f}")
        logger.info("="*60)

        return final_report

    def _save_kfold_results(self, results: Dict):
        """保存K折结果"""
        # 保存JSON格式的详细结果
        results_file = config.paths.kfold_results_path
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"📁 K折结果已保存: {results_file}")

        # 保存预测结果（如果配置启用）
        if config.experiment.kfold_experiment['save_fold_predictions']:
            predictions_file = config.paths.kfold_results_dir / 'ensemble_predictions.npz'
            np.savez(predictions_file,
                    test_labels=self.test_data['labels'].cpu().numpy(),
                    avg_predictions=results['ensemble_results']['predictions']['average'],
                    fold_predictions=np.stack(self.fold_test_predictions),  # [FIX] 各折原始预测
                    test_edges=self.test_data['edges'].cpu().numpy(),
                    vote_predictions=results['ensemble_results']['predictions']['voting'])
            logger.info(f"📁 预测结果已保存: {predictions_file}")

    def _generate_visualizations(self):
        """生成K折训练的可视化"""
        logger.info("📊 生成K折可视化...")

        # 为每个折生成训练曲线
        for fold, history in enumerate(self.fold_histories):
            self.visualizer.plot_training_curves(
                history,
                save_name=f'fold_{fold+1}_training_curves.png'
            )

        # 生成K折性能对比图
        self._plot_kfold_comparison()

    def _plot_kfold_comparison(self):
        """绘制K折性能对比图"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('K-Fold Cross Validation Results', fontsize=16)

        metrics = ['AUC', 'AUPR', 'F1', 'Accuracy']

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            # 提取每折的值
            values = [fold_result[metric] for fold_result in self.fold_results]
            folds = list(range(1, self.n_folds + 1))

            # 绘制条形图
            ax.bar(folds, values, color='skyblue', edgecolor='navy', alpha=0.7)

            # 添加平均线
            mean_value = np.mean(values)
            ax.axhline(y=mean_value, color='red', linestyle='--',
                      label=f'Mean: {mean_value:.4f}')

            # 设置标签
            ax.set_xlabel('Fold')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} per Fold')
            ax.set_xticks(folds)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = config.paths.plots_dir / 'kfold_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 K折对比图已保存: {save_path}")


class SimpleTrainer:
    """简单训练器 - 用于单次训练（非K折）"""

    def __init__(self, model: nn.Module, data, device: Optional[str] = None):
        # [FIX] 该训练器仍在完整图上训练和评估（含测试靶向边），未做泄漏修复，禁止使用
        raise NotImplementedError('SimpleTrainer 未修复目标边泄漏，请使用 KFoldTrainer (USE_KFOLD=True)')
        self.model = model.to(device or config.device)
        self.data = data.to(device or config.device)
        self.device = device or config.device

        # 创建数据分割
        self.train_data, self.val_data, self.test_data = create_train_test_split(self.data)

        # 训练组件
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.BCELoss()
        self.early_stopping = None

        # 可视化工具
        self.visualizer = AdvancedVisualization(config.paths.plots_dir)

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_aupr': [],
            'val_f1': [],
            'lr': []
        }

        logger.info(f"训练器初始化完成，设备: {self.device}")

    def setup_training_components(self):
        """设置训练组件"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.7,
            patience=8,
            min_lr=1e-6
        )

        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            mode='max'
        )

        logger.info("训练组件设置完成")

    def train_epoch(self, epoch: int, epoch_pbar: tqdm) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        train_edges = self.train_data['edges']
        train_labels = self.train_data['labels']
        batch_size = config.training.batch_size

        for i in range(0, len(train_edges), batch_size):
            num_batches += 1
            batch_edges = train_edges[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            self.optimizer.zero_grad()

            # XGFT模型前向传播
            x_local, x_global = self.model(self.data.x_dict, self.data.edge_index_dict)

            mirna_indices = batch_edges[:, 0]
            gene_indices = batch_edges[:, 1]
            predictions = self.model.predict_links(x_local, x_global, mirna_indices, gene_indices)

            loss = self.criterion(predictions, batch_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if num_batches % 5 == 0:
                epoch_pbar.set_postfix({
                    'batch': f"{num_batches}/{len(train_edges) // batch_size}",
                    'batch_loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / num_batches:.4f}"
                })

        return {'loss': total_loss / num_batches}

    def evaluate(self, split: str = 'val') -> Dict[str, float]:
        """评估模型"""
        self.model.eval()

        if split == 'val':
            eval_data = self.val_data
        elif split == 'test':
            eval_data = self.test_data
        else:
            eval_data = self.train_data

        with torch.no_grad():
            x_local, x_global = self.model(self.data.x_dict, self.data.edge_index_dict)

            eval_edges = eval_data['edges']
            eval_labels = eval_data['labels']

            mirna_indices = eval_edges[:, 0]
            gene_indices = eval_edges[:, 1]

            predictions = self.model.predict_links(x_local, x_global, mirna_indices, gene_indices)

            loss = self.criterion(predictions, eval_labels).item()

            y_true = eval_labels.cpu().numpy()
            y_scores = predictions.cpu().numpy()
            y_pred = (y_scores >= 0.5).astype(int)

            metrics = MetricsCalculator.calculate_metrics(y_true, y_pred, y_scores)
            metrics['loss'] = loss

            return metrics

    def train(self):
        """训练模型"""
        logger.info("🚀 开始训练...")

        self.setup_training_components()
        best_val_auc = 0

        main_pbar = tqdm(
            total=config.training.num_epochs,
            desc="🧠 Training Progress",
            file=sys.stdout,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )

        try:
            for epoch in range(config.training.num_epochs):
                epoch_start_time = time.time()

                train_metrics = self.train_epoch(epoch, main_pbar)
                val_metrics = self.evaluate('val')

                self.history['train_loss'].append(train_metrics['loss'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_auc'].append(val_metrics['AUC'])
                self.history['val_aupr'].append(val_metrics['AUPR'])
                self.history['val_f1'].append(val_metrics['F1'])
                self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

                self.scheduler.step(val_metrics['AUC'])

                if val_metrics['AUC'] > best_val_auc:
                    best_val_auc = val_metrics['AUC']
                    improvement_indicator = "📈"
                    ModelUtils.save_model_checkpoint(
                        self.model, self.optimizer, epoch, val_metrics,
                        config.paths.best_model_path
                    )
                else:
                    improvement_indicator = "📉"

                epoch_time = time.time() - epoch_start_time

                main_pbar.set_postfix({
                    'AUC': f"{val_metrics['AUC']:.4f}{improvement_indicator}",
                    'AUPR': f"{val_metrics['AUPR']:.4f}",
                    'F1': f"{val_metrics['F1']:.4f}",
                    'Loss': f"{val_metrics['loss']:.4f}",
                    'LR': f"{self.optimizer.param_groups[0]['lr']:.1e}",
                    'Time': f"{epoch_time:.1f}s"
                })

                main_pbar.update(1)

                # 每5个epoch输出详细验证指标
                if (epoch + 1) % 5 == 0:
                    main_pbar.write(f"\n📊 Epoch {epoch+1} 验证集指标:")
                    main_pbar.write(f"   Loss:      {val_metrics['loss']:.4f}")
                    main_pbar.write(f"   AUC:       {val_metrics['AUC']:.4f} {'(Best!)' if val_metrics['AUC'] >= best_val_auc else ''}")
                    main_pbar.write(f"   AUPR:      {val_metrics['AUPR']:.4f}")
                    main_pbar.write(f"   F1:        {val_metrics['F1']:.4f}")
                    main_pbar.write(f"   Accuracy:  {val_metrics['Accuracy']:.4f}")
                    main_pbar.write(f"   Precision: {val_metrics['Precision']:.4f}")
                    main_pbar.write(f"   Recall:    {val_metrics['Recall']:.4f}")
                    main_pbar.write(f"   LR:        {self.optimizer.param_groups[0]['lr']:.1e}")
                    main_pbar.write(f"   训练Loss:  {train_metrics['loss']:.4f}\n")

                if self.early_stopping(val_metrics['AUC'], self.model):
                    main_pbar.set_description(f"🛑 Early Stopped")
                    main_pbar.write(f"早停在第 {epoch+1} 轮")
                    break

        except KeyboardInterrupt:
            main_pbar.set_description("⏹️ Interrupted")
            main_pbar.write("训练被用户中断")
        finally:
            main_pbar.close()

        logger.info("🎉 训练完成! 进行最终测试...")

        try:
            ModelUtils.load_model_checkpoint(self.model, config.paths.best_model_path, device=self.device)
            logger.info("✅ 成功加载最佳模型权重")
        except Exception as e:
            logger.warning(f"⚠️ 加载最佳模型失败: {e}, 使用当前权重进行测试")

        test_metrics = self.evaluate('test')

        logger.info("🏆 最终测试结果:")
        for metric, value in test_metrics.items():
            if metric not in ['TP', 'TN', 'FP', 'FN']:
                logger.info(f"  {metric}: {value:.4f}")

        logger.info("📊 生成训练曲线...")
        self.visualizer.plot_training_curves(self.history)

        return test_metrics


def train_model(model: nn.Module = None, data = None, use_kfold: bool = None) -> Tuple:
    """训练模型的便利函数"""

    # 如果没有指定，从配置读取
    if use_kfold is None:
        use_kfold = config.data.use_kfold

    if use_kfold:
        logger.info("🔄 使用K折交叉验证训练")
        trainer = KFoldTrainer(data, n_folds=config.data.n_folds)
        results = trainer.train()
        return trainer, results
    else:
        logger.info("🔄 使用单次训练")
        trainer = SimpleTrainer(model, data)
        results = trainer.train()
        return trainer, results