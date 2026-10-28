"""
XGFT模型模块 - 交互式图谱融合变换器
文件名: models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
import numpy as np
from typing import Dict, Tuple, Optional
from config import config
import logging
import math

logger = logging.getLogger(__name__)


class XGFT_Layer(nn.Module):
    """
    ✨ 交互式图谱融合变换器层 (XGFT Layer) - v2.0 优化版
    - 全局注意力：采用高效的"全局令牌"近似法，避免O(N^2)计算瓶颈
    - 保留核心的持续、双向、门控式交互思想
    """

    def __init__(self, hidden_dim: int, num_heads: int, metadata: tuple,
                 dropout: float = 0.2, attn_dropout: float = 0.1,
                 num_global_tokens: int = 1, global_mode: str = 'token',
                 expander_degree: int = 3, num_virtual: int = 4,
                 expander_seed: int = 20260920, encoder: str = 'hgt'):
        super().__init__()
        self.encoder = encoder              # [ENCODER] hgt（默认，原实现）/ rgcn / han —— 基线对比用

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.metadata = metadata
        # [GLOBAL-TOKENS] 全局令牌数 M：M=1 与原实现等价，M=0 关闭全局分支，M>1 为真正的诱导点注意力
        self.num_global_tokens = num_global_tokens
        # [EXPHORMER] global_mode: 'none' 无全局分支 / 'token' 原令牌方案 / 'expander' 扩张图稀疏注意力
        self.global_mode = 'none' if num_global_tokens <= 0 and global_mode == 'token' else global_mode
        self.use_global = self.global_mode != 'none'
        self.expander_degree = expander_degree
        self.num_virtual = num_virtual
        self.expander_seed = expander_seed
        self._exp_cache = None

        # 📍 局部处理模块 (HGT)
        if encoder == 'hgt':
            self.local_gnn = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads
            )
        elif encoder == 'rgcn':
            # [ENCODER] R-GCN：每种关系一个独立的 GraphConv（均值聚合 + 自环权重），跨关系求和
            from torch_geometric.nn import HeteroConv, GraphConv
            self.local_gnn = HeteroConv({et: GraphConv(hidden_dim, hidden_dim, aggr='mean') for et in metadata[1]},
                                        aggr='sum')
        elif encoder == 'han':
            # [ENCODER] HAN：节点级注意力 + 关系（语义）级注意力
            from torch_geometric.nn import HANConv
            self.local_gnn = HANConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads, dropout=attn_dropout)
        else:
            raise ValueError(f'未知编码器 {encoder}')
        self.norm_local = nn.LayerNorm(hidden_dim)

        # 🌐 ✨ 高效的全局注意力模块 - 使用全局令牌方法
        # [GLOBAL-TOKENS] M 个可学习诱导点。注意 M=1 时下方第二步注意力的 softmax 作用在长度为 1 的
        # 序列上，恒等于 1，输出与 Query 无关且所有节点相同——即退化为一个共享常数偏置。
        # 数值验证见 check_global_tokens.py。M>1 才产生节点特异的全局交互。
        self.global_attn_token = nn.Parameter(torch.randn(1, max(num_global_tokens, 1), hidden_dim))

        # [EXPHORMER] 扩张图稀疏全局注意力：每个节点只对「扩张图邻居 + 虚拟全局节点」做注意力。
        # 随机 d-正则图具有谱扩张性，O(|V|+|E|) 代价下近似全局信息混合（Shirzad et al., ICML 2023）。
        # 与 M=1 的关键区别：Key 是一个集合而非单个向量，softmax 不会退化为恒等广播。
        if self.global_mode == 'expander':
            from torch_geometric.nn import TransformerConv
            self.expander_attn = TransformerConv(
                in_channels=hidden_dim, out_channels=hidden_dim // num_heads,
                heads=num_heads, concat=True, dropout=attn_dropout, beta=False)
            if num_virtual > 0:
                self.virtual_emb = nn.Parameter(torch.randn(num_virtual, hidden_dim) * 0.02)
        self.global_attn_mechanism = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        self.norm_global = nn.LayerNorm(hidden_dim)

        # 📈 FFN (前馈网络) - 分别为局部和全局流设计
        self.ffn_local = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn_global = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm_ffn_local = nn.LayerNorm(hidden_dim)
        self.norm_ffn_global = nn.LayerNorm(hidden_dim)

        # ✨ 创新核心：可学习的交织融合门控参数
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 全局 -> 局部 的融合权重
        self.beta = nn.Parameter(torch.tensor(0.1))  # 局部 -> 全局 的融合权重

        logger.info(f"✅ XGFT_Layer初始化: {hidden_dim}维, {num_heads}头, 高效全局注意力")

    def reset_parameters(self):
        """参数初始化"""
        self.local_gnn.reset_parameters()

        # 初始化全局令牌
        nn.init.xavier_uniform_(self.global_attn_token)

        # 初始化FFN
        for module in [self.ffn_local, self.ffn_global]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # 初始化融合参数
        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.beta, 0.1)

    def _efficient_global_attention(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        高效的全局注意力近似实现 - O(N)复杂度
        使用全局令牌作为信息中枢，避免O(N^2)的计算
        """
        # 收集所有节点特征
        node_features_list = []
        node_counts = []
        node_types = list(x_dict.keys())

        for node_type in node_types:
            features = x_dict[node_type]
            node_features_list.append(features)
            node_counts.append(features.size(0))

        # 拼接所有节点
        all_nodes = torch.cat(node_features_list, dim=0)  # (N_total, hidden_dim)

        # 第一步：所有节点向全局令牌写入信息
        # global_token作为Query，所有节点作为Key和Value
        # (1, 1, hidden_dim) x (1, N_total, hidden_dim) -> (1, 1, hidden_dim)
        global_summary, _ = self.global_attn_mechanism(
            self.global_attn_token,  # Query: (1, 1, hidden_dim)
            all_nodes.unsqueeze(0),  # Key: (1, N_total, hidden_dim)
            all_nodes.unsqueeze(0)  # Value: (1, N_total, hidden_dim)
        )

        # 第二步：全局令牌向所有节点广播信息
        # 所有节点作为Query，global_token作为Key和Value
        # (1, N_total, hidden_dim) x (1, 1, hidden_dim) -> (1, N_total, hidden_dim)
        global_features_flat, _ = self.global_attn_mechanism(
            all_nodes.unsqueeze(0),  # Query: (1, N_total, hidden_dim)
            global_summary,  # Key: (1, 1, hidden_dim)
            global_summary  # Value: (1, 1, hidden_dim)
        )
        global_features_flat = global_features_flat.squeeze(0)  # (N_total, hidden_dim)

        # 将结果拆分回字典
        split_global_features = torch.split(global_features_flat, node_counts, dim=0)

        h_global = {}
        for i, node_type in enumerate(node_types):
            h_global[node_type] = split_global_features[i]

        return h_global

    def _build_expander(self, n: int, device) -> torch.Tensor:
        """[EXPHORMER] 随机 d-正则扩张图：d 个随机置换的并集，对称化后度数为 2d。

        置换法保证正则性且为 O(n)；随机正则图以高概率具有良好谱隙（扩张性）。
        图由 expander_seed 确定，跨折、跨 epoch 完全一致，可复现。
        """
        if self._exp_cache is not None and self._exp_cache[0] == n:
            return self._exp_cache[1].to(device)
        g = torch.Generator().manual_seed(self.expander_seed)
        src = torch.arange(n).repeat(self.expander_degree)
        dst = torch.cat([torch.randperm(n, generator=g) for _ in range(self.expander_degree)])
        keep = src != dst  # 去掉置换产生的自环（注意力本身含自身项由 TransformerConv 的 root 权重提供）
        src, dst = src[keep], dst[keep]
        ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])  # 对称化
        self._exp_cache = (n, ei)
        return ei.to(device)

    def _virtual_edges(self, n: int, v: int, device) -> torch.Tensor:
        """[EXPHORMER] 每个真实节点与每个虚拟全局节点双向连边（虚拟节点编号 n..n+v-1）。"""
        real = torch.arange(n, device=device).repeat_interleave(v)
        virt = (n + torch.arange(v, device=device)).repeat(n)
        return torch.stack([torch.cat([real, virt]), torch.cat([virt, real])])

    def _expander_attention(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """[EXPHORMER] 在「扩张图 + 虚拟节点」这张稀疏图上做多头注意力，返回真实节点的输出。"""
        node_types = list(x_dict.keys())
        feats = [x_dict[t] for t in node_types]
        counts = [f.size(0) for f in feats]
        h = torch.cat(feats, dim=0)
        n, device = h.size(0), h.device
        ei = self._build_expander(n, device)
        if self.num_virtual > 0:
            h = torch.cat([h, self.virtual_emb.to(h.dtype)], dim=0)
            ei = torch.cat([ei, self._virtual_edges(n, self.num_virtual, device)], dim=1)
        out = self.expander_attn(h, ei)[:n]
        return dict(zip(node_types, torch.split(out, counts, dim=0)))

    def forward(self, x_local_dict: Dict[str, torch.Tensor],
                x_global_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[tuple, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        前向传播：并行处理 + 交织融合
        """

        # 💾 保存输入用于残差连接
        res_local = {k: v.clone() for k, v in x_local_dict.items()}
        res_global = {k: v.clone() for k, v in x_global_dict.items()}

        # 🏠 1. 局部信息处理 (HGT)
        try:
            if not edge_index_dict:
                logger.warning("边索引字典为空，跳过局部GNN处理")
                h_local = {k: self.norm_local(v) for k, v in x_local_dict.items()}
            else:
                h_local = self.local_gnn(x_local_dict, edge_index_dict)
                if self.encoder != 'hgt':
                    # [ENCODER] 没有入边的节点类型不会出现在输出里（或为 None）：沿用输入表示
                    h_local = {t: (h_local.get(t) if h_local.get(t) is not None else x_local_dict[t])
                               for t in x_local_dict}
                # 残差连接 + 层归一化
                for node_type in h_local:
                    if node_type in res_local:
                        h_local[node_type] = self.norm_local(h_local[node_type] + res_local[node_type])
                    else:
                        h_local[node_type] = self.norm_local(h_local[node_type])
        except Exception as e:
            logger.warning(f"局部GNN处理失败: {e}, 使用输入特征")
            h_local = {k: self.norm_local(v) for k, v in x_local_dict.items()}

        # [GLOBAL-TOKENS] M=0：完全关闭全局分支，仅保留局部流（预测头也只用局部嵌入）
        if not self.use_global:
            x_local_next = {}
            for node_type in h_local:
                ffn_out = self.ffn_local(h_local[node_type])
                nxt = self.norm_ffn_local(ffn_out + h_local[node_type])
                if self.training:
                    nxt = F.dropout(nxt, p=self.dropout, training=True)
                x_local_next[node_type] = nxt
            return x_local_next, dict(x_local_next)

        # 🌍 2. ✨ 全局信息处理 (高效版) - O(N)复杂度
        try:
            h_global_raw = (self._expander_attention(x_global_dict) if self.global_mode == 'expander'
                            else self._efficient_global_attention(x_global_dict))
            # 残差连接 + 层归一化
            h_global = {}
            for node_type in h_global_raw:
                if node_type in res_global:
                    h_global[node_type] = self.norm_global(h_global_raw[node_type] + res_global[node_type])
                else:
                    h_global[node_type] = self.norm_global(h_global_raw[node_type])
        except Exception as e:
            logger.warning(f"全局注意力处理失败: {e}, 使用输入特征")
            h_global = {k: self.norm_global(v) for k, v in x_global_dict.items()}

        # 🔀 3. ✨ 交织融合 (Cross-Fusion) - 核心创新
        # 使用 sigmoid 将 alpha 和 beta 限制在 (0, 1) 之间
        alpha_gate = torch.sigmoid(self.alpha)
        beta_gate = torch.sigmoid(self.beta)

        x_local_next = {}
        x_global_next = {}

        for node_type in h_local:
            if node_type in h_global:
                # 双向信息交换
                # 全局信息注入局部流
                local_update = (1 - alpha_gate) * h_local[node_type] + alpha_gate * h_global[node_type]
                # 局部信息注入全局流
                global_update = (1 - beta_gate) * h_global[node_type] + beta_gate * h_local[node_type]
            else:
                # 如果某个节点类型在全局字典中不存在，保持原样
                local_update = h_local[node_type]
                global_update = h_global.get(node_type, h_local[node_type])

            # 🚀 4. FFN处理 + 残差连接
            ffn_local_out = self.ffn_local(local_update)
            x_local_next[node_type] = self.norm_ffn_local(ffn_local_out + local_update)

            if node_type in h_global:
                ffn_global_out = self.ffn_global(global_update)
                x_global_next[node_type] = self.norm_ffn_global(ffn_global_out + global_update)
            else:
                x_global_next[node_type] = x_local_next[node_type]

            # Dropout
            if self.training:
                x_local_next[node_type] = F.dropout(x_local_next[node_type], p=self.dropout, training=True)
                x_global_next[node_type] = F.dropout(x_global_next[node_type], p=self.dropout, training=True)

        return x_local_next, x_global_next


class GradientReversal(torch.autograd.Function):
    """[DEBIAS] 前向恒等，反向取负：使主干抑制可被对抗头利用的信息（度数/热度）。"""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


class XGFT_Model(nn.Module):
    """
    ✨ 交互式图谱融合变换器模型 (X-GFT)

    核心创新：
    1. 持续的局部-全局信息交互
    2. 可学习的双向融合门控
    3. 层次化的特征表示学习
    """

    def __init__(self, metadata: tuple, node_feature_dims: Dict[str, int],
                 hidden_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 3, dropout: float = 0.2,
                 attn_dropout: float = 0.1, pair_feature_dim: int = 0,
                 evidence_dim: int = 0, debias: bool = False,
                 num_global_tokens: int = 1, global_mode: str = 'token',
                 expander_degree: int = 3, num_virtual: int = 4, mechkg: Optional[dict] = None,
                 encoder: str = 'hgt', seq_only: bool = False):
        super().__init__()
        self.encoder = encoder
        # [SEQ-MLP] 同训练流程的仅序列基线：预测只用配对序列特征，不用任何图谱信息（默认关闭）
        self.seq_only = seq_only
        if seq_only:
            assert pair_feature_dim > 0, '仅序列基线需要配对序列特征'
            self.seq_head = nn.Sequential(nn.Linear(pair_feature_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                          nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                                          nn.Linear(hidden_dim // 2, 1))
        # [GLOBAL-TOKENS] 无全局分支时预测头只接收局部嵌入（2*hidden_dim 而非 4*hidden_dim）
        self.num_global_tokens = num_global_tokens
        self.global_mode = 'none' if num_global_tokens <= 0 and global_mode == 'token' else global_mode
        self.use_global = self.global_mode != 'none'

        self.hidden_dim = hidden_dim
        # [EARLY-FUSION] 可选的 miRNA–Gene 配对特征（如序列互补位点）。为 0 时模型与原版完全相同。
        self.pair_feature_dim = pair_feature_dim
        # [EVIDENCE-GATE] 证据条件化门控：按候选对的证据可得性加权三路专家
        self.evidence_dim = evidence_dim
        # [DEBIAS] 热度去偏：对抗头从节点嵌入预测其训练度数，经梯度反转抑制
        self.debias = debias
        self.num_layers = num_layers
        self.dropout = dropout
        self.metadata = metadata
        self.node_feature_dims = node_feature_dims

        # 🎯 输入投影层
        self.projection_layers = nn.ModuleDict()
        for node_type, raw_dim in node_feature_dims.items():
            if raw_dim > 0:
                self.projection_layers[node_type] = nn.Sequential(
                    nn.Linear(raw_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5)
                )
                logger.info(f"  为 {node_type} 创建投影层: {raw_dim} -> {hidden_dim}")

        # 🏗️ 堆叠多个XGFT层
        self.xgft_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = XGFT_Layer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                metadata=metadata,
                dropout=dropout,
                attn_dropout=attn_dropout,
                num_global_tokens=num_global_tokens,
                global_mode=global_mode,
                expander_degree=expander_degree,
                num_virtual=num_virtual,
                encoder=encoder
            )
            self.xgft_layers.append(layer)
            logger.info(f"  添加XGFT层 {i+1}/{num_layers}")

        # [EARLY-FUSION] 配对特征编码器：pair_feature_dim -> hidden_dim // 2，与节点嵌入拼接后送入预测头
        pair_hidden = 0
        if pair_feature_dim > 0:
            pair_hidden = hidden_dim // 2
            self.pair_encoder = nn.Sequential(
                nn.Linear(pair_feature_dim, pair_hidden),
                nn.LayerNorm(pair_hidden),
                nn.GELU(),
                nn.Dropout(dropout * 0.5)
            )
            logger.info(f"  [EARLY-FUSION] 配对特征编码器: {pair_feature_dim} -> {pair_hidden}")

        # [EVIDENCE-GATE/gate_rep] 表示层门控：证据编码与门控权重
        self.evidence_mode = getattr(config.model, 'evidence_mode', 'gate_rep') if evidence_dim > 0 else None
        evidence_hidden = 0
        if self.evidence_mode == 'gate_rep':
            evidence_hidden = hidden_dim // 2
            self.evidence_encoder = nn.Sequential(
                nn.Linear(evidence_dim, evidence_hidden), nn.LayerNorm(evidence_hidden), nn.GELU(),
                nn.Dropout(dropout * 0.5))
            self.branch_gate = nn.Sequential(
                nn.Linear(evidence_dim + max(pair_feature_dim, 1), hidden_dim), nn.GELU(),
                nn.Dropout(dropout * 0.5), nn.Linear(hidden_dim, 3))

        # 🔗 最终的链接预测头 (拼接局部和全局嵌入)
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * (4 if self.use_global else 2) + pair_hidden + evidence_hidden, hidden_dim * 2),  # 局部+全局各2份；[GLOBAL-TOKENS] M=0 时仅局部2份
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        if evidence_dim > 0 and self.evidence_mode == 'mixture':
            # 三路专家：图谱表示流 / 配对序列流 / 图谱显式证据计数流
            expert = lambda d: nn.Sequential(nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                                             nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
            self.expert_graph = expert(hidden_dim * 4)
            self.expert_pair = expert(max(pair_feature_dim, 1))
            self.expert_evidence = expert(evidence_dim)
            # 门控只看"有哪些证据可用"，不看节点热度，输出三路权重
            self.evidence_gate = nn.Sequential(
                nn.Linear(evidence_dim + max(pair_feature_dim, 1), hidden_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, 3))
            logger.info(f"  [EVIDENCE-GATE] 证据门控启用: 证据维度 {evidence_dim}, 三路专家")
        # [MECHKG] 机制通道化的自解释证据头（默认关闭；开启时替代 link_predictor 做最终打分）
        self.mechkg = None
        if mechkg:
            from mechkg import MechKGHead
            self.mechkg = MechKGHead(hidden_dim, pair_feature_dim, k=mechkg['k'], prior_r=mechkg['r'],
                                     tau=mechkg['tau'], residual=mechkg['residual'], dropout=dropout,
                                     drop_graph=mechkg.get('drop_graph', 0.0), drop_seq=mechkg.get('drop_seq', 0.0),
                                     disease_spec=mechkg.get('disease_spec', False),
                                     off_channels=mechkg.get('off_channels', ()),
                                     linear_concept=mechkg.get('linear', False))
            logger.info(f"  [MECHKG] 证据头启用: K={mechkg['k']}, r={mechkg['r']}, 残差={mechkg['residual']}, "
                        f"通道={self.mechkg.channel_names}, 通道失活 图谱={mechkg.get('drop_graph', 0.0)} "
                        f"序列={mechkg.get('drop_seq', 0.0)}, 疾病特异性={mechkg.get('disease_spec', False)}, "
                        f"关闭通道={','.join(mechkg.get('off_channels', ())) or '无'}, "
                        f"线性打分={mechkg.get('linear', False)}")
        if debias:
            self.degree_head = nn.ModuleDict({nt: nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                                                                nn.Linear(hidden_dim // 2, 1))
                                              for nt in ('miRNA', 'Gene')})
            logger.info("  [DEBIAS] 热度对抗头启用 (miRNA/Gene)")

        if self.global_mode == 'expander':
            logger.info(f"  [GLOBAL-TOKENS] M=expander (扩张图稀疏全局注意力, "
                        f"度数 {expander_degree}, 虚拟节点 {num_virtual})")
        else:
            mdesc = {0: "已关闭全局分支", 1: "退化为共享常数偏置"}.get(
                num_global_tokens, "诱导点全局注意力")
            logger.info(f"  [GLOBAL-TOKENS] M={num_global_tokens} ({mdesc})")
        logger.info(f"  [ENCODER] 编码器={encoder}" + ("；[SEQ-MLP] 仅序列基线（预测不使用图谱）" if seq_only else ''))
        logger.info("✅ XGFT_Model 初始化完成 (交互式局部-全局融合)")
        logger.info(f"   节点类型投影层: {list(self.projection_layers.keys())}")
        logger.info(f"   XGFT层数: {num_layers}")
        logger.info(f"   隐藏维度: {hidden_dim}")

        self.reset_parameters()

    def reset_parameters(self):
        """参数初始化"""
        # 初始化投影层
        for proj_seq in self.projection_layers.values():
            for module in proj_seq:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # 初始化XGFT层
        for layer in self.xgft_layers:
            layer.reset_parameters()

        # [EARLY-FUSION] 初始化配对特征编码器
        if self.pair_feature_dim > 0:
            for module in self.pair_encoder:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        # 初始化链接预测头
        for module in self.link_predictor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_dict_raw: Dict[str, torch.Tensor],
                edge_index_dict: Dict[tuple, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        前向传播

        Returns:
            Tuple[Dict, Dict]: (局部嵌入字典, 全局嵌入字典)
        """

        # 🎯 1. 投影到统一维度
        x_dict = {}
        for node_type, raw_features in x_dict_raw.items():
            if node_type in self.projection_layers:
                projected_features = self.projection_layers[node_type](raw_features)
                x_dict[node_type] = projected_features

                # 调试检查
                if torch.isnan(projected_features).any():
                    logger.warning(f"投影后的 {node_type} 特征包含NaN值")
            else:
                x_dict[node_type] = raw_features

        # 🔄 2. 初始化局部和全局流（开始时相同）
        x_local, x_global = x_dict.copy(), x_dict.copy()

        # 🏗️ 3. 通过XGFT层进行持续交织融合
        for i, xgft_layer in enumerate(self.xgft_layers):
            try:
                x_local, x_global = xgft_layer(x_local, x_global, edge_index_dict)

                # 检查输出
                for node_type in x_local:
                    if torch.isnan(x_local[node_type]).any():
                        logger.warning(f"XGFT层{i+1}后局部流 {node_type} 包含NaN值")
                    if torch.isnan(x_global[node_type]).any():
                        logger.warning(f"XGFT层{i+1}后全局流 {node_type} 包含NaN值")

            except Exception as e:
                logger.error(f"XGFT层{i+1}处理失败: {e}")
                break

        return x_local, x_global

    def predict_links(self, x_local_dict: Dict[str, torch.Tensor],
                      x_global_dict: Dict[str, torch.Tensor],
                      mirna_indices: torch.Tensor,
                      gene_indices: torch.Tensor,
                      pair_features: Optional[torch.Tensor] = None,
                      evidence: Optional[torch.Tensor] = None,
                      return_parts: bool = False, graph: Optional[Dict] = None) -> torch.Tensor:
        """
        链接预测 - 融合局部和全局信息
        """
        # [SEQ-MLP] 仅序列基线
        if getattr(self, 'seq_only', False):
            return torch.sigmoid(self.seq_head(pair_features)).squeeze(-1)
        # [MECHKG] 证据头：需要当前消息图（训练时为批内遮蔽后的图）
        if self.mechkg is not None:
            assert graph is not None, '[MECHKG] predict_links 必须传入当前消息图 graph'
            return self.mechkg(x_local_dict, graph, mirna_indices, gene_indices,
                               pair_features=pair_features, return_parts=return_parts)
        # 获取miRNA的局部和全局嵌入
        mirna_local = x_local_dict['miRNA'][mirna_indices]
        mirna_global = x_global_dict['miRNA'][mirna_indices]

        # 获取Gene的局部和全局嵌入
        gene_local = x_local_dict['Gene'][gene_indices]
        gene_global = x_global_dict['Gene'][gene_indices]

        # 拼接所有特征（[GLOBAL-TOKENS] M=0 时全局流与局部流相同，不重复拼入）
        parts = ([mirna_local, mirna_global, gene_local, gene_global] if self.use_global
                 else [mirna_local, gene_local])

        # [EVIDENCE-GATE/gate_rep] 逐对门控缩放三路表示后送入原预测头（保留交互建模能力）
        if self.evidence_dim > 0 and self.evidence_mode == 'gate_rep':
            assert evidence is not None, '模型启用了证据门控，预测时必须提供 evidence'
            pf = pair_features if self.pair_feature_dim > 0 else torch.zeros(len(evidence), 1, device=evidence.device)
            rep_graph = torch.cat(parts, dim=1)
            rep_pair = self.pair_encoder(pair_features) if self.pair_feature_dim > 0 else None
            rep_ev = self.evidence_encoder(evidence)
            w = torch.sigmoid(self.branch_gate(torch.cat([evidence, pf], dim=1)))  # 三路独立门控
            blocks = [w[:, 0:1] * rep_graph]
            if rep_pair is not None:
                blocks.append(w[:, 1:2] * rep_pair)
            blocks.append(w[:, 2:3] * rep_ev)
            scores = self.link_predictor(torch.cat(blocks, dim=1)).squeeze()
            if return_parts:
                return scores, w, None
            return scores

        # [EVIDENCE-GATE/mixture] 三路专家 + 逐对门控（权重即解释）
        if self.evidence_dim > 0:
            assert evidence is not None, '模型启用了证据门控，预测时必须提供 evidence'
            pf = pair_features if self.pair_feature_dim > 0 else torch.zeros(len(evidence), 1, device=evidence.device)
            experts = torch.cat([self.expert_graph(torch.cat(parts, dim=1)),
                                 self.expert_pair(pf), self.expert_evidence(evidence)], dim=1)
            weights = torch.softmax(self.evidence_gate(torch.cat([evidence, pf], dim=1)), dim=1)
            logits = (weights * experts).sum(1)
            scores = torch.sigmoid(logits)
            if return_parts:
                return scores, weights, experts
            return scores

        # [EARLY-FUSION] 拼接编码后的配对特征
        if self.pair_feature_dim > 0:
            assert pair_features is not None, '模型启用了配对特征，预测时必须提供 pair_features'
            parts.append(self.pair_encoder(pair_features))
        concat_features = torch.cat(parts, dim=1)

        # 链接预测
        scores = self.link_predictor(concat_features)
        return scores.squeeze()

    def degree_prediction(self, x_dict, lambd: float):
        """[DEBIAS] 经梯度反转后由节点嵌入预测度数；返回 {类型: 预测值}"""
        return {nt: self.degree_head[nt](GradientReversal.apply(x_dict[nt], lambd)).squeeze(-1)
                for nt in self.degree_head}

    def get_node_embeddings(self, x_dict_raw: Dict[str, torch.Tensor],
                           edge_index_dict: Dict[tuple, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        获取节点嵌入（用于分析和可视化）
        """
        self.eval()
        with torch.no_grad():
            x_local, x_global = self.forward(x_dict_raw, edge_index_dict)
        return x_local, x_global

    def analyze_fusion_weights(self) -> Dict[str, float]:
        """
        分析各层的融合权重
        """
        fusion_analysis = {}

        for i, layer in enumerate(self.xgft_layers):
            alpha_value = torch.sigmoid(layer.alpha).item()
            beta_value = torch.sigmoid(layer.beta).item()

            fusion_analysis[f'layer_{i+1}'] = {
                'alpha (global->local)': alpha_value,
                'beta (local->global)': beta_value,
                'bidirectional_strength': (alpha_value + beta_value) / 2
            }

        return fusion_analysis


def create_model(metadata: tuple, node_feature_dims: Dict[str, int],
                 model_type: str = 'xgft', enhanced: bool = False) -> nn.Module:
    """
    模型工厂函数 - 创建XGFT模型

    Args:
        metadata: 图的元数据
        node_feature_dims: 各节点类型的原始特征维度
        model_type: 模型类型 (默认为'xgft')
        enhanced: 是否使用增强版本 (保留兼容性)

    Returns:
        创建的XGFT模型
    """

    logger.info("🧠 创建XGFT模型:")
    logger.info(f"   节点特征维度: {node_feature_dims}")
    logger.info(f"   目标隐藏维度: {config.model.hidden_dim}")

    model = XGFT_Model(
        metadata=metadata,
        node_feature_dims=node_feature_dims,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        attn_dropout=config.model.attn_dropout,
        pair_feature_dim=getattr(config.model, 'pair_feature_dim', 0),  # [EARLY-FUSION] 默认 0 = 原模型
        evidence_dim=getattr(config.model, 'evidence_dim', 0),  # [EVIDENCE-GATE] 默认 0 = 关闭
        debias=getattr(config.model, 'debias', False),  # [DEBIAS] 默认关闭
        num_global_tokens=getattr(config.model, 'num_global_tokens', 1),  # [GLOBAL-TOKENS] 默认 1 = 原实现
        global_mode=getattr(config.model, 'global_mode', 'token'),  # [EXPHORMER] 默认沿用令牌方案
        expander_degree=getattr(config.model, 'expander_degree', 3),
        num_virtual=getattr(config.model, 'num_virtual', 4),
        mechkg=({'k': config.model.mechkg_k, 'r': config.model.mechkg_r, 'tau': config.model.mechkg_tau,
                 'residual': config.model.mechkg_residual,
                 'drop_graph': getattr(config.model, 'mechkg_drop_graph', 0.0),
                 'drop_seq': getattr(config.model, 'mechkg_drop_seq', 0.0),
                 'disease_spec': getattr(config.model, 'mechkg_disease_spec', False),
                 'off_channels': tuple(getattr(config.model, 'mechkg_off_channels', ())),
                 'linear': getattr(config.model, 'mechkg_linear', False)}
                if getattr(config.model, 'mechkg', False) else None),  # [MECHKG] 默认关闭
        encoder=getattr(config.model, 'encoder', 'hgt'),  # [ENCODER] 默认 HGT
        seq_only=getattr(config.model, 'seq_only', False)  # [SEQ-MLP] 默认关闭
    )

    logger.info("✅ 创建XGFT模型 (交互式局部-全局融合)")

    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"   总参数: {total_params:,}")
    logger.info(f"   可训练参数: {trainable_params:,}")

    # 打印投影层信息
    proj_params = sum(p.numel() for name, p in model.named_parameters()
                     if 'projection_layers' in name)
    logger.info(f"   投影层参数: {proj_params:,} ({proj_params/total_params*100:.1f}%)")

    xgft_params = sum(p.numel() for name, p in model.named_parameters()
                     if 'xgft_layers' in name)
    logger.info(f"   XGFT层参数: {xgft_params:,} ({xgft_params/total_params*100:.1f}%)")

    return model


def load_pretrained_projections(model: nn.Module, projection_path: str):
    """加载预训练的投影层权重"""
    try:
        checkpoint = torch.load(projection_path, map_location='cpu')
        if 'projection_layers' in checkpoint:
            model.projection_layers.load_state_dict(checkpoint['projection_layers'])
            logger.info(f"✅ 成功加载预训练投影层: {projection_path}")
        else:
            logger.warning(f"⚠️ 权重文件中未找到投影层: {projection_path}")
    except Exception as e:
        logger.error(f"❌ 加载投影层失败: {e}")


def save_projection_layers(model: nn.Module, save_path: str):
    """单独保存投影层权重"""
    try:
        projection_state = {
            'projection_layers': model.projection_layers.state_dict(),
            'node_feature_dims': getattr(model, 'node_feature_dims', {}),
            'hidden_dim': model.hidden_dim
        }
        torch.save(projection_state, save_path)
        logger.info(f"✅ 投影层权重已保存: {save_path}")
    except Exception as e:
        logger.error(f"❌ 保存投影层失败: {e}")


def analyze_model_capacity(model: nn.Module, data) -> Dict:
    """分析模型容量和特征处理能力"""
    model.eval()
    analysis = {}

    with torch.no_grad():
        # 分析投影层
        if hasattr(model, 'projection_layers'):
            projection_quality = {}
            for node_type, raw_features in data.x_dict.items():
                if node_type in model.projection_layers:
                    projected = model.projection_layers[node_type](raw_features)
                    projection_quality[node_type] = {
                        'raw_mean': raw_features.mean().item(),
                        'raw_std': raw_features.std().item(),
                        'projected_mean': projected.mean().item(),
                        'projected_std': projected.std().item(),
                        'projection_info_preservation': torch.corrcoef(
                            torch.stack([
                                torch.norm(raw_features, dim=1),
                                torch.norm(projected, dim=1)
                            ])
                        )[0, 1].item()
                    }
            analysis['projection_quality'] = projection_quality

        # 分析XGFT融合权重
        if hasattr(model, 'analyze_fusion_weights'):
            analysis['xgft_fusion_weights'] = model.analyze_fusion_weights()

        # 分析参数分布
        param_stats = {}
        for name, param in model.named_parameters():
            param_stats[name] = {
                'shape': list(param.shape),
                'mean': param.mean().item(),
                'std': param.std().item(),
                'grad_norm': param.grad.norm().item() if param.grad is not None else 0.0
            }
        analysis['parameter_statistics'] = param_stats

    return analysis