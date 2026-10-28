"""
[MECHKG] 机制通道化的自解释知识图谱推理头（Mechanism-channel Rationale Network）

对每个候选对 (m, g)，从**当前消息图**（已做训练遮蔽）按生物机制抽取有预算的证据实例：
  C1 协同调控  m –seq_sim– m′ –target– g          （中间节点 m′）
               m –target– g′ –seq_sim– g          （中间节点 g′）
  C2 网络邻近  m –target– g′ –PPI– g              （中间节点 g′）
  C3 功能共通路 m –target– g′ –in/GO– P –∋– g      （中间节点 P，附带 m 的靶点落在 P 中的数量）
  C4 表型共病  m –assoc– D (–is_a–) –assoc– g      （中间节点 D，含一步 is_a 扩展）
  C5 直接互补  配对序列特征（可选，与图谱通道并列的普通通道）

每条证据由查询 q(m,g) 打分，经 Gumbel-sigmoid 得到保留概率 α（信息瓶颈：KL(α‖r) 稀疏先验，GSAT/GMT 系）。
预测只经由 α 加权的证据流过：logit = b + Σ_c w_c(q)·s_c + r(h_m,h_g)（残差可关）。

防泄漏：靶向边只取自传入的消息图（验证/测试对不在其中，批内正例已遮蔽）；
另对每个候选对显式剔除其自身靶向边 (m,g) 的贡献（C1–C3），无论它是否在图中。
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS = ['collab', 'ppi', 'pathway', 'disease']      # 图谱机制通道（C1–C4）
SUB = ['collab_mirna', 'collab_gene', 'ppi', 'pathway', 'disease']  # 证据子类型（关系嵌入）


def _dense(ix: torch.Tensor, n_a: int, n_b: int, device, sym=False) -> torch.Tensor:
    A = torch.zeros(n_a, n_b, device=device)
    if ix.numel():
        A[ix[0], ix[1]] = 1.0
        if sym:
            A[ix[1], ix[0]] = 1.0
    return A


def configure_from_data(model, data):
    """[MECHKG] 需要节点名称的设置（模型建好后调用；未启用相关选项时不做任何事）。
    disease_spec：排除 ID 不是具体 MeSH 描述符的疾病分类节点（如 MESH:C = 疾病总类）。"""
    import re
    head = getattr(model, 'mechkg', None)
    if head is None or not head.disease_spec or 'Disease' not in data.node_types:
        return
    names = data.reverse_mapping['Disease']
    bad = [i for i in range(data['Disease'].num_nodes) if not re.fullmatch(r'MESH:[CD]\d+', str(names[i]))]
    head.exclude_disease = torch.as_tensor(bad, dtype=torch.long)
    head._static = None                                                     # 下次前向按新设置重建


class MechKGHead(nn.Module):
    def __init__(self, hidden_dim: int, pair_feature_dim: int = 0, k: int = 16,
                 prior_r: float = 0.5, tau: float = 1.0, residual: bool = True, dropout: float = 0.2,
                 drop_graph: float = 0.0, drop_seq: float = 0.0, disease_spec: bool = False,
                 off_channels=(), linear_concept: bool = False):
        super().__init__()
        H = hidden_dim
        # 线性机制打分：s_c = uᵀz_c + c·cover + b，使每条证据的贡献可精确加和（证据级精确分解）
        self.linear_concept = linear_concept
        self.off = tuple(off_channels)                                  # 通道消融：训练与推理都置零
        # 疾病证据特异性：排除非具体疾病的分类节点，并以信息量（覆盖越广越低）作为证据标量
        self.disease_spec = disease_spec
        self.exclude_disease = None                                         # 由 configure_from_data 设置
        self.drop_graph, self.drop_seq = drop_graph, drop_seq          # 机制通道随机失活概率（仅训练）
        self.k, self.r, self.tau, self.residual = k, prior_r, tau, residual
        self.pair_feature_dim = pair_feature_dim
        self.query = nn.Sequential(nn.Linear(2 * H, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(dropout))
        self.rel_emb = nn.Embedding(len(SUB), H)
        self.inst = nn.Sequential(nn.Linear(H + 1, H), nn.LayerNorm(H), nn.GELU())    # [h_mid ⊕ 标量] + 关系嵌入
        self.att_q, self.att_v = nn.Linear(H, H, bias=False), nn.Linear(H, H)
        self.att_out = nn.Linear(H, 1)
        self.empty = nn.Parameter(torch.zeros(len(CHANNELS), H))                  # 无证据时的通道表示
        if linear_concept:
            self.concept = nn.ModuleList([nn.Linear(H + 1, 1) for _ in CHANNELS])  # z_c ⊕ 覆盖率 → s_c（线性）
        else:
            self.concept = nn.ModuleList([nn.Sequential(nn.Linear(H + 1, H // 2), nn.GELU(), nn.Linear(H // 2, 1))
                                          for _ in CHANNELS])                       # z_c ⊕ 证据覆盖率 → s_c
        n_c = len(CHANNELS) + (1 if pair_feature_dim > 0 else 0)
        if pair_feature_dim > 0:
            self.seq_concept = nn.Sequential(nn.Linear(pair_feature_dim, H // 2), nn.GELU(), nn.Linear(H // 2, 1))
        self.gate = nn.Linear(H, n_c)                                               # w_c(q) = softplus(·)
        if residual:
            self.res = nn.Sequential(nn.Linear(2 * H, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(H, H // 2), nn.GELU(), nn.Linear(H // 2, 1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.channel_names = CHANNELS + (['sequence'] if pair_feature_dim > 0 else [])
        self._static = None
        self.last_ib_loss = None

    # ---------- 静态辅助关系（与折无关，惰性构建，不进 state_dict） ----------
    def _prepare(self, graph: Dict, n: Dict[str, int], device):
        get = lambda a, b, key: [ix for (h, r, t), ix in graph.items()
                                 if h == a and t == b and not r.startswith('reverse_') and key(r)]
        cat = lambda ixs: torch.cat(ixs, 1) if ixs else torch.zeros(2, 0, dtype=torch.long, device=device)
        nm, ng, nd, np_ = n['miRNA'], n['Gene'], n.get('Disease', 0), n.get('Pathway', 0)
        S = {}
        S['msim'] = _dense(cat(get('miRNA', 'miRNA', lambda r: 'similar' in r)), nm, nm, device, sym=True)
        S['gsim'] = _dense(cat(get('Gene', 'Gene', lambda r: 'similar' in r)), ng, ng, device, sym=True)
        S['ppi'] = _dense(cat(get('Gene', 'Gene', lambda r: 'ppi' in r)), ng, ng, device, sym=True)
        for k_ in ('msim', 'gsim', 'ppi'):
            S[k_].fill_diagonal_(0)
        S['gp'] = _dense(cat(get('Gene', 'Pathway', lambda r: True)), ng, np_, device)
        md = _dense(cat(get('miRNA', 'Disease', lambda r: True)), nm, nd, device)
        gd = _dense(cat(get('Gene', 'Disease', lambda r: True)), ng, nd, device)
        isa = _dense(cat(get('Disease', 'Disease', lambda r: 'is_a' in r)), nd, nd, device)
        up = (isa + torch.eye(nd, device=device)).clamp(max=1)                   # 自身 + 一步 is_a 父类
        S['md'], S['gd'] = (md @ up).clamp(max=1), (gd @ up).clamp(max=1)
        if self.disease_spec:
            if self.exclude_disease is not None and len(self.exclude_disease):
                S['md'][:, self.exclude_disease] = 0.0
                S['gd'][:, self.exclude_disease] = 0.0
            # 信息量 IC(d) = −log p_miRNA(d) − log p_gene(d)，按扩展后的关联覆盖率计算（无标签），缩放到约 [0,1]
            ic = -torch.log((S['md'].sum(0) + 1) / (nm + 1)) - torch.log((S['gd'].sum(0) + 1) / (ng + 1))
            S['dis_ic'] = ic / ic.max().clamp(min=1e-6)
        self._static = S

    @staticmethod
    def _targets(graph: Dict, nm: int, ng: int, device) -> torch.Tensor:
        T = torch.zeros(nm, ng, device=device)
        for (h, r, t), ix in graph.items():
            if {h, t} == {'miRNA', 'Gene'} and 'validated_interaction' in r:
                m, g = (ix[0], ix[1]) if h == 'miRNA' else (ix[1], ix[0])
                T[m, g] = 1.0
        return T

    def _topk(self, valid: torch.Tensor, prio: torch.Tensor):
        """从 valid[B,N] 中按优先级取至多 K 个；返回 (索引[B,K], 有效[B,K])"""
        k = min(self.k, valid.shape[1])
        score = torch.where(valid > 0, prio, torch.full_like(prio, -1.0))
        val, idx = score.topk(k, dim=1)
        return idx, (val >= 0).float()

    def _node_prio(self, n: int, device, salt: int) -> torch.Tensor:
        """与配对无关的固定优先级（仅用于超出预算 K 时打破平局），缓存复用。"""
        key = ('prio', salt)
        if key not in self._static:
            g = torch.Generator(device='cpu').manual_seed(20260920 + salt)
            self._static[key] = torch.rand(n, generator=g).to(device) * 0.5
        return self._static[key]

    # ---------- 证据抽取 ----------
    def extract(self, graph: Dict, n: Dict[str, int], m: torch.Tensor, g: torch.Tensor):
        dev = m.device
        if self._static is None:
            self._prepare(graph, n, dev)
        S = self._static
        T = self._targets(graph, n['miRNA'], n['Gene'], dev)
        B = len(m)
        Tm = T[m].clone()
        Tm[torch.arange(B, device=dev), g] = 0.0                                 # 剔除自身靶向边
        out = {}
        # C1a: m′ ∈ sim(m) 且 m′ 靶向 g
        v = S['msim'][m] * T[:, g].t()
        out['collab_mirna'] = self._topk(v, self._node_prio(n['miRNA'], dev, 1).expand(B, -1)) + ('miRNA', None)
        # C1b: g′ ∈ T(m) 且 g′ 与 g 序列相似
        v = Tm * S['gsim'][g]
        out['collab_gene'] = self._topk(v, self._node_prio(n['Gene'], dev, 2).expand(B, -1)) + ('Gene', None)
        # C2: g′ ∈ T(m) 且 PPI(g′, g)
        v = Tm * S['ppi'][g]
        out['ppi'] = self._topk(v, self._node_prio(n['Gene'], dev, 3).expand(B, -1)) + ('Gene', None)
        # C3: P ∋ g 且 P 含 m 的其它靶点；标量 = log1p(m 的靶点落在 P 中的数量)
        cnt = Tm @ S['gp']
        v = (cnt > 0).float() * S['gp'][g]
        c = torch.log1p(cnt)
        idx, ok = self._topk(v, c + self._node_prio(S['gp'].shape[1], dev, 4).expand(B, -1))
        out['pathway'] = (idx, ok, 'Pathway', c.gather(1, idx))
        # C4: D ∈ dis(m) ∩ dis(g)（含 is_a 父类）
        v = S['md'][m] * S['gd'][g]
        if self.disease_spec:
            ic = S['dis_ic'].expand(B, -1)
            idx, ok = self._topk(v, ic + self._node_prio(S['md'].shape[1], dev, 5).expand(B, -1) * 0.01)
            out['disease'] = (idx, ok, 'Disease', ic.gather(1, idx))   # 预算内优先保留最特异的疾病
        else:
            out['disease'] = self._topk(v, self._node_prio(S['md'].shape[1], dev, 5).expand(B, -1)) + ('Disease', None)
        return out

    # ---------- 前向 ----------
    def forward(self, x: Dict[str, torch.Tensor], graph: Dict, m: torch.Tensor, g: torch.Tensor,
                pair_features: Optional[torch.Tensor] = None, return_parts: bool = False,
                drop: Optional[Dict[str, torch.Tensor]] = None, zero_channels=(),
                soft_mask: Optional[Dict[str, torch.Tensor]] = None):
        """drop: {证据子类型: [B,K] 布尔}，为 True 的证据视为被删除（用于忠实性检验）；
        zero_channels: 置零的通道名（用于通道干预）。"""
        n = {k_: v.shape[0] for k_, v in x.items()}
        hm, hg = x['miRNA'][m], x['Gene'][g]
        q = self.query(torch.cat([hm, hg], 1))
        ev = self.extract(graph, n, m, g)
        groups = {'collab': ['collab_mirna', 'collab_gene'], 'ppi': ['ppi'], 'pathway': ['pathway'],
                  'disease': ['disease']}
        scores, ib, parts = [], [], {}
        terms, consts = {}, []            # [线性打分] 每条证据的未加权贡献、各通道常数项
        for ci, ch in enumerate(CHANNELS):
            vs, oks, alphas = [], [], []
            for sub in groups[ch]:
                idx, ok, ntype, sc = ev[sub]
                if drop is not None and sub in drop:
                    ok = ok * (~drop[sub]).float()
                if soft_mask is not None and sub in soft_mask:          # 连续掩码（事后归因基线用；默认不启用）
                    ok = ok * soft_mask[sub]
                h = x[ntype][idx]                                                   # [B,K,H]
                sc = torch.zeros_like(ok).unsqueeze(-1) if sc is None else sc.unsqueeze(-1)
                v = self.inst(torch.cat([h, sc], -1)) + self.rel_emb.weight[SUB.index(sub)]
                a = self.att_out(torch.tanh(self.att_q(q).unsqueeze(1) + self.att_v(v))).squeeze(-1)
                if self.training:
                    u = torch.rand_like(a).clamp(1e-6, 1 - 1e-6)
                    alpha = torch.sigmoid((a + torch.log(u) - torch.log1p(-u)) / self.tau)
                else:
                    alpha = torch.sigmoid(a)
                alpha = alpha * ok
                vs.append(v); oks.append(ok); alphas.append(alpha)
                parts[sub] = (idx, ok, alpha)
                if ok.sum() > 0 and ch not in self.off:                     # 被消融的通道不参与信息瓶颈
                    p = torch.sigmoid(a)[ok > 0].clamp(1e-6, 1 - 1e-6)
                    ib.append(p * torch.log(p / self.r) + (1 - p) * torch.log((1 - p) / (1 - self.r)))
            v, ok, alpha = torch.cat(vs, 1), torch.cat(oks, 1), torch.cat(alphas, 1)
            K = ok.shape[1]
            z = (alpha.unsqueeze(-1) * v).sum(1) / K
            cover = ok.sum(1, keepdim=True) / K
            z = z + (cover == 0).float() * self.empty[ci]
            scores.append(self.concept[ci](torch.cat([z, cover], 1)).squeeze(-1))
            if self.linear_concept:
                u, c, b = self.concept[ci].weight[0, :-1], self.concept[ci].weight[0, -1], self.concept[ci].bias[0]
                for sub, vv, oo, aa in zip(groups[ch], vs, oks, alphas):
                    terms[sub] = (aa * (vv @ u) + c * oo) / K                        # 证据 i：(α_i uᵀv_i + c·ok_i)/K
                consts.append(b + (cover.squeeze(1) == 0).float() * (self.empty[ci] @ u))
        if self.pair_feature_dim > 0:
            scores.append(self.seq_concept(pair_features).squeeze(-1))
        s = torch.stack(scores, 1)                                                 # [B,C]
        w = F.softplus(self.gate(q))
        mult = w                                                                    # 每个通道的最终乘子
        # 机制通道随机失活（训练时）：逐样本逐通道以概率 p 置零整条机制，迫使各通道独立成证据、
        # 避免最强通道（序列）压制图谱通道
        if self.training and (self.drop_graph > 0 or self.drop_seq > 0):
            p = torch.full((len(self.channel_names),), self.drop_graph, device=s.device)
            if 'sequence' in self.channel_names:
                p[self.channel_names.index('sequence')] = self.drop_seq
            mult = mult * torch.bernoulli((1 - p).expand_as(mult))
        for ch in tuple(zero_channels) + self.off:
            mult = mult.index_fill(1, torch.tensor([self.channel_names.index(ch)], device=s.device), 0.0)
        contrib = mult * s
        logit = self.bias + contrib.sum(1)
        res = self.res(torch.cat([hm, hg], 1)).squeeze(-1) if self.residual else torch.zeros_like(logit)
        logit = logit + res
        self.last_ib_loss = torch.cat(ib).mean() if ib else logit.new_zeros(())
        prob = torch.sigmoid(logit)
        if return_parts:
            out = {'contrib': contrib, 'residual': res, 'weights': w, 'scores': s, 'evidence': parts}
            if self.linear_concept:
                # 证据级精确分解：logit = bias + Σ_i phi_i + Σ_c phi_const_c + 序列贡献 + 残差
                sub_ch = {'collab_mirna': 0, 'collab_gene': 0, 'ppi': 1, 'pathway': 2, 'disease': 3}
                out['phi'] = {sub: mult[:, sub_ch[sub]:sub_ch[sub] + 1] * t for sub, t in terms.items()}
                out['phi_const'] = mult[:, :len(CHANNELS)] * torch.stack(consts, 1)
            return prob, out
        return prob
