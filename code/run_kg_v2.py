"""
XGFT（models.py 未改动）在 KG v2 + PPI 上的 5 折训练，泄漏修复与原图运行相同。
输出写入 ./run_kg_v2/ 下（results/checkpoints/plots/...），不覆盖原图运行的结果。
用法：在 消融实验_fixed 目录下执行  python run_kg_v2.py
"""
import os
import sys
from pathlib import Path

import argparse

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', default='run_kg_v2')
parser.add_argument('--mask', action='store_true', help='mask each batch positive edges from the message graph')
parser.add_argument('--dynamic-neg', action='store_true', help='resample training negatives every epoch')
parser.add_argument('--neg-ratio', type=float, default=1, help='training negatives per positive')
parser.add_argument('--pair-features', action='store_true', help='early fusion of miRNA-UTR seed-site features')
parser.add_argument('--drop-relations', default='', help='comma-separated KG relations to remove (ablation)')
parser.add_argument('--mode', default='full_biological', help='main.EXPERIMENT_MODE, e.g. llm_only = no expression')
parser.add_argument('--kg-file', default='kg_v2_xgft_coding_nosim.csv', help='KG csv under data_kg_v2/')
parser.add_argument('--expr-fix', action='store_true', help='expression: PCA 64 + zero fill with missing mask')
parser.add_argument('--evidence-gate', action='store_true', help='A: pair-conditioned evidence gating (3 experts)')
parser.add_argument('--debias', type=float, default=0, help='B: popularity debiasing strength (0=off)')
parser.add_argument('--global-tokens', type=int, default=1,
                    help='全局注意力诱导点数 M：0=关闭全局分支，1=原实现（退化为常数偏置），>1=真正的全局交互')
parser.add_argument('--matched-sets', type=int, default=0,
                    help='[NEG-MATCH] 训练负例中端点度匹配的组数（每组=正例数）；0=全均匀随机，与既有实验一致')
parser.add_argument('--early-stop-metric', default='standard', choices=['standard', 'hard'],
                    help='[NEG-MATCH] 早停/选型准则：standard=随机负例验证集（默认），hard=度匹配验证集')
parser.add_argument('--global-mode', default='token', choices=['token', 'expander', 'none'],
                    help='全局分支形式：token=令牌方案, expander=扩张图稀疏注意力(Exphormer), none=关闭')
parser.add_argument('--expander-degree', type=int, default=3, help='扩张图度数 d（对称化后 2d）')
parser.add_argument('--num-virtual', type=int, default=4, help='虚拟全局节点数；0 = 纯扩张图')
parser.add_argument('--split-file', default='coding_named_splits.csv',
                    help='splits/ 下的划分文件；剔除 lncRNA 后的新数据用 coding_named_splits.csv')
parser.add_argument('--pair-feature-file', default='pair_seq_features_coding.npz',
                    help='data_kg_v2/ 下的配对特征文件；新数据用 pair_seq_features_coding.npz')
parser.add_argument('--mechkg', action='store_true', help='[MECHKG] 机制通道化自解释证据头')
parser.add_argument('--mechkg-k', type=int, default=16, help='[MECHKG] 每个证据子类型的预算 K')
parser.add_argument('--mechkg-r', type=float, default=0.5, help='[MECHKG] 信息瓶颈稀疏先验 r')
parser.add_argument('--mechkg-beta', type=float, default=0.01, help='[MECHKG] 信息瓶颈损失权重')
parser.add_argument('--mechkg-tau', type=float, default=1.0, help='[MECHKG] Gumbel-sigmoid 温度')
parser.add_argument('--mechkg-no-residual', action='store_true', help='[MECHKG] 关闭实体残差项（纯证据预测）')
parser.add_argument('--mechkg-drop-graph', type=float, default=0.0, help='[MECHKG] 训练时图谱通道随机失活概率')
parser.add_argument('--mechkg-drop-seq', type=float, default=0.0, help='[MECHKG] 训练时序列通道随机失活概率')
parser.add_argument('--mechkg-disease-spec', action='store_true',
                    help='[MECHKG] 疾病证据特异性：排除分类节点（MESH:C），以信息量作证据标量并优先保留')
parser.add_argument('--mechkg-off', default='', help='[MECHKG] 通道消融：逗号分隔的通道名，训练与推理都关闭'
                                                     '（collab,ppi,pathway,disease,sequence）')
parser.add_argument('--mechkg-linear', action='store_true',
                    help='[MECHKG] 线性机制打分：每条证据贡献可精确加和（证据级精确分解）')
parser.add_argument('--encoder', default='hgt', choices=['hgt', 'rgcn', 'han'],
                    help='[ENCODER] 图编码器：hgt（默认）/ rgcn / han —— 基线对比')
parser.add_argument('--seq-only', action='store_true',
                    help='[SEQ-MLP] 同训练流程的仅序列基线：预测只用 6 维配对序列特征')
parser.add_argument('--seed', type=int, default=None, help='随机种子（默认沿用 config.seed=42）')
ARGS = parser.parse_args()
RUN_DIR = HERE / ARGS.run_dir
RUN_DIR.mkdir(exist_ok=True)
os.chdir(RUN_DIR)  # config.paths 为相对路径：切换目录后所有输出落在 run_kg_v2/ 下
sys.path.insert(0, str(HERE))

from config import config  # noqa: E402
import main  # noqa: E402

DATA_ROOT = HERE.parent / 'data'
ORIGINAL = DATA_ROOT
config.data.path_overrides = {
    'gene_expression': ORIGINAL / 'expression' / 'EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena',
    'mirna_expression': ORIGINAL / 'expression' / 'pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena',
    'llm_embeddings': DATA_ROOT / 'processed' / 'kg_v2_text_embeddings.pkl',
    'interaction_file': ORIGINAL / 'basic' / 'interaction.csv',
}
main.DATA_FILE = str(DATA_ROOT / 'processed' / ARGS.kg_file)
# 消融：去掉指定关系后写入本次运行目录，validated_interaction 不允许去掉
DROP = [r for r in ARGS.drop_relations.split(',') if r]
if DROP:
    import pandas as pd
    assert 'validated_interaction' not in DROP
    kg = pd.read_csv(main.DATA_FILE)
    unknown = set(DROP) - set(kg.relation)
    assert not unknown, f'未知关系: {unknown}'
    kg = kg[~kg.relation.isin(DROP)]
    main.DATA_FILE = str(RUN_DIR / 'kg_ablation.csv')
    kg.to_csv(main.DATA_FILE, index=False)
    print('ablation: dropped', DROP, '->', kg.relation.value_counts().to_dict(), flush=True)
main.EXPERIMENT_MODE = ARGS.mode
config.data.kg_file = main.DATA_FILE
# 与原图运行使用完全相同的训练/验证/测试实体对，保证两张图可直接比较
config.data.fixed_split_file = DATA_ROOT / 'splits' / ARGS.split_file
config.model.num_global_tokens = 0 if ARGS.global_mode == 'none' else ARGS.global_tokens
config.model.global_mode = ARGS.global_mode
config.training.matched_neg_sets = ARGS.matched_sets
config.training.early_stop_metric = ARGS.early_stop_metric
config.model.expander_degree = ARGS.expander_degree
config.model.num_virtual = ARGS.num_virtual
config.training.mask_supervised_edges = ARGS.mask
config.training.dynamic_negatives = ARGS.dynamic_neg
config.training.train_neg_ratio = ARGS.neg_ratio
if ARGS.evidence_gate:
    config.model.evidence_dim = 3
if ARGS.debias > 0:
    config.model.debias = True
    config.training.debias_lambda = ARGS.debias
config.model.mechkg = ARGS.mechkg
config.model.mechkg_k = ARGS.mechkg_k
config.model.mechkg_r = ARGS.mechkg_r
config.model.mechkg_beta = ARGS.mechkg_beta
config.model.mechkg_tau = ARGS.mechkg_tau
config.model.mechkg_residual = not ARGS.mechkg_no_residual
config.model.mechkg_drop_graph = ARGS.mechkg_drop_graph
config.model.mechkg_drop_seq = ARGS.mechkg_drop_seq
config.model.mechkg_disease_spec = ARGS.mechkg_disease_spec
config.model.mechkg_linear = ARGS.mechkg_linear
config.model.encoder = ARGS.encoder
config.model.seq_only = ARGS.seq_only
config.model.mechkg_off_channels = tuple(c for c in ARGS.mechkg_off.split(',') if c)
assert set(config.model.mechkg_off_channels) <= {'collab', 'ppi', 'pathway', 'disease', 'sequence'}, '未知通道名'
if ARGS.seed is not None:
    config.seed = ARGS.seed
    print(f'[SEED] 随机种子 = {ARGS.seed}', flush=True)
if ARGS.expr_fix:
    config.data.missing_expression = 'zero_mask'
    config.data.expression_pca = 64
if ARGS.pair_features:
    config.model.pair_feature_file = DATA_ROOT / 'processed' / ARGS.pair_feature_file
    config.model.pair_feature_dim = 6

if __name__ == '__main__':
    main.main()
