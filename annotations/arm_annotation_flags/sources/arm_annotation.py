"""方向 3：已验证靶点数据库中的“臂注释歧义”审计（2026-10-03）。

问题：同一发夹的 5p/3p 两臂 seed 不同，但次要臂（已验证靶点较少的一臂）的靶点约有两成与主臂重合。
这些重合靶点是真正被两臂共同调控，还是原文未分臂 / 过表达前体后被同时记到两臂上？
两条独立证据：① seed 位点（靶基因 3′UTR 上是自身臂还是另一臂的典型位点）；
② CLASHub HEK293T 嵌合读段（结合的是哪一臂；同一臂内比较“共享 / 独有”，抵消臂本身表达量差异）。
这是对标签来源的审计，不评估模型，故使用全部已验证正例（含测试集）。本地 CPU，不训练、不推理。
  python arm_annotation.py
输入：../trusted_pipeline/kg_v2/sources/mirbase_v22_hsa.gff3、splits/coding_named_splits.csv、
      data_kg_v2/pair_seq_features_coding.npz、binding_function_20260927/source/CLASH_human_HEK293T_Expanded.csv
输出：arm_annotation_20261003/summary.md、label_classes.csv（每个次要臂/主臂标签的分类）、by_hairpin.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('..')
OUT = Path('arm_annotation_20261003')
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20261003)

# ---------- 发夹 → 两臂 ----------
rows = []
for line in open(ROOT / 'trusted_pipeline/kg_v2/sources/mirbase_v22_hsa.gff3', encoding='utf-8'):
    if line.startswith('#'):
        continue
    c = line.rstrip('\n').split('\t')
    if c[2] != 'miRNA':
        continue
    attr = dict(kv.split('=', 1) for kv in c[8].split(';'))
    rows.append((attr['Name'].lower(), attr['Derives_from']))
mat = pd.DataFrame(rows, columns=['mature', 'precursor'])

S = pd.read_csv('splits/coding_named_splits.csv')
S['m'] = S.miRNA.str.lower()
# 划分文件中每个非测试配对在 5 折里各出现一次（4 次训练 + 1 次验证），先去重
pos = S[S.label == 1].drop_duplicates(['m', 'gene'])
T = pos.groupby('m').gene.apply(set).to_dict()
SPLIT = {(m, g): ('测试集' if sp == 'test' else '非测试（5 折交叉）') for m, g, sp in zip(pos.m, pos.gene, pos.split)}
NAME = dict(zip(S.m, S.miRNA))                                     # 恢复原始大小写用于展示

arm_pairs = set()
for pre, d in mat[mat.mature.isin(T)].groupby('precursor'):
    ms = sorted(set(d.mature))
    if len(ms) == 2:
        arm_pairs.add(tuple(ms))

# ---------- 序列位点 ----------
z = np.load('data_kg_v2/pair_seq_features_coding.npz')
X = z['features']
mi = {m.lower(): i for i, m in enumerate(z['mirnas'])}
gi = {g: i for i, g in enumerate(z['genes'])}
genes = np.array(list(gi))
site = lambda m, g: bool(X[mi[m], gi[g], :3].sum() > 0)            # 8mer / 7mer-m8 / 7mer-A1

# ---------- CLASH ----------
CL = pd.read_csv('binding_function_20260927/source/CLASH_human_HEK293T_Expanded.csv',
                 usecols=['miRNA_name', 'gene_name', 'WT_Detected_8', 'target_site_region'])
CL['m'] = 'hsa-' + CL.miRNA_name.str.lower()
CL = CL[CL.WT_Detected_8 >= 1]
BIND = set(zip(CL.m, CL.gene_name))
CL_MIRS = set(CL.m)

recs, bg = [], []
for a, b in arm_pairs:
    if a not in mi or b not in mi:
        continue
    minor, major = (a, b) if len(T.get(a, ())) <= len(T.get(b, ())) else (b, a)
    for arm, other, role in ((minor, major, '次要臂'), (major, minor, '主臂')):
        for g in T.get(arm, ()):
            if g not in gi:
                continue
            recs.append({'role': role, 'arm': arm, 'other_arm': other, 'gene': g, 'split': SPLIT[(arm, g)],
                         'shared': g in T.get(other, ()), 'own_site': site(arm, g), 'other_site': site(other, g),
                         'clash_own': (arm, g) in BIND, 'clash_other': (other, g) in BIND,
                         'clash_measurable': arm in CL_MIRS and other in CL_MIRS})
    for g in rng.choice(genes, 200, replace=False):          # 背景：随机基因
        bg.append({'arm': minor, 'own_site': site(minor, g), 'other_site': site(major, g),
                   'clash_own': (minor, g) in BIND, 'clash_other': (major, g) in BIND,
                   'clash_measurable': minor in CL_MIRS and major in CL_MIRS})
R = pd.DataFrame(recs)
BG = pd.DataFrame(bg)
R['class'] = np.select([R.own_site & R.other_site, R.own_site, R.other_site], ['两臂位点都有', '只有自身位点', '只有另一臂位点'], '都没有')
R['suspect'] = (R.role == '次要臂') & R.shared & ~R.own_site & R.other_site
R.to_csv(OUT / 'label_classes.csv', index=False, encoding='utf-8-sig')


def summarize(d):
    m = d[d.clash_measurable]
    return pd.Series({'标签数': len(d), '自身位点': d.own_site.mean(), '另一臂位点': d.other_site.mean(),
                      '只有另一臂位点': (~d.own_site & d.other_site).mean(),
                      'CLASH 可测': len(m), 'CLASH 自身臂结合': m.clash_own.mean() if len(m) else np.nan,
                      'CLASH 另一臂结合': m.clash_other.mean() if len(m) else np.nan})


groups = {'次要臂 · 与主臂共享': R[(R.role == '次要臂') & R.shared], '次要臂 · 独有': R[(R.role == '次要臂') & ~R.shared],
          '主臂 · 与次要臂共享': R[(R.role == '主臂') & R.shared], '主臂 · 独有': R[(R.role == '主臂') & ~R.shared],
          '背景（次要臂 × 随机基因）': BG}
TAB = pd.DataFrame({k: summarize(v) for k, v in groups.items()}).T

# 按发夹
H = (R[R.role == '次要臂'].groupby(['arm', 'other_arm'])
     .agg(次要臂标签=('gene', 'size'), 与主臂共享=('shared', 'sum'), 疑似注释歧义=('suspect', 'sum'))
     .sort_values('疑似注释歧义', ascending=False))
H.to_csv(OUT / 'by_hairpin.csv', encoding='utf-8-sig')

sus = R[R.suspect]
fmt = lambda v: f'{v:.1%}' if isinstance(v, float) and v <= 1 else (f'{int(v)}' if pd.notna(v) else '—')
L = ['# 已验证靶点的臂注释歧义审计', '',
     f'本数据集中两臂都有已验证靶点的发夹 {len(arm_pairs)} 个；次要臂 = 已验证靶点较少的一臂。全部已验证正例（含测试集）。',
     '“位点”= 8mer / 7mer-m8 / 7mer-A1；CLASH = CLASHub HEK293T 嵌合读段（≥1 个 WT 文库检出，任意区域），只统计两臂都在 CLASH 中出现过的发夹。', '',
     '| 组 | ' + ' | '.join(TAB.columns) + ' |', '|---|' + '---:|' * len(TAB.columns)]
L += [f'| {i} | ' + ' | '.join(fmt(v) for v in r) + ' |' for i, r in TAB.iterrows()]
L += ['', f'**疑似臂注释歧义**（次要臂、与主臂共享、无自身位点、有主臂位点）：{len(sus)} 个标签，'
      f'占次要臂标签 {len(sus) / (R.role == "次要臂").sum():.1%}、占全部 {len(pos)} 个已验证正例 {len(sus) / len(pos):.1%}。',
      '按划分：' + '，'.join(f'{k} {v}' for k, v in sus.split.value_counts().items()) + '。', '',
     '疑似最多的发夹（前 10）：', '', '| 次要臂 | 主臂 | 次要臂标签 | 与主臂共享 | 疑似 |', '|---|---|---:|---:|---:|']
L += [f'| {NAME.get(a, a)} | {NAME.get(b, b)} | {r.次要臂标签} | {r.与主臂共享} | {r.疑似注释歧义} |' for (a, b), r in H.head(10).iterrows()]
(OUT / 'summary.md').write_text('\n'.join(L), encoding='utf-8')
print('\n'.join(L))
