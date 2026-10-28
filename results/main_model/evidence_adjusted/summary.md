# 扣除热度后的证据排序（mk_ns_nodrop）

定义：每个证据子类型、每折，在验证集正例证据上做 OLS：φ ~ log1p(m 训练靶点数) + log1p(g 训练靶点数) + log1p(节点大小)，残差 φ̃ 为扣除热度后的贡献，只用于证据排序。拟合不使用外部结局；案例规则在看证据前固定（见脚本说明）。

## 1. 证据层的热度份额（各折 R² 均值）

| 子类型 | 证据条数/折 | R² |
|---|---:|---:|
| 共病 | 10,692 | 0.482 |
| 通路/GO | 16,697 | 0.356 |
| PPI | 680 | 0.283 |

## 2. 检验

| 检验 | 子类型 | 特征 / 参照 | 原始 φ | 校正 φ̃ |
|---|---|---|---:|---:|
| 与热度特征的 Spearman（验证集证据） | 通路/GO | m 训练靶点数 | 0.407 | -0.000 |
| 与热度特征的 Spearman（验证集证据） | 通路/GO | g 训练靶点数 | 0.279 | 0.037 |
| 与热度特征的 Spearman（验证集证据） | 通路/GO | 节点大小（主） | 0.039 | 0.010 |
| 与热度特征的 Spearman（验证集证据） | 共病 | m 训练靶点数 | -0.137 | 0.033 |
| 与热度特征的 Spearman（验证集证据） | 共病 | g 训练靶点数 | 0.599 | 0.012 |
| 与热度特征的 Spearman（验证集证据） | 共病 | 节点大小（主） | 0.003 | -0.010 |
| 与热度特征的 Spearman（验证集证据） | PPI | m 训练靶点数 | 0.179 | -0.089 |
| 与热度特征的 Spearman（验证集证据） | PPI | g 训练靶点数 | -0.182 | 0.016 |
| 与热度特征的 Spearman（验证集证据） | PPI | 节点大小（主） | -0.313 | 0.155 |
| 跨折 top-1 Jaccard（测试正例） | 共病 | 随机 0.167 | 0.486 | 0.425 |
| 跨折 top-3 Jaccard（测试正例） | 共病 | 随机 0.236 | 0.568 | 0.512 |
| 跨折 top-1 Jaccard（测试正例） | 通路/GO | 随机 0.104 | 0.370 | 0.377 |
| 跨折 top-3 Jaccard（测试正例） | 通路/GO | 随机 0.154 | 0.405 | 0.404 |
| 跨折 top-1 Jaccard（测试正例） | PPI | 随机 0.343 | 0.558 | 0.524 |
| 跨折 top-3 Jaccard（测试正例） | PPI | 随机 0.387 | 0.509 | 0.564 |
| top-1 通路节点注释基因数中位数（测试正例） | 通路/GO | 全部通路证据 65 | 81.000 | 90.000 |

## 3. 案例：miR-34a 新模型 Top20 中外部显著下调者的共识证据（≥3/5 个模型排进前 3）

主案例 = 另有 CLASH 严格 3′UTR 位点者。观测/期望 = miR-34a 训练靶点落在该通路的个数 / 按通路大小的期望（5 折平均）。

| 排名 | 基因 | 结合 | log2FC | 排序依据 | 子类型 | 名称 | 节点大小 | 前3折数 | φ | φ̃ | 观测/期望 |
|---:|---|---|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | PPARG |  | -0.66 | 校正 φ̃ | 通路/GO | cell differentiation | 102 | 3 | 0.193 | +0.042 | 12.0/5.0 |
| 1 | PPARG |  | -0.66 | 校正 φ̃ | 通路/GO | transcription cis-regulatory region binding | 131 | 3 | 0.189 | +0.039 | 14.3/6.5 |
| 1 | PPARG |  | -0.66 | 校正 φ̃ | 通路/GO | negative regulation of gene expression | 107 | 3 | 0.189 | +0.038 | 11.0/5.2 |
| 1 | PPARG |  | -0.66 | 原始 φ | 通路/GO | cell differentiation | 102 | 3 | 0.193 | +0.042 | 12.0/5.0 |
| 1 | PPARG |  | -0.66 | 原始 φ | 通路/GO | negative regulation of gene expression | 107 | 3 | 0.189 | +0.038 | 11.0/5.2 |
| 3 | HSPA1B | ✓ | -0.44 | 校正 φ̃ | 通路/GO | centrosome | 124 | 5 | 0.197 | +0.056 | 10.4/6.1 |
| 3 | HSPA1B | ✓ | -0.44 | 原始 φ | 通路/GO | centrosome | 124 | 5 | 0.197 | +0.056 | 10.4/6.1 |
| 4 | IKBKE |  | -0.84 | 校正 φ̃ | 通路/GO | ubiquitin protein ligase binding | 139 | 3 | 0.242 | +0.124 | 16.0/6.9 |
| 4 | IKBKE |  | -0.84 | 校正 φ̃ | 通路/GO | immune response | 90 | 4 | 0.241 | +0.122 | 9.2/4.5 |
| 4 | IKBKE |  | -0.84 | 原始 φ | 通路/GO | ubiquitin protein ligase binding | 139 | 3 | 0.242 | +0.124 | 16.0/6.9 |
| 4 | IKBKE |  | -0.84 | 原始 φ | 通路/GO | immune response | 90 | 4 | 0.241 | +0.122 | 9.2/4.5 |
| 7 | NOTCH3 |  | -0.58 | 校正 φ̃ | 通路/GO | receptor complex | 91 | 5 | 0.223 | +0.072 | 11.2/4.5 |
| 7 | NOTCH3 |  | -0.58 | 校正 φ̃ | 通路/GO | WP_EPITHELIAL_TO_MESENCHYMAL_TRANSITION_IN_COLORECTAL_CANCER | 114 | 4 | 0.217 | +0.066 | 11.8/5.6 |
| 7 | NOTCH3 |  | -0.58 | 校正 φ̃ | 通路/GO | WP_NEURAL_CREST_DIFFERENTIATION | 72 | 3 | 0.198 | +0.046 | 10.0/3.6 |
| 7 | NOTCH3 |  | -0.58 | 校正 φ̃ | 通路/GO | WP_BREAST_CANCER_PATHWAY | 119 | 3 | 0.194 | +0.043 | 16.3/5.9 |
| 7 | NOTCH3 |  | -0.58 | 原始 φ | 通路/GO | receptor complex | 91 | 5 | 0.223 | +0.072 | 11.2/4.5 |
| 7 | NOTCH3 |  | -0.58 | 原始 φ | 通路/GO | WP_EPITHELIAL_TO_MESENCHYMAL_TRANSITION_IN_COLORECTAL_CANCER | 114 | 4 | 0.217 | +0.066 | 11.8/5.6 |
| 7 | NOTCH3 |  | -0.58 | 原始 φ | 通路/GO | WP_NEURAL_CREST_DIFFERENTIATION | 72 | 3 | 0.198 | +0.046 | 10.0/3.6 |
| 7 | NOTCH3 |  | -0.58 | 原始 φ | 通路/GO | WP_BREAST_CANCER_PATHWAY | 119 | 3 | 0.194 | +0.043 | 16.3/5.9 |
| 9 | PKP4 |  | -0.30 | 校正 φ̃ | 通路/GO | perinuclear region of cytoplasm | 166 | 5 | 0.269 | +0.151 | 8.8/8.2 |
| 9 | PKP4 |  | -0.30 | 校正 φ̃ | 通路/GO | adherens junction | 53 | 4 | 0.259 | +0.138 | 5.8/2.6 |
| 9 | PKP4 |  | -0.30 | 校正 φ̃ | 通路/GO | cell-cell signaling | 75 | 4 | 0.235 | +0.115 | 6.5/3.7 |
| 9 | PKP4 |  | -0.30 | 原始 φ | 通路/GO | perinuclear region of cytoplasm | 166 | 5 | 0.269 | +0.151 | 8.8/8.2 |
| 9 | PKP4 |  | -0.30 | 原始 φ | 通路/GO | adherens junction | 53 | 4 | 0.259 | +0.138 | 5.8/2.6 |
| 9 | PKP4 |  | -0.30 | 原始 φ | 通路/GO | cell-cell signaling | 75 | 4 | 0.235 | +0.115 | 6.5/3.7 |
| 10 | ALDOA |  | -0.39 | 原始 φ | 通路/GO | WP_CLEAR_CELL_RENAL_CELL_CARCINOMA_PATHWAYS | 48 | 3 | 0.198 | +0.067 | 5.0/2.4 |
| 11 | RRAS |  | -1.69 | 校正 φ̃ | 通路/GO | WP_INTEGRATED_BREAST_CANCER_PATHWAY | 107 | 4 | 0.151 | +0.015 | 16.8/5.3 |
| 11 | RRAS |  | -1.69 | 校正 φ̃ | 通路/GO | GTPase activity | 85 | 4 | 0.146 | +0.009 | 5.0/4.2 |
| 11 | RRAS |  | -1.69 | 原始 φ | 通路/GO | WP_INTEGRATED_BREAST_CANCER_PATHWAY | 107 | 5 | 0.151 | +0.015 | 16.8/5.3 |
| 11 | RRAS |  | -1.69 | 原始 φ | 通路/GO | GTPase activity | 85 | 3 | 0.146 | +0.009 | 5.0/4.2 |
| 11 | RRAS |  | -1.69 | 原始 φ | 通路/GO | WP_EGFR_TYROSINE_KINASE_INHIBITOR_RESISTANCE | 70 | 3 | 0.139 | +0.001 | 13.0/3.5 |
| 13 | CASP2 |  | -0.48 | 校正 φ̃ | 通路/GO | enzyme binding | 156 | 3 | 0.232 | +0.084 | 16.7/7.8 |
| 13 | CASP2 |  | -0.48 | 校正 φ̃ | 通路/GO | positive regulation of apoptotic process | 123 | 4 | 0.228 | +0.079 | 10.5/6.1 |
| 13 | CASP2 |  | -0.48 | 校正 φ̃ | 通路/GO | DNA damage response | 110 | 3 | 0.228 | +0.079 | 7.3/5.5 |
| 13 | CASP2 |  | -0.48 | 原始 φ | 通路/GO | enzyme binding | 156 | 3 | 0.232 | +0.084 | 16.7/7.8 |
| 13 | CASP2 |  | -0.48 | 原始 φ | 通路/GO | positive regulation of apoptotic process | 123 | 4 | 0.228 | +0.079 | 10.5/6.1 |
| 13 | CASP2 |  | -0.48 | 原始 φ | 通路/GO | DNA damage response | 110 | 3 | 0.228 | +0.079 | 7.3/5.5 |
| 14 | HMGA1 |  | -0.23 | 校正 φ̃ | 通路/GO | regulation of DNA-templated transcription | 159 | 4 | 0.246 | +0.083 | 13.5/7.9 |
| 14 | HMGA1 |  | -0.23 | 原始 φ | 通路/GO | regulation of DNA-templated transcription | 159 | 4 | 0.246 | +0.083 | 13.5/7.9 |
| 19 | EFNB1 |  | -0.64 | 校正 φ̃ | 通路/GO | synapse | 59 | 3 | 0.224 | +0.092 | 3.7/2.9 |
| 19 | EFNB1 |  | -0.64 | 校正 φ̃ | 通路/GO | cell adhesion | 112 | 3 | 0.211 | +0.081 | 6.3/5.6 |
| 19 | EFNB1 |  | -0.64 | 校正 φ̃ | 通路/GO | cell-cell signaling | 75 | 3 | 0.204 | +0.072 | 6.7/3.7 |
| 19 | EFNB1 |  | -0.64 | 原始 φ | 通路/GO | synapse | 59 | 4 | 0.224 | +0.092 | 3.8/2.9 |
| 19 | EFNB1 |  | -0.64 | 原始 φ | 通路/GO | cell adhesion | 112 | 4 | 0.211 | +0.081 | 6.5/5.6 |
| 19 | EFNB1 |  | -0.64 | 原始 φ | 通路/GO | cell-cell signaling | 75 | 4 | 0.204 | +0.072 | 6.5/3.7 |

校正后没有任何证据达到 ≥3/5 折前 3 共识的案例：ALDOA。
