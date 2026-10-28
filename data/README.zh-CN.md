# 主模型数据清单

[English](README.md) | **简体中文**

这里的文件是主模型实际使用的输入，以及便于审计的序列和处理记录。
没有纳入全部历史数据、旧版知识图谱、TCGA 临床/生存表或网站导出。

| 目录/文件 | 用途 |
|---|---|
| `processed/kg_v2_xgft_coding_nosim.csv` | 最终知识图谱，200,224 行三元组；含 head/relation/tail 和显式节点类型 |
| `processed/kg_v2_xgft_coding_nosim.isolated_nodes.csv` | 保留 1 个孤立 miRNA，避免固定划分实体丢失 |
| `processed/kg_v2_text_embeddings.pkl` | 通路/疾病预计算文本向量，每个 768 维 |
| `processed/pair_seq_features_coding.npz` | 883 × 3303 × 6 配对特征表，含 mirnas/genes 名称数组 |
| `processed/node_texts.csv` | 用于文本嵌入的节点原始文本 |
| `processed/family_assignment.csv` | 基于成熟 miRNA seed 的家族注释，用于分析 |
| `processed/removed_noncoding_targets.csv`、`symbol_fixes.csv` | 非编码目标剔除、符号修正记录 |
| `splits/coding_named_splits.csv` | 固定五折训练/验证集以及独立测试集，含 split/miRNA/gene/label |
| `splits/hard_test_pairs.csv` | 已冻结的端点度精确匹配测试对，沿用原 hard 测试缓存 |
| `basic/interaction.csv` | 原始实验支持作用对；加载器用于实体名称识别，最终监督边以最终 KG 和固定划分为准 |
| `expression/*.xena` | 完整基因和 miRNA TCGA 表达矩阵；原有标准化与 PCA 输入全部保留 |
| `sequences/sequences.pkl` | 冻结的成熟 miRNA 和 canonical 3′UTR 序列，用于核验序列特征 |

最终节点数量：miRNA 883、Gene 3303、Pathway/GO 14071、Disease 4199。
最终知识图谱的靶向关系为 `validated_interaction`；图中包含完整靶向对，
训练/推理时必须调用 `build_message_graph` 按折移除全部目标边，再只加回该折训练正例。
不能直接把完整知识图谱的靶向边当成测试时消息图。

序列特征依次为 log1p(8mer、7mer-m8、7mer-A1、6mer 的排他位点数)、log1p(UTR 长度)、序列可用标记。
它们按全部配对无监督标准化；标准化逻辑仍在 `trainer.py` 中。
`code/sequence_features.py` 保存原始位点定义，未改动函数实现。

表达矩阵的全局列标准化和 randomized PCA 均沿用原实现，seed=42。
不要为减小上传体积只截取图中基因/miRNA 的表达行，否则会改变标准化与 PCA，已训练权重的输入也会改变。

来源概况沿用原项目记录：实验作用对来自 miRTarBase/miRATBase；表达谱为 TCGA pancan Xena；
知识图谱辅助关系包括 GO、MSigDB 通路、CTD 疾病关联和 STRING PPI；
文本向量沿用 PubMedBERT 预计算结果；成熟 miRNA 与 canonical 3′UTR 为原项目冻结的 miRBase/Ensembl 序列。
本次未重新下载、更新或重新确认这些外部数据；确切原项目文件路径见根目录 `PROVENANCE.json`。
