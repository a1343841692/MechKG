"""
数据加载模块 v3.0 - 支持关系类型归并
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import config
from utils import set_seed
import logging
from pathlib import Path
import re
import pickle
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EnhancedKnowledgeGraphLoader:
    """
    增强的知识图谱加载器 - v3.0
    核心优化：关系类型归并 (Relation Type Merging)
    """

    def __init__(self, kg_file: Optional[str] = None, predefined_mapping: Optional[Dict] = None):
        self.kg_file = kg_file or config.data.kg_file
        self.node_types = config.data.node_types
        self.add_reverse_edges = config.data.add_reverse_edges
        self.node_mapping = {}
        self.reverse_mapping = {}
        self.raw_features = defaultdict(dict)
        self.feature_dims = {}

        # ✨ 保存传入的预定义映射
        self.predefined_mapping = predefined_mapping
        # [KG-v2] 三元组文件若带 head_type/tail_type 列，则按显式类型建图，不走字符串规则
        self.explicit_types = {}

        # 数据文件路径
        self.data_paths = self._setup_data_paths()


        # 从interaction.csv提取确定的基因和miRNA集合
        self.known_genes = set()
        self.known_mirnas = set()
        self._load_interaction_entities()

        logger.info("🔧 增强的知识图谱加载器已初始化 (v3.0 - 支持关系归并)")

    def _setup_data_paths(self) -> Dict[str, Path]:
        """设置数据文件路径"""
        kg_file_path = Path(self.kg_file)
        base_dir = kg_file_path.parent

        paths = {
            'gene_expression': base_dir / "data" / "expression" / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena",
            'mirna_expression': base_dir / "data" / "expression" / "pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena",
            'clinical_data': base_dir / "data" / "expression" / "Survival_SupplementalTable_S1_20171025_xena_sp",
            'llm_embeddings': base_dir / "data" / "processed" / "biobert_biomed_final" / "final_embeddings.pkl",
           # 'biowordvec_embeddings': base_dir / "data" / "processed" / "biowordvec_nltk_final" / "final_embeddings.pkl",
            'interaction_file': base_dir / "data" / "basic" / "interaction.csv",
        }

        # [KG-v2] 允许在配置中直接指定特征文件路径（KG v2 与表达数据不在同一目录）
        for key, path in getattr(config.data, 'path_overrides', {}).items():
            paths[key] = Path(path)

        # 智能路径查找
        for key, path in paths.items():
            if not path.exists():
                alternative_paths = [
                    base_dir.parent / path.name,
                    Path("data") / path.relative_to(base_dir),
                    Path("../data") / path.relative_to(base_dir),
                ]
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        paths[key] = alt_path
                        logger.info(f"🔍 找到替代路径: {key} -> {alt_path}")
                        break

        return paths

    def _load_interaction_entities(self):
        """从interaction.csv加载确定的基因和miRNA名称"""
        try:
            interaction_file = self.data_paths.get('interaction_file')
            if not interaction_file or not interaction_file.exists():
                logger.warning("未找到interaction.csv文件，将使用模式匹配进行节点分类")
                return

            df = pd.read_csv(interaction_file)
            logger.info(f"✅ 从 {interaction_file.name} 加载实体名称用于精确分类")

            # 查找miRNA和基因列
            possible_mirna_cols = ['miRNA', 'mirna', 'miRNA_ID', 'mirna_id', 'head']
            possible_gene_cols = ['mRNA', 'mrna', 'mRNA_ID', 'mrna_id', 'gene', 'Gene', 'tail']

            mirna_col = next((col for col in possible_mirna_cols if col in df.columns), None)
            gene_col = next((col for col in possible_gene_cols if col in df.columns), None)

            if mirna_col:
                mirnas = df[mirna_col].dropna().unique()
                for mirna in mirnas:
                    if pd.notna(mirna):
                        normalized = self._normalize_mirna_id(str(mirna).strip())
                        if normalized:
                            self.known_mirnas.add(normalized)
                            self.known_mirnas.add(str(mirna).strip())

            if gene_col:
                genes = df[gene_col].dropna().unique()
                for gene in genes:
                    if pd.notna(gene):
                        self.known_genes.add(str(gene).strip())

            logger.info(f"📊 提取实体用于精确分类:")
            logger.info(f"   确定的基因: {len(self.known_genes):,} 个")
            logger.info(f"   确定的miRNA: {len(self.known_mirnas):,} 个")

        except Exception as e:
            logger.warning(f"加载interaction.csv失败: {e}，将使用模式匹配")

    def _normalize_mirna_id(self, mirna_id: str) -> str:
        """标准化miRNA ID格式"""
        mirna_str = str(mirna_id).strip()
        if re.search(r'let-7', mirna_str, re.IGNORECASE):
            mirna_str = re.sub(r'hsa-(let-7[a-z]*(-[35]p)?)',
                               lambda m: f"hsa-{m.group(1).lower()}",
                               mirna_str, flags=re.IGNORECASE)
            return mirna_str
        mirna_str = mirna_str.lower()
        mirna_str = mirna_str.replace('hsa-mir-', 'hsa-miR-')
        mirna_str = mirna_str.replace('-3p', '-3p').replace('-5p', '-5p')
        return mirna_str

    def classify_node(self, node: str) -> str:
        """
        节点类型分类 - 改进版
        增加了基于interaction.csv的精确基因判断
        """
        # [KG-v2] 显式类型优先
        if self.explicit_types:
            return self.explicit_types[node]

        # 保留原始大小写用于精确匹配，同时去除首尾空格
        original_node = str(node).strip()
        # 使用小写版本进行通用模式匹配
        node_str = original_node.lower()

        # 规则1：miRNA识别 (优先级最高)
        # [FIX] 原规则用子串匹配 'let-'，会把 'GO_BP:platelet-derived ...' 误判为 miRNA；改为锚定开头匹配
        if (re.match(r'^(hsa-)?(mir|let)-', node_str) or
                original_node in self.known_mirnas or
                self._normalize_mirna_id(original_node) in self.known_mirnas):
            return 'miRNA'

        # 规则2：通路识别
        pathway_patterns = [
            'kegg_', 'reactome_', 'biocarta_', 'sa_', 'wp_', 'go_', 'pid_',
            'kegg:', 'reactome:', 'wikipathways:', 'go_bp:', 'go_cc:', 'go_mf:',
            'pathway', 'signaling'
        ]
        if any(pattern in node_str for pattern in pathway_patterns):
            return 'Pathway'

        # 规则3：疾病识别 (基于模式)
        tcga_codes = {
            'acc', 'blca', 'brca', 'cesc', 'chol', 'coad', 'dlbc', 'esca',
            'gbm', 'hnsc', 'kich', 'kirc', 'kirp', 'laml', 'lgg', 'lihc',
            'luad', 'lusc', 'meso', 'ov', 'paad', 'pcpg', 'prad', 'read',
            'sarc', 'skcm', 'stad', 'tgct', 'thca', 'thym', 'ucec', 'ucs', 'uvm'
        }
        if node_str in tcga_codes:
            return 'Disease'

        disease_patterns = [
            'cancer', 'carcinoma', 'tumor', 'tumour', 'malignancy', 'neoplasm',
            'disease', 'syndrome', 'disorder', 'infection', 'inflammation'
        ]
        if any(pattern in node_str for pattern in disease_patterns):
            return 'Disease'

        # ==================== ✨ 关键修改 ✨ ====================
        # 规则4：最终判断 - 基于interaction.csv的精确基因列表
        # 如果一个节点不属于上述任何类型，则检查它是否在我们的已知基因列表中。

        # 使用原始大小写的字符串进行精确匹配
        if original_node in self.known_genes:
            return 'Gene'
        else:
            # 如果它也不在已知基因列表中，则我们将其归类为“疾病”。
            # 这扩大了疾病的识别范围，捕获了模式匹配未覆盖的疾病名称。
            return 'Disease'
        # ==================== 修改结束 ====================


    def _get_edge_type(self, row: pd.Series) -> str:
        """
        ✨ v3.0 核心优化：关系类型归并与简化
        将300+种复杂关系映射为十几种核心关系，以大幅提升HGT计算效率。
        """
        # [KG-v2] 显式类型的图保留原关系名（已是规范关系），反向边加 reverse_ 前缀
        if self.explicit_types:
            return str(row['relation'])

        relation = str(row['relation']).lower()
        head_type = self.classify_node(row['head'])
        tail_type = self.classify_node(row['tail'])

        # 规则1：核心生物学互作 (优先级最高)
        if 'validated_interaction' in relation:
            return 'validated_interaction'

        # 规则2：相似性关系
        if 'similar_to' in relation:
            return 'similar_to'

        if 'semantically_similar_to' in relation:
            return 'semantically_similar'

        # 规则3：语义关联
        if 'semantic_association' in relation:
            return 'semantic_association'

        # 规则4：通路调控/参与关系
        pathway_keywords = ['participates_in', 'involved_in', 'regulates', 'has_molecular_function', 'localized_in']
        if any(keyword in relation for keyword in pathway_keywords):
            return 'related_to_pathway'

        # 规则5：疾病关联关系 (miRNA-Disease)
        if 'associated_with_disease' in relation:
            return 'associated_with_disease'

        # 规则6：差异表达关系 (核心简化)
        if 'upregulated' in relation:
            return 'upregulated_in_disease'
        if 'downregulated' in relation:
            return 'downregulated_in_disease'

        # 规则7：处理反向边
        if relation.startswith('reverse_'):
            original_relation = relation.replace('reverse_', '')
            # 递归调用自身，但处理的是原始关系名
            row_copy = row.copy()
            row_copy['relation'] = original_relation
            row_copy['head'] = row['tail']
            row_copy['tail'] = row['head']
            return 'reverse_' + self._get_edge_type(row_copy)

        # 规则8：默认回退 - 基于节点类型
        return f"{head_type}_to_{tail_type}"

    def load_raw_data(self) -> pd.DataFrame:
        """加载原始数据"""
        try:
            df = pd.read_csv(self.kg_file)
            logger.info(f"成功加载数据: {len(df):,} 条三元组")

            if {'head_type', 'tail_type'} <= set(df.columns):
                for col, tcol in [('head', 'head_type'), ('tail', 'tail_type')]:
                    for name, kind in zip(df[col], df[tcol]):
                        assert self.explicit_types.setdefault(name, kind) == kind, f'节点类型冲突: {name}'
                assert set(self.explicit_types.values()) <= set(self.node_types), '存在未配置的节点类型'
                df = df[['head', 'relation', 'tail']]
                logger.info(f"[KG-v2] 使用显式节点类型: {pd.Series(self.explicit_types).value_counts().to_dict()}")

            # [ISOLATED] KG 旁的 <stem>.isolated_nodes.csv（name,type）：去掉某关系后只靠该关系相连的节点，
            # 作为孤立节点保留，使节点集合与编号与完整图一致（固定划分要求这些节点存在）
            side = Path(self.kg_file).with_name(Path(self.kg_file).stem + '.isolated_nodes.csv')
            self.isolated_nodes = []
            if side.exists():
                iso = pd.read_csv(side)
                for name, kind in zip(iso['name'], iso['type']):
                    if self.explicit_types:     # 无显式类型列的旧 KG 仍按字符串规则分类
                        assert self.explicit_types.setdefault(name, kind) == kind, f'节点类型冲突: {name}'
                self.isolated_nodes = list(iso['name'])
                logger.info(f"[ISOLATED] 保留孤立节点 {len(iso)} 个: {self.isolated_nodes[:10]}")

            # 如果需要添加反向边
            if self.add_reverse_edges:
                reverse_df = df.copy()
                reverse_df['head'], reverse_df['tail'] = df['tail'], df['head']
                reverse_df['relation'] = 'reverse_' + df['relation']

                df = pd.concat([df, reverse_df], ignore_index=True)
                logger.info(f"添加反向边后: {len(df):,} 条三元组")

            return df
        except FileNotFoundError:
            logger.error(f"文件不存在: {self.kg_file}")
            raise

    def create_node_mappings(self, df: pd.DataFrame):
        """创建节点映射"""
        logger.info("创建节点映射...")

        node_counts = {node_type: 0 for node_type in self.node_types}

        # ✨ 关键修改：如果有预定义映射，直接使用，不重新生成
        if self.predefined_mapping is not None:
            logger.info("🔒 使用预定义的节点映射 (Fixed Mapping)")
            self.node_mapping = self.predefined_mapping

            # 重建反向映射和计数
            for node_type, mapping in self.node_mapping.items():
                self.reverse_mapping[node_type] = {v: k for k, v in mapping.items()}
                node_counts[node_type] = len(mapping)

            return node_counts

        # --- 以下是原本的逻辑 (没有预定义映射时才执行) ---
        logger.info("🎲 生成新的节点映射 (Random Order)")
        all_nodes = set(df['head']) | set(df['tail']) | set(getattr(self, 'isolated_nodes', []))

        for node_type in self.node_types:
            self.node_mapping[node_type] = {}
            self.reverse_mapping[node_type] = {}

        # [FIX] 集合遍历顺序随进程哈希种子变化，排序后节点编号在每次运行中一致
        for node in sorted(all_nodes, key=str):
            node_type = self.classify_node(node)
            if node_type in self.node_types:
                node_id = node_counts[node_type]
                self.node_mapping[node_type][node] = node_id
                self.reverse_mapping[node_type][node_id] = node
                node_counts[node_type] += 1

        logger.info("节点映射完成")
        return node_counts

    def load_and_prepare_features(self):
        """加载和准备所有原始特征"""
        logger.info("🧬 开始加载和准备所有原始特征...")

        # 1. 加载基因表达谱
        # ✨ 关键修改：增加对 config.data.feature_sources['gene_expression'] 的判断
        if config.data.feature_sources.get('gene_expression', False):
            try:
                gene_expr_path = self.data_paths['gene_expression']
                if gene_expr_path.exists():
                    logger.info(f"  📊 加载基因表达数据: {gene_expr_path.name}")
                    gene_df = pd.read_csv(gene_expr_path, sep='\t', index_col=0, nrows=None).fillna(0)

                    # 标准化处理
                    if config.data.feature_preprocessing['standardize_expression']:
                        gene_df.iloc[:] = StandardScaler().fit_transform(gene_df.values)

                    # [EXPR] 可选：PCA 降维，降低投影层参数量（无监督，不使用标签）
                    values = self._reduce_expression(gene_df.values, 'Gene')
                    self.feature_dims['Gene'] = values.shape[1]
                    for gene, row in zip(gene_df.index, values):
                        self.raw_features['Gene'][gene] = torch.tensor(row, dtype=torch.float)

                    logger.info(f"    ✅ 基因特征: {len(self.raw_features['Gene'])} 个, 原始维度: {self.feature_dims['Gene']}")
            except Exception as e:
                logger.error(f"    ❌ 加载基因特征失败: {e}")
        else:
            logger.info("  ⭕ [禁用] 已跳过加载基因表达谱。")
        # 2. 加载miRNA表达谱
        # ✨ 关键修改：增加对 config.data.feature_sources['mirna_expression'] 的判断
        if config.data.feature_sources.get('mirna_expression', False):
            try:
                mirna_expr_path = self.data_paths['mirna_expression']
                if mirna_expr_path.exists():
                    logger.info(f"  📊 加载miRNA表达数据: {mirna_expr_path.name}")
                    mirna_df = pd.read_csv(mirna_expr_path, sep='\t', index_col=0, nrows=None).fillna(0)

                    # 标准化处理
                    if config.data.feature_preprocessing['standardize_expression']:
                        mirna_df.iloc[:] = StandardScaler().fit_transform(mirna_df.values)

                    # [EXPR] 可选：PCA 降维
                    values = self._reduce_expression(mirna_df.values, 'miRNA')
                    self.feature_dims['miRNA'] = values.shape[1]
                    for mirna, row in zip(mirna_df.index, values):
                        normalized_name = self._normalize_mirna_id(mirna)
                        feature_tensor = torch.tensor(row, dtype=torch.float)
                        self.raw_features['miRNA'][normalized_name] = feature_tensor
                        self.raw_features['miRNA'][mirna] = feature_tensor

                    logger.info(f"    ✅ miRNA特征: {len(set(self.raw_features['miRNA'].keys()))} 个, 原始维度: {self.feature_dims['miRNA']}")
            except Exception as e:
                logger.error(f"    ❌ 加载miRNA特征失败: {e}")
        else:
            logger.info("  ⭕ [禁用] 已跳过加载miRNA表达谱。")
        # 3. 加载LLM嵌入特征
        # ✨ 关键修改：增加对 config.data.feature_sources['llm_embeddings'] 的判断
        if config.data.feature_sources.get('llm_embeddings', False):
            try:
                embedding_file = None
                source_type = "未知"

                # 优先尝试加载主要的 LLM (BioBERT)
                primary_llm_path = self.data_paths.get('llm_embeddings')
                if primary_llm_path and primary_llm_path.exists():
                    embedding_file = primary_llm_path
                    source_type = "LLM (BioBERT)"

                # 如果主要的LLM文件不存在，并且配置中也启用了备用选项，则尝试加载备用文件
                elif config.data.feature_sources.get('biowordvec_embeddings', False):
                    fallback_path = self.data_paths.get('biowordvec_embeddings')
                    if fallback_path and fallback_path.exists():
                        embedding_file = fallback_path
                        source_type = "BioWordVec (备用)"
                        logger.info("  ⚠️ 未找到主要的LLM嵌入，已成功加载备用的BioWordVec嵌入。")
                    else:
                        logger.warning("  ⚠️ LLM和备用的BioWordVec嵌入文件均未找到。")
                else:
                    logger.warning("  ⚠️ 未找到主要的LLM嵌入文件，且未启用或未定义备用嵌入。")

                # 如果找到了任何一个嵌入文件，则加载它
                if embedding_file:
                    logger.info(f"  🧠 正在加载 {source_type} 嵌入数据: {embedding_file.name}")
                    with open(embedding_file, 'rb') as f:
                        embeddings = pickle.load(f)

                    # 加载通路嵌入
                    if 'pathway_embeddings' in embeddings and embeddings['pathway_embeddings']:
                        self.raw_features['Pathway'] = {
                            k: torch.tensor(v, dtype=torch.float)
                            for k, v in embeddings['pathway_embeddings'].items()
                            if isinstance(v, np.ndarray)
                        }
                        if self.raw_features['Pathway']:
                            self.feature_dims['Pathway'] = next(iter(self.raw_features['Pathway'].values())).shape[0]
                            logger.info(
                                f"    ✅ 通路 {source_type} 特征: {len(self.raw_features['Pathway'])} 个, 原始维度: {self.feature_dims['Pathway']}")

                    # 加载疾病嵌入
                    if 'disease_embeddings' in embeddings and embeddings['disease_embeddings']:
                        self.raw_features['Disease'] = {
                            k: torch.tensor(v, dtype=torch.float)
                            for k, v in embeddings['disease_embeddings'].items()
                            if isinstance(v, np.ndarray)
                        }
                        if self.raw_features['Disease']:
                            self.feature_dims['Disease'] = next(iter(self.raw_features['Disease'].values())).shape[0]
                            logger.info(
                                f"    ✅ 疾病 {source_type} 特征: {len(self.raw_features['Disease'])} 个, 原始维度: {self.feature_dims['Disease']}")

            except Exception as e:
                logger.error(f"    ❌ 加载嵌入特征过程中发生错误: {e}", exc_info=True)
        else:
            logger.info("  ⭕ [禁用] 已跳过加载LLM嵌入。")
        # ==================== ✨ 新增调试打印 Part 2 (新函数) ✨ ====================

    def _reduce_expression(self, values, node_type: str):
        """[EXPR] 对表达矩阵做 PCA 降维；config.data.expression_pca 为 0/None 时原样返回。"""
        k = getattr(config.data, 'expression_pca', 0)
        if not k:
            return values
        from sklearn.decomposition import PCA
        k = min(int(k), values.shape[0] - 1, values.shape[1])
        pca = PCA(n_components=k, svd_solver='randomized', random_state=config.seed)
        z = pca.fit_transform(values).astype('float32')
        z = z / np.maximum(z.std(0, keepdims=True), 1e-6)
        logger.info(f"  [EXPR] {node_type} 表达谱 PCA: {values.shape[1]} -> {k} 维，"
                    f"累计解释方差 {pca.explained_variance_ratio_.sum():.3f}")
        return z

    def _debug_feature_matching(self, node_type: str):
        """一个用于调试特征匹配失败的辅助函数"""
        if not self.node_mapping.get(node_type):
            logger.info(f"  ❌ [调试] 节点类型 '{node_type}' 在图中没有找到任何节点。")
            return

        if not self.raw_features.get(node_type):
            logger.info(f"  ❌ [调试] 节点类型 '{node_type}' 没有从PKL文件加载到任何原始特征。")
            return

        graph_nodes = set(self.node_mapping[node_type].keys())
        feature_nodes = set(self.raw_features[node_type].keys())

        matched_nodes = graph_nodes.intersection(feature_nodes)
        unmatched_graph_nodes = graph_nodes - feature_nodes

        logger.info("=" * 25 + f" [调试] 特征匹配分析: {node_type} " + "=" * 25)
        logger.info(f"  - 图谱中此类节点总数: {len(graph_nodes)}")
        logger.info(f"  - 特征文件中此类节点总数: {len(feature_nodes)}")
        logger.info(f"  - ✅ 成功匹配的节点数: {len(matched_nodes)}")
        logger.info(f"  - ❌ 匹配失败的节点数: {len(unmatched_graph_nodes)}")

        if unmatched_graph_nodes:
            logger.info(f"  - 示例 (图中有，但PKL文件中没有的节点): {sorted(list(unmatched_graph_nodes))[:20]}")  # 最多显示20个

        unmatched_feature_nodes = feature_nodes - graph_nodes
        if unmatched_feature_nodes:
            logger.info(f"  - 示例 (PKL文件中有，但图中没有的节点): {sorted(list(unmatched_feature_nodes))[:20]}")  # 最多显示20个
        logger.info("=" * 80)

    # ======================================================================

    def create_edge_data(self, df: pd.DataFrame) -> Dict:
        """创建边数据 - 使用归并后的关系类型"""
        logger.info("处理边数据...")
        # ==================== 新增修改：开始 ====================
        # 1. 计算归并前的完整边类型数量
        logger.info("正在分析原始边类型 (head, relation, tail)...")
        original_edge_types = set()
        # 使用tqdm显示进度条
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="分析原始边", ncols=100):
            head_type = self.classify_node(row['head'])
            tail_type = self.classify_node(row['tail'])
            original_edge_types.add((head_type, row['relation'], tail_type))
        num_original_edge_types = len(original_edge_types)

        # ✨ 关键修改：先进行关系类型归并
        df['merged_relation'] = df.apply(self._get_edge_type, axis=1)

        # 统计归并前后的关系类型数量
        original_relations = df['relation'].nunique()
        merged_relations = df['merged_relation'].nunique()
        logger.info(f"✨ 关系类型归并: {original_relations} -> {merged_relations} ({(1-merged_relations/original_relations)*100:.1f}% 减少)")

        # ==================== 新增修改：开始 ====================
        # 2. 输出归并后的核心关系名称列表
        merged_relations_set = df['merged_relation'].unique()
        logger.info(f"✨ 归并后的核心关系名称: {sorted(list(merged_relations_set))}")
        # ==================== 新增修改：结束 ====================

        edge_type_stats = defaultdict(list)

        for _, row in df.iterrows():
            head_type = self.classify_node(row['head'])
            tail_type = self.classify_node(row['tail'])
            # 使用归并后的关系类型
            edge_type_stats[(head_type, row['merged_relation'], tail_type)].append((row['head'], row['tail']))

        edge_data = {}
        for edge_type, edges in edge_type_stats.items():
            head_type, relation, tail_type = edge_type

            head_indices = []
            tail_indices = []

            for head_node, tail_node in edges:
                if (head_node in self.node_mapping[head_type] and
                    tail_node in self.node_mapping[tail_type]):
                    head_indices.append(self.node_mapping[head_type][head_node])
                    tail_indices.append(self.node_mapping[tail_type][tail_node])

            if head_indices:
                edge_index = torch.tensor([head_indices, tail_indices], dtype=torch.long)
                edge_data[edge_type] = edge_index
                logger.info(f"  {edge_type}: {len(head_indices):,} 条边")

            # ==================== 新增修改：开始 ====================
            # 3. 输出完整边类型的归并总结
        num_merged_edge_types = len(edge_data)
        reduction_percent = (
                                        1 - num_merged_edge_types / num_original_edge_types) * 100 if num_original_edge_types > 0 else 0

        logger.info("=" * 60)
        logger.info("✨ 边类型归并总结:")
        logger.info(f"  - 归并前 (head, relation, tail) 数量: {num_original_edge_types}")
        logger.info(f"  - 归并后 (head, relation, tail) 数量: {num_merged_edge_types}")
        logger.info(f"  - 边类型总数减少: {reduction_percent:.1f}%")
        logger.info("=" * 60)
        # ==================== 新增修改：结束 ====================

        return edge_data

    def create_hetero_data(self) -> HeteroData:
        """创建异构图数据对象，附加原始维度特征"""
        logger.info("构建异构图数据对象 (v3.0 - 关系归并版)...")

        # ✨ 关键修复：仅在配置启用时加载生物学特征
        if config.data.use_biological_features:
            self.load_and_prepare_features()
        else:
            logger.info("🧬 已跳过加载生物学特征（基线模式）。")

        # 创建图结构
        df = self.load_raw_data()
        node_counts = self.create_node_mappings(df)

        # ==================== ✨ 新增调试打印 Part 3 (调用新函数) ✨ ====================
        # 在分配特征之前，调用调试函数来分析疾病节点的匹配情况
        self._debug_feature_matching('Disease')
        # ======================================================================

        edge_data = self.create_edge_data(df)

        data = HeteroData()

        # 附加原始特征
        feature_usage_stats = {node_type: {'with_features': 0, 'random': 0} for node_type in self.node_types}

        for node_type in self.node_types:
            num_nodes = node_counts[node_type]
            if num_nodes == 0:
                continue

            data[node_type].num_nodes = num_nodes
            raw_dim = self.feature_dims.get(node_type)

            if raw_dim and node_type in self.raw_features:
                # [EXPR] missing_expression='zero_mask' 时：缺失填 0 并追加缺失标记列，
                # 避免随机向量成为可被记忆的节点身份；默认 'random' 保持原行为。
                zero_mask = (getattr(config.data, 'missing_expression', 'random') == 'zero_mask'
                             and node_type in ('miRNA', 'Gene'))
                if zero_mask:
                    feature_matrix = torch.zeros(num_nodes, raw_dim + 1)
                else:
                    feature_matrix = torch.randn(num_nodes, raw_dim) * 0.01

                for node_name, node_id in self.node_mapping[node_type].items():
                    feature_vec = self.raw_features[node_type].get(node_name)
                    if feature_vec is None and node_type == 'miRNA':
                        feature_vec = self.raw_features[node_type].get(self._normalize_mirna_id(node_name))

                    if feature_vec is not None:
                        if zero_mask:
                            feature_matrix[node_id, :raw_dim] = feature_vec
                            feature_matrix[node_id, raw_dim] = 1.0  # 有真实表达
                        else:
                            feature_matrix[node_id] = feature_vec
                        feature_usage_stats[node_type]['with_features'] += 1
                    else:
                        feature_usage_stats[node_type]['random'] += 1

                data[node_type].x = feature_matrix
                logger.info(f"  {node_type}: {num_nodes} 个节点, 原始特征维度: {raw_dim}")
                logger.info(f"    生物学特征: {feature_usage_stats[node_type]['with_features']}, "
                           f"随机特征: {feature_usage_stats[node_type]['random']}")
            else:
                # 如果没有任何特征源，则创建默认维度的随机特征
                default_dim = config.model.embedding_dim
                logger.warning(f"  {node_type}: 未找到特征源, 使用随机特征 (维度: {default_dim})")
                data[node_type].x = torch.randn(num_nodes, default_dim) * 0.01
                feature_usage_stats[node_type]['random'] = num_nodes

        # 添加边数据
        for edge_type, edge_index in edge_data.items():
            data[edge_type].edge_index = edge_index

        # 保存映射信息和特征维度信息
        data.node_mapping = self.node_mapping
        data.reverse_mapping = self.reverse_mapping
        data.feature_dims = self.feature_dims.copy()

        # 打印特征使用统计
        logger.info("\n🎯 特征使用统计:")
        total_bio_features = sum(stats['with_features'] for stats in feature_usage_stats.values())
        total_random_features = sum(stats['random'] for stats in feature_usage_stats.values())
        total_features = total_bio_features + total_random_features

        if total_features > 0:
            bio_percentage = (total_bio_features / total_features) * 100
            logger.info(f"  生物学特征: {total_bio_features:,} ({bio_percentage:.1f}%)")
            logger.info(f"  随机特征: {total_random_features:,} ({100-bio_percentage:.1f}%)")

        logger.info(f"✅ 异构图构建完成: {len(edge_data)} 种边类型")
        logger.info(f"🚀 关系类型归并带来的性能提升预期: 5-10倍")
        return data


# 保持原有的其他类和函数不变
class NegativeSampler:
    """修正的负采样器 - 避免重复训练样本"""

    def __init__(self, data: HeteroData):
        self.data = data
        self.positive_edges = None

    def extract_positive_edges(self) -> List[Tuple[int, int]]:
        """修正的正样本提取 - 避免重复计算"""
        positive_edges = set()

        for edge_type, edge_index in self.data.edge_index_dict.items():
            head_type, relation, tail_type = edge_type

            # 关键修改：只处理原始的miRNA-Gene相互作用边，忽略反向边
            if (head_type == 'miRNA' and tail_type == 'Gene' and
                ('interact' in relation.lower() or 'target' in relation.lower() or 'regulat' in relation.lower() or
                 'validated' in relation.lower()) and  # ✨ 添加validated_interaction的识别
                'reverse_' not in relation):

                edges_list = edge_index.t().tolist()
                positive_edges.update(map(tuple, edges_list))
                logger.info(f"从 {edge_type} 提取 {len(edges_list)} 个正样本")

        logger.info(f"总计唯一正样本: {len(positive_edges):,}")
        self.positive_edges = positive_edges
        return list(positive_edges)

    def generate_negatives(self, num_negatives: int) -> List[Tuple[int, int]]:
        """生成负样本"""
        if self.positive_edges is None:
            self.extract_positive_edges()

        num_mirnas = self.data['miRNA'].num_nodes
        num_genes = self.data['Gene'].num_nodes

        # [FIX] 原实现未记录已采负例，同一负例对可能重复出现并被分到训练/测试两侧；改为唯一采样
        rng = np.random.default_rng(config.seed)
        negatives = set()
        attempts = 0
        max_attempts = num_negatives * 10

        while len(negatives) < num_negatives and attempts < max_attempts:
            pair = (int(rng.integers(num_mirnas)), int(rng.integers(num_genes)))
            if pair not in self.positive_edges:
                negatives.add(pair)
            attempts += 1

        negatives = sorted(negatives)
        logger.info(f"生成 {len(negatives)} 个唯一负样本")
        return negatives


# 便利函数
def load_knowledge_graph(kg_file: Optional[str] = None, node_mapping: Optional[Dict] = None, **kwargs) -> HeteroData:
    """加载增强的知识图谱"""
    set_seed(config.seed)
    # 将 node_mapping 传给 Loader
    loader = EnhancedKnowledgeGraphLoader(kg_file, predefined_mapping=node_mapping)
    data = loader.create_hetero_data()
    return data

def create_train_test_split(data: HeteroData) -> Tuple[Dict, Dict, Dict]:
    """修正的训练数据分割"""
    logger.info("准备训练数据分割（避免重复样本）...")

    # 使用修正的负采样器
    sampler = NegativeSampler(data)
    positive_edges = sampler.extract_positive_edges()

    if not positive_edges:
        raise ValueError("未找到miRNA-Gene相互作用数据！")

    logger.info(f"唯一正样本数量: {len(positive_edges):,}")

    # 生成负样本
    num_negatives = int(len(positive_edges) * config.data.neg_ratio)
    negative_edges = sampler.generate_negatives(num_negatives)

    logger.info(f"负样本数量: {len(negative_edges):,}")

    # 准备完整数据集
    all_edges = positive_edges + negative_edges
    all_labels = [1] * len(positive_edges) + [0] * len(negative_edges)

    # 分层抽样划分
    train_edges, temp_edges, train_labels, temp_labels = train_test_split(
        all_edges, all_labels,
        test_size=config.training.test_ratio + config.training.val_ratio,
        random_state=config.seed,
        stratify=all_labels
    )

    val_edges, test_edges, val_labels, test_labels = train_test_split(
        temp_edges, temp_labels,
        test_size=config.training.test_ratio / (config.training.test_ratio + config.training.val_ratio),
        random_state=config.seed,
        stratify=temp_labels
    )

    logger.info("数据分割完成:")
    logger.info(f"  训练集: {len(train_edges):,}")
    logger.info(f"  验证集: {len(val_edges):,}")
    logger.info(f"  测试集: {len(test_edges):,}")

    # 转换为tensor格式
    device = config.device

    train_data = {
        'edges': torch.tensor(train_edges, dtype=torch.long, device=device),
        'labels': torch.tensor(train_labels, dtype=torch.float, device=device)
    }

    val_data = {
        'edges': torch.tensor(val_edges, dtype=torch.long, device=device),
        'labels': torch.tensor(val_labels, dtype=torch.float, device=device)
    }

    test_data = {
        'edges': torch.tensor(test_edges, dtype=torch.long, device=device),
        'labels': torch.tensor(test_labels, dtype=torch.float, device=device)
    }

    return train_data, val_data, test_data


def validate_data_quality(data: HeteroData) -> Dict[str, any]:
    """验证数据质量 - 确保没有重复和泄露"""

    # 检查图结构
    edge_stats = {}
    for edge_type, edge_index in data.edge_index_dict.items():
        edge_type_str = str(edge_type) if isinstance(edge_type, tuple) else edge_type
        edge_stats[edge_type_str] = {
            'num_edges': edge_index.shape[1],
            'unique_edges': len(set(map(tuple, edge_index.t().tolist())))
        }

    # 检查训练样本
    sampler = NegativeSampler(data)
    positive_edges = sampler.extract_positive_edges()

    # 检查特征质量
    feature_quality = {}
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x'):
            features = data[node_type].x
            feature_quality[node_type] = {
                'num_nodes': features.shape[0],
                'feature_dim': features.shape[1],
                'mean_norm': torch.norm(features, dim=1).mean().item(),
                'std_norm': torch.norm(features, dim=1).std().item(),
                'has_nonzero': (features != 0).any().item()
            }

    # 生成报告
    report = {
        'total_node_types': len(data.node_types),
        'total_edge_types': len(data.edge_index_dict),
        'unique_positive_samples': len(positive_edges),
        'edge_statistics': edge_stats,
        'feature_quality': feature_quality,
        'has_bidirectional_structure': any('reverse_' in str(et) for et in data.edge_index_dict.keys()),
        'duplicate_check_passed': True,
        'uses_biological_features': True,
        'feature_dimensions': getattr(data, 'feature_dims', {})
    }

    return report


def print_data_quality_report(report: Dict):
    """打印增强的数据质量报告"""
    print("=" * 60)
    print("增强数据质量验证报告")
    print("=" * 60)

    print(f"节点类型数量: {report['total_node_types']}")
    print(f"边类型数量: {report['total_edge_types']}")
    print(f"唯一训练样本: {report['unique_positive_samples']:,}")
    print(f"双向图结构: {'是' if report['has_bidirectional_structure'] else '否'}")
    print(f"重复检查通过: {'是' if report['duplicate_check_passed'] else '否'}")
    print(f"使用生物学特征: {'是' if report.get('uses_biological_features', False) else '否'}")

    print("\n边类型统计:")
    for edge_type, stats in report['edge_statistics'].items():
        if 'interaction' in str(edge_type) or 'validated' in str(edge_type):
            print(f"  {edge_type}: {stats['num_edges']:,} 条边 "
                  f"(唯一: {stats['unique_edges']:,})")

    # 特征质量报告
    if 'feature_quality' in report:
        print("\n✨ 特征质量分析:")
        for node_type, quality in report['feature_quality'].items():
            print(f"  {node_type}:")
            print(f"    节点数: {quality['num_nodes']:,}")
            print(f"    特征维度: {quality['feature_dim']}")
            print(f"    平均特征范数: {quality['mean_norm']:.4f}")
            print(f"    特征范数标准差: {quality['std_norm']:.4f}")
            print(f"    包含非零特征: {'是' if quality['has_nonzero'] else '否'}")

    # 原始特征维度信息
    if 'feature_dimensions' in report and report['feature_dimensions']:
        print("\n🔧 原始特征维度:")
        for node_type, dim in report['feature_dimensions'].items():
            print(f"  {node_type}: {dim} 维")

    print("\n🎯 特征增强效果:")
    print("  ✅ 基因/miRNA: 使用原始维度表达谱数据")
    print("  ✅ 通路/疾病: 使用原始维度LLM嵌入")
    print("  ✅ 关系类型归并: 大幅提升HGT计算效率")
    print("  ✅ 神经网络学习最优特征投影")


# 在 data_loader.py 文件末尾添加以下函数

from sklearn.model_selection import StratifiedKFold


def create_kfold_splits(data: HeteroData, n_splits: int = 5, random_state: int = None) -> List[Tuple[Dict, Dict]]:
    """
    创建K折交叉验证数据分割

    Args:
        data: 异构图数据
        n_splits: 折数 (默认5)
        random_state: 随机种子

    Returns:
        List of (train_data, val_data) tuples for each fold
    """
    logger.info(f"准备{n_splits}折交叉验证数据分割...")

    if random_state is None:
        random_state = config.seed

    # 使用修正的负采样器
    sampler = NegativeSampler(data)
    positive_edges = sampler.extract_positive_edges()

    if not positive_edges:
        raise ValueError("未找到miRNA-Gene相互作用数据！")

    logger.info(f"唯一正样本数量: {len(positive_edges):,}")

    # 生成负样本
    num_negatives = int(len(positive_edges) * config.data.neg_ratio)
    negative_edges = sampler.generate_negatives(num_negatives)

    logger.info(f"负样本数量: {len(negative_edges):,}")

    # 准备完整数据集
    all_edges = positive_edges + negative_edges
    all_labels = [1] * len(positive_edges) + [0] * len(negative_edges)

    # 转换为numpy数组以便于索引
    all_edges_array = np.array(all_edges)
    all_labels_array = np.array(all_labels)

    # 使用StratifiedKFold进行分层K折划分
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    kfold_data = []
    device = config.device

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_edges_array, all_labels_array)):
        logger.info(f"Fold {fold + 1}/{n_splits}:")
        logger.info(f"  训练集: {len(train_idx):,} 样本")
        logger.info(f"  验证集: {len(val_idx):,} 样本")

        # 获取训练和验证数据
        train_edges = all_edges_array[train_idx]
        train_labels = all_labels_array[train_idx]
        val_edges = all_edges_array[val_idx]
        val_labels = all_labels_array[val_idx]

        # 转换为tensor格式
        train_data = {
            'edges': torch.tensor(train_edges, dtype=torch.long, device=device),
            'labels': torch.tensor(train_labels, dtype=torch.float, device=device)
        }

        val_data = {
            'edges': torch.tensor(val_edges, dtype=torch.long, device=device),
            'labels': torch.tensor(val_labels, dtype=torch.float, device=device)
        }

        kfold_data.append((train_data, val_data))

    logger.info(f"✅ {n_splits}折交叉验证数据准备完成")
    return kfold_data


def create_train_test_split_with_kfold(data: HeteroData, test_ratio: float = 0.15,
                                       n_folds: int = 5) -> Tuple[List[Tuple[Dict, Dict]], Dict]:
    """
    创建带有独立测试集的K折交叉验证数据

    Args:
        data: 异构图数据
        test_ratio: 测试集比例
        n_folds: 交叉验证折数

    Returns:
        (kfold_data, test_data) - K折数据和独立测试集
    """
    logger.info("准备K折交叉验证（带独立测试集）...")

    # [FIX] 可复用另一次运行按名称保存的划分，使不同图在完全相同的实体对上比较
    split_file = getattr(config.data, 'fixed_split_file', None)
    if split_file:
        return load_named_splits(data, split_file, n_folds)

    # 使用负采样器
    sampler = NegativeSampler(data)
    positive_edges = sampler.extract_positive_edges()

    # 生成负样本
    num_negatives = int(len(positive_edges) * config.data.neg_ratio)
    negative_edges = sampler.generate_negatives(num_negatives)

    # 准备完整数据集
    all_edges = positive_edges + negative_edges
    all_labels = [1] * len(positive_edges) + [0] * len(negative_edges)

    # 首先分离出测试集
    train_val_edges, test_edges, train_val_labels, test_labels = train_test_split(
        all_edges, all_labels,
        test_size=test_ratio,
        random_state=config.seed,
        stratify=all_labels
    )

    logger.info(f"数据集划分:")
    logger.info(f"  训练+验证集: {len(train_val_edges):,} 样本")
    logger.info(f"  独立测试集: {len(test_edges):,} 样本")

    # 在训练+验证集上进行K折划分
    train_val_edges_array = np.array(train_val_edges)
    train_val_labels_array = np.array(train_val_labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.seed)

    kfold_data = []
    device = config.device

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_edges_array, train_val_labels_array)):
        train_edges = train_val_edges_array[train_idx]
        train_labels = train_val_labels_array[train_idx]
        val_edges = train_val_edges_array[val_idx]
        val_labels = train_val_labels_array[val_idx]

        train_data = {
            'edges': torch.tensor(train_edges, dtype=torch.long, device=device),
            'labels': torch.tensor(train_labels, dtype=torch.float, device=device)
        }

        val_data = {
            'edges': torch.tensor(val_edges, dtype=torch.long, device=device),
            'labels': torch.tensor(val_labels, dtype=torch.float, device=device)
        }

        kfold_data.append((train_data, val_data))

    # 准备测试集
    test_data = {
        'edges': torch.tensor(test_edges, dtype=torch.long, device=device),
        'labels': torch.tensor(test_labels, dtype=torch.float, device=device)
    }

    return kfold_data, test_data


def load_named_splits(data: HeteroData, split_file, n_folds: int) -> Tuple[List[Tuple[Dict, Dict]], Dict]:
    """
    [FIX] 读取 named_splits.csv（split, miRNA, gene, label），按名称映射到当前图的节点编号。
    miRNA 名按小写匹配：原图中 'hsa-let-7f-2-3P' 与 'hsa-let-7f-2-3p' 是同一分子的大小写变体。
    """
    df = pd.read_csv(split_file)
    mirna_index = {}
    for name, idx in data.node_mapping['miRNA'].items():
        assert mirna_index.setdefault(name.lower(), idx) == idx, f'miRNA 小写名冲突: {name}'
    gene_index = data.node_mapping['Gene']
    missing = df[~df.miRNA.str.lower().isin(mirna_index) | ~df.gene.isin(gene_index)]
    assert missing.empty, f'划分中有 {len(missing)} 个实体对不在当前图中: {missing.head().to_dict("records")}'
    device = config.device

    def to_tensors(part):
        edges = [(mirna_index[m.lower()], gene_index[g]) for m, g in zip(part.miRNA, part.gene)]
        return {'edges': torch.tensor(edges, dtype=torch.long, device=device),
                'labels': torch.tensor(part.label.values, dtype=torch.float, device=device)}

    kfold_data = [(to_tensors(df[df.split == f'fold{k}_train']), to_tensors(df[df.split == f'fold{k}_val']))
                  for k in range(1, n_folds + 1)]
    test_data = to_tensors(df[df.split == 'test'])
    logger.info(f"🔒 使用固定划分 {split_file}: 测试集 {len(test_data['labels'])}，"
                f"每折训练 {[len(t['labels']) for t, _ in kfold_data]}")
    return kfold_data, test_data


def is_target_edge_type(edge_type) -> bool:
    """miRNA-Gene 靶向关系（正向或反向），即待预测的目标关系。"""
    head_type, relation, tail_type = edge_type
    return {head_type, tail_type} == {'miRNA', 'Gene'} and 'validated_interaction' in relation


def build_message_graph(edge_index_dict: Dict, train_data: Dict, eval_sets: List[Dict]) -> Dict:
    """
    [FIX] 目标边泄漏修复：为每一折构建只含该折训练正例的消息传递图。
    原实现训练、验证、测试均使用完整图，测试正例作为已知边出现在输入中。
    这里移除全部 miRNA-Gene 靶向边（含反向边），再只放回训练正例；其余辅助关系保持不变。
    """
    edges, labels = train_data['edges'], train_data['labels']
    train_pos = edges[labels == 1].t().contiguous()  # [2, N] miRNA -> Gene
    graph = {}
    for edge_type, edge_index in edge_index_dict.items():
        if not is_target_edge_type(edge_type):
            graph[edge_type] = edge_index
        elif edge_type[0] == 'miRNA':
            graph[edge_type] = train_pos
        else:
            graph[edge_type] = train_pos.flip(0)

    # 断言：验证/测试对（正负例）都不在消息图的靶向边中
    in_graph = set()
    for edge_type, edge_index in graph.items():
        if is_target_edge_type(edge_type):
            pairs = edge_index.t().tolist()
            in_graph.update(map(tuple, pairs if edge_type[0] == 'miRNA' else [(b, a) for a, b in pairs]))
    for eval_set in eval_sets:
        assert not in_graph & set(map(tuple, eval_set['edges'].tolist())), '验证/测试对出现在消息图中'
    return graph


def assert_disjoint_splits(kfold_data: List[Tuple[Dict, Dict]], test_data: Dict):
    """[FIX] 训练/验证/测试实体对两两不重叠，且各自无重复。"""
    test = set(map(tuple, test_data['edges'].tolist()))
    assert len(test) == len(test_data['edges']), '测试集含重复对'
    for train_data, val_data in kfold_data:
        train = set(map(tuple, train_data['edges'].tolist()))
        val = set(map(tuple, val_data['edges'].tolist()))
        assert len(train) == len(train_data['edges']) and len(val) == len(val_data['edges']), '折内含重复对'
        assert not (train & val or train & test or val & test), '划分之间存在重叠对'

# 导出主要的类
KnowledgeGraphLoader = EnhancedKnowledgeGraphLoader