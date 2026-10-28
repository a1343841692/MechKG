# Main-Model Data

**English** | [简体中文](README.zh-CN.md)

This directory contains the actual inputs used by the main model, along with frozen sequences and processing records for auditing.
Historical datasets, older knowledge graphs, TCGA clinical/survival tables, and website exports are excluded.

| Path | Purpose |
|---|---|
| `processed/kg_v2_xgft_coding_nosim.csv` | Final knowledge graph with 200,224 triples; includes `head`, `relation`, `tail`, and explicit node types |
| `processed/kg_v2_xgft_coding_nosim.isolated_nodes.csv` | Preserves one isolated miRNA so that entities in the fixed splits remain available |
| `processed/kg_v2_text_embeddings.pkl` | Precomputed text embeddings for pathway and disease nodes, with 768 dimensions each |
| `processed/pair_seq_features_coding.npz` | An 883 × 3303 × 6 pair-feature table, with `mirnas` and `genes` name arrays |
| `processed/node_texts.csv` | Original node text used to compute text embeddings |
| `processed/family_assignment.csv` | Mature-miRNA seed-based family annotations for analysis |
| `processed/removed_noncoding_targets.csv`, `processed/symbol_fixes.csv` | Records of noncoding-target removal and gene-symbol corrections |
| `splits/coding_named_splits.csv` | Fixed five-fold training/validation splits and the independent test set, with `split`, `miRNA`, `gene`, and `label` columns |
| `splits/hard_test_pairs.csv` | Frozen test pairs with exactly matched endpoint frequencies, retained from the original hard-test cache |
| `basic/interaction.csv` | Original experimentally supported interactions, used by the loader to identify entity names; the final knowledge graph and fixed splits define the supervised edges |
| `expression/*.xena` | Complete TCGA gene and miRNA expression matrices, preserving all inputs to the original standardization and PCA steps |
| `sequences/sequences.pkl` | Frozen mature-miRNA and canonical 3′UTR sequences for checking sequence features |

The final graph contains 883 miRNA nodes, 3,303 gene nodes, 14,071 pathway/GO nodes, and 4,199 disease nodes.
The target relation is `validated_interaction`. Although the supplied graph contains the complete set of target pairs,
training and evaluation must call `build_message_graph` to remove all target edges and restore only the current fold's training positives.
Do not use the full graph's target edges directly as an evaluation message graph.

The six sequence features are, in order: log1p of the mutually exclusive counts of 8mer, 7mer-m8, 7mer-A1, and 6mer sites; log1p of UTR length; and a sequence-availability indicator.
They are standardized over all pairs without using labels. The standardization logic remains in `trainer.py`.
`code/sequence_features.py` preserves the original site-feature definitions unchanged.

Column-wise standardization of the expression matrices and randomized PCA follow the original implementation with seed 42.
Do not reduce upload size by retaining only expression rows for graph genes or miRNAs: this would change standardization, PCA, and therefore the inputs to the trained checkpoints.

The following source summary follows the original project records: experimentally supported interactions come from miRTarBase/miRATBase;
expression matrices are TCGA pancan Xena data; auxiliary graph relations include GO, MSigDB pathways, CTD disease associations, and STRING PPI;
text vectors are the existing precomputed PubMedBERT embeddings; and mature-miRNA and canonical 3′UTR sequences are the project's frozen miRBase/Ensembl sequences.
These external datasets were not downloaded again, updated, or independently reverified during packaging.
Exact original project-relative paths are listed in the root `PROVENANCE.json`.
