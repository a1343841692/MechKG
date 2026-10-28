# Arm-Annotation Flags

**English** | [简体中文](README.zh-CN.md)

This release contains **137 unique validated miRNA–gene labels flagged for possible 5p/3p arm-annotation ambiguity**, covering **78 miRNAs**.
The list was frozen by the original audit on October 3, 2026 and packaged on October 28, 2026. It is an independent label-quality audit, not a set of errors detected by MechKG.

## Start here

- `arm_annotation_flags.csv`: the 137 flagged pairs, with English-coded categories, original identifiers, fixed-split membership, canonical-site classes and CLASH indicators.
- `field_dictionary.csv`: definitions of every column. The unique key is `(arm, gene)`.
- `evidence/`: frozen sequence-feature rows for the flagged pairs and opposite arms, relevant WT-detected CLASH rows, and the full detected-miRNA index.
- `audit/`: unmodified full audit outputs: 9,353 role-specific records and per-arm-pair counts. This table is not a list of 9,353 flagged labels.
- `sources/`: the original audit script and frozen miRBase mature-to-precursor mapping. The original script uses the original project layout; it is included as a reference, not as a standalone reproduction entry point.
- `diagnostics/`: the complete existing follow-up score diagnostic, including its protocol, input provenance and negative matched-comparison result. Chinese original reports are preserved with `.zh-CN.md` names.
- `verify_flags.py`: a standard-library check of this package, original flag extraction, feature/CLASH agreement and file checksums. Run `python verify_flags.py` inside this directory.

## Flag definition and counts

A flagged label satisfies all four conditions: (1) it belongs to the minor arm, defined by fewer validated targets in the original audit; (2) the target gene is shared with the opposite arm; (3) there is no canonical site for the flagged arm; and (4) the opposite arm has a canonical site.
Canonical means **8mer, 7mer-m8 or 7mer-A1** in the canonical 3′UTR. A 6mer is not canonical under this audit's definition.

| Item | Count |
|---|---:|
| Unique flagged labels | 137 |
| Flagged miRNAs | 78 |
| Non-test / held-out validation labels | 110 |
| Independent test labels | 27 |
| Flagged-arm site class `none` | 104 |
| Flagged-arm site class `6mer` | 33 |
| Both arms observed somewhere in WT-detected CLASH (`clash_measurable`) | 81 |
| CLASH support for the flagged arm/gene (`clash_own`) | 1 |
| CLASH support for the opposite arm/gene (`clash_other`) | 46 |

The 137 labels are approximately 1.4% of the 10,112 unique validated positives. Both training and test labels were available to this label-source audit; it is not a held-out model-performance evaluation.
CLASH was **not part of the flag-selection rule**. Not every flagged label has CLASH support. Its indicators are additional evidence, and absence of detection does not prove absence of binding. The original CLASH rule uses WT_Detected_8 >= 1 and includes any annotated target region, not just the 3′UTR.

## Interpretation and later diagnostic

These are candidates for literature/database review, not confirmed annotation mistakes. Do not relabel or delete them automatically.
Potential explanations include publications that did not resolve the mature arm and precursor overexpression; the audit does not establish which explanation applies to an individual pair.

The later matched score analysis **did not establish that MechKG gives these labels lower out-of-fold scores**. In primary comparison B, P(control > flagged) was 0.478 (95% miRNA-cluster bootstrap interval 0.418–0.540) for total logits and 0.455 (0.410–0.507) for summed graph-evidence contributions. Both intervals include 0.5. These flags should therefore remain a model-independent data-quality annotation.

## Provenance and scope

Original outputs: `消融实验_fixed/arm_annotation_20261003/label_classes.csv` and `by_hairpin.csv`.
Fixed splits: `消融实验_fixed/splits/coding_named_splits.csv`. Sequence inputs: `data_kg_v2/pair_seq_features_coding.npz`.
Source CLASH table: `binding_function_20260927/source/CLASH_human_HEK293T_Expanded.csv`; only relevant detected rows are included here.
The full original CLASH CSV is not included. Exact source paths, sizes and SHA-256 values are in `PROVENANCE.json`.
Core data and checkpoints remain in the existing MechKG release archives; no additional model weights are included in this package.
No model training or inference was run during packaging. Original inputs and outputs were preserved. Third-party data permissions are unchanged.
