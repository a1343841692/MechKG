# MechKG

**English** | [简体中文](README.zh-CN.md)

miRNA–mRNA target prediction and analysis of mechanism-channel contributions.
This repository contains the code, data documentation, and existing results for the final main model, **`mk_ns_nodrop`**.
The released model files contain **only its five fold-specific checkpoints**.

## Download the data and checkpoints

The complete input data, five checkpoints, and standalone code bundle are available in
[Release v1.0.1](https://github.com/a1343841692/MechKG/releases/tag/v1.0.1).
Both this repository and its release assets are private. Sign in with a GitHub account that has access to the repository.

| Release asset | Contents | Approximate size |
|---|---|---:|
| `MechKG_code.zip` | Standalone code, existing main-model results, bilingual READMEs, and verification records | 14 MB |
| `MechKG_data.zip` | Required inputs, complete expression matrices, knowledge graph, fixed splits, and features | 440 MB |
| `MechKG_models_5fold.zip` | The five portable checkpoints of the main model | 49 MB |
| `ARCHIVE_SHA256SUMS.txt` | SHA-256 checksums for the three ZIP archives | — |

To use the code from this repository:

```bash
gh repo clone a1343841692/MechKG
cd MechKG
gh release download v1.0.1 --repo a1343841692/MechKG --pattern 'MechKG_data.zip' --pattern 'MechKG_models_5fold.zip' --dir downloads
python -m zipfile -e downloads/MechKG_data.zip .
python -m zipfile -e downloads/MechKG_models_5fold.zip .
python -m pip install -r code/requirements.txt
python code/check_release.py
```

Alternatively, download all three ZIP archives and extract them into the same empty directory.
See the [standalone bundle instructions](docs/UPLOAD_BUNDLE_README.md).
Keep this approach separate from the Git clone: the repository and standalone code bundle have their own READMEs and file manifests.
Extracting the code ZIP over a Git clone would replace the repository-specific files.

## Run the model

```bash
# Evaluate the existing checkpoints separately for each fold.
# Outputs are written to runs/predictions/.
python code/predict_main.py --protocol both --device cuda

# Export channel contributions for one fold.
python code/predict_main.py --fold 1 --explain --output runs/explain_fold1

# Retrain using the final configuration without overwriting released checkpoints.
python code/train_main.py --device cuda
```

Use `--device cpu` if CUDA is unavailable. Recorded package versions are in `code/environment_record.json`.
Loading the expression matrices and computing PCA require substantial CPU time and memory; full training also takes time.

## Model and evaluation

- Final knowledge graph: `kg_v2_xgft_coding_nosim.csv`, with its isolated-node sidecar and no sequence-similarity edges.
- HGT encoder with a linear evidence readout. The collaborative channel, channel dropout, entity residual, and global branch are disabled.
- Expression inputs use 64 PCA components plus an availability indicator. Pairwise sequence inputs have six features. Dynamic training negatives use a 1:3 positive-to-negative ratio.
- Fixed five-fold splits share one independent test set. Each fold's message graph contains only that fold's training positives as target edges.
- Primary results report the mean and sample standard deviation of individual fold models, rather than an ensemble.
- Every released model-state tensor is identical to its original checkpoint. Optimizer states were removed from the portable files.

| Evaluation protocol | Individual-model AUC, mean ± sample SD | Individual-model AP, mean ± sample SD |
|---|---:|---:|
| Standard independent test | 0.9201 ± 0.0026 | 0.9244 ± 0.0024 |
| Endpoint-degree-matched hard test | 0.6969 ± 0.0091 | 0.6934 ± 0.0152 |

These values come from previously saved fold predictions. No retraining or model forward pass was performed while preparing or uploading the release.
CPU checks covered file integrity, strict loading of checkpoint structures, split disjointness, and recomputation of metrics from saved predictions.
The new training and prediction entry points have not undergone full training or forward-pass verification; numerical reproducibility across platforms still requires confirmation in the execution environment.

Further information: [model card](MODEL_CARD.md), [data README](data/README.md),
[final configuration](models/main_model_config.json), [checkpoint provenance](models/model_manifest.json),
[source-file provenance](PROVENANCE.json), and [original verification record](VERIFICATION.json).

Third-party data retain the provenance and permissions recorded in the original project. This repository does not assign a new open-source license.
