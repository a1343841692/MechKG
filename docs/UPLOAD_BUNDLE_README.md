# MechKG: Main-Model Code, Data, and Checkpoints

**English** | [简体中文](UPLOAD_BUNDLE_README.zh-CN.md)

Original bundle prepared on October 26, 2026; bilingual READMEs added on October 28, 2026.
The final main model is **`mk_ns_nodrop`**, supplied as **five fold-specific checkpoints**.

This standalone bundle contains the inputs and code needed to reproduce the main-model workflow.
Historical models, ablation and baseline weights, website files, caches, remote-connection settings, and credentials are excluded.
The original workspace code, source data, and full training checkpoints were left unchanged.

## Downloadable archives

The release provides three ZIP archives:

| Archive | Contents |
|---|---|
| `MechKG_code.zip` | `code/`, existing main-model outputs in `results/`, documentation, model configuration, and file manifests |
| `MechKG_data.zip` | All required inputs in `data/`, including full expression matrices, the knowledge graph, splits, and text/sequence features |
| `MechKG_models_5fold.zip` | `models/`: exactly five `.pth` files, their configuration, and provenance records |

Extract all three archives into the **same empty directory**. No other directories from the original project are required.
For services that accept multiple attachments, the three ZIPs can be uploaded separately with this README.

```text
README.md                  # Default English instructions
README.zh-CN.md            # Simplified Chinese instructions
MODEL_CARD.md
PROVENANCE.json             # Original project-relative source paths
FILE_MANIFEST.json          # File sizes and SHA-256 checksums
SHA256SUMS.txt
code/                      # Core code and standalone entry points
data/                      # See data/README.md
models/                    # fold_1_model.pth through fold_5_model.pth
results/                   # Previously generated main-model outputs
```

## Environment and commands

Open a terminal in the extracted directory. Package versions are recorded in `code/environment_record.json`.
The local release checks used Python 3.13. Use a matching PyTorch environment, or prepare a PyTorch build appropriate for your hardware before installing the dependencies in `code/requirements.txt`.
Loading expression matrices and computing PCA require substantial memory and CPU time. All original matrix rows and columns are included to preserve the model inputs.

```bash
python -m pip install -r code/requirements.txt

# Check files, checkpoint structures, splits, and stored predictions on CPU.
# This does not train the model or run inference.
python code/check_release.py

# Check SHA-256 checksums and the checkpoint count using only the standard library.
python code/check_release.py --hash-only

# Evaluate each released fold on the standard and frozen hard test sets.
python code/predict_main.py --protocol both --device cuda

# Evaluate one fold and export channel contributions to a separate directory.
python code/predict_main.py --fold 1 --protocol standard --explain --output runs/explain_fold1

# Retrain from scratch into runs/retrained_main/ without replacing released weights.
python code/train_main.py --device cuda
```

Replace `--device cuda` with `--device cpu` if CUDA is unavailable. Full training and inference are slower on CPU.
The entry points resolve paths within the bundle and set `MPLBACKEND=Agg`. You can set `PYTHONIOENCODING=utf-8` before launching Python.
Use `train_main.py` and `predict_main.py` as the entry points; `main.py` and `prediction.py` remain as modules from the original code.
Predictions and primary metrics are reported separately for the five folds, rather than as an ensemble.

## Data and reporting conventions

- `models/main_model_config.json` specifies the final configuration: HGT encoder, linear evidence readout, collaborative channel disabled, no channel dropout, no entity residual, and no global branch.
- `kg_v2_xgft_coding_nosim.csv` excludes the incorrect sequence-similarity edges. Its isolated-node sidecar is retained.
- Expression inputs use 64 PCA components plus an availability indicator, with missing expression values filled with zero. Pairwise sequence inputs have six features.
- Supervised positive edges are masked within each training batch. Dynamic negatives use a 1:3 positive-to-negative ratio. At evaluation, target edges in each fold's message graph come only from that fold's training positives, excluding validation and test targets.
- The five folds share one independent test set and are not five fully independent experiments. Primary results use individual-model means and sample standard deviations.
- The original `results/main_model/results/kfold/kfold_results.json` uses a population standard deviation in its `std` fields. Sample standard deviations used for reporting are in `results/main_model_summary.json`.
- The original metric named `AUPR` is scikit-learn's average precision, abbreviated as AP.

## Checkpoints and verification scope

`models/*.pth` are portable inference checkpoints. Every `model_state_dict` tensor is exactly equal to its original counterpart.
Optimizer states were removed, and NumPy metadata values were converted to native Python types, allowing loading with `torch.load(..., weights_only=True)`.
Original MD5 checksums, training epochs, metrics, and conversion records are in `models/model_manifest.json`.
Use the original full checkpoints if you need to resume an interrupted training run; the released entry point retrains from scratch.

Core model implementations such as `code/models.py`, `mechkg.py`, and `data_loader.py` were preserved.
Paths were adapted to the standalone layout. The trainer calls `sampling.py` for endpoint-matched sampling; that function was extracted unchanged from its original module.

Release checks covered files, structures, splits, and previously saved predictions on CPU. No training or model inference was performed.
The new training and prediction entry points have not undergone full training or forward-pass verification.
Cross-platform numerical reproducibility still needs confirmation, especially for randomized PCA and linear algebra library versions.
`PROVENANCE.json` records source files; `VERIFICATION.json` records the original checks, and `README_UPDATE_VERIFICATION.json` records the bilingual documentation checks.

Third-party data and precomputed embeddings retain the provenance and permissions recorded in the original project.
The bundle does not change those permissions or assign a new open-source license.
