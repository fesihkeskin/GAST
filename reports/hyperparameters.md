# GAST Hyperparameters & CLI Reference

This document is the **source of truth** for the current command-line arguments and architecture knobs used by:
- `main.py` (train/test entry point + optional t-SNE)
- `src/training/train.py` (training loop, splits, adaptive loss, checkpointing)
- `src/training/test.py` (evaluation + metrics/artifacts)
- `src/models/model_architecture.py` (GAST definition)
- `src/training/gast_hyperparameter_tuning.py` (Optuna coarse→fine search)
- `src/training/generate_optuna_reports.py` (Optuna visualization/reporting)

> Note: `main.py` exposes `--model_type`, but the current `train.py` / `test.py` always instantiate `GAST`. So `--model_type` is currently informational unless you extend the dispatcher.

---

## 1) Dataset, Mode, and Derived Paths

- **`dataset`** (str, choices from `DATASET_PATHS`)
  - Selects which HSI dataset to use (e.g., `Indian_Pines`, `Pavia_University`, `Houston13`, etc.).

- **`mode`** (`train` | `test`)
  - `train`: runs training and saves splits + checkpoint + training history.
  - `test`: loads saved splits + checkpoint and writes evaluation artifacts.

**Derived (not CLI)**
- `cube_path`, `gt_path` are resolved from `DATASET_PATHS`.
- `raw_gt` is loaded as a NumPy array and used for stratified splitting and metrics.

---

## 2) Data Splitting & Patch Extraction

- **`train_ratio`** (float, default `0.05`)
  - Fraction of labeled pixels used for training.

- **`val_ratio`** (float, default `0.05`)
  - Fraction of labeled pixels used for validation (relative to remaining after train split, as implemented in the splitter).

- **`patch_size`** (int, default `11`)
  - Spatial patch size $k \times k$ extracted around each labeled pixel.
  - Also controls the GAST graph size: number of nodes is $k^2$.

- **`stride`** (int, default `5`)
  - Stride used by `HyperspectralDataset` when extracting patches.
  - **Important nuance (current behavior):**
    - Training uses `stride=args.stride` for train/val/test splits.
    - Testing (`src/training/test.py`) builds:
      - `train_ds` with `stride=args.stride` (to reproduce train mean/std normalization),
      - `test_ds` with `stride=1` (dense evaluation over the saved test indices).

---

## 3) Training Control & Outputs

- **`batch_size`** (int, default `128`)
  - Batch size for DataLoader (train/val/test).

- **`epochs`** (int, default `200`)
  - Max epochs.

- **`early_stop`** (int, default `20`)
  - Early-stopping patience: training stops if validation accuracy does not improve for `early_stop` epochs.

- **`output_dir`** (str, default `./models/checkpoints`)
  - All artifacts are written under this directory, including:
    - `splits/` (`train_idx_seed_*.npy`, `val_idx_seed_*.npy`, `test_idx_seed_*.npy`)
    - `splits/class_distribution_seed_*.txt`
    - `gast_best_<dataset>.pth`
    - `train_history_<dataset>_<model_name>.json`
    - `test_results/` (when running test)

- **`num_workers`** (int, default `4`)
  - DataLoader workers.

- **`seed`** (int, default `242`)
  - Controls split reproducibility and training randomness.

---

## 4) Optimization Hyperparameters

- **`lr`** (float, default `5e-4`)
  - AdamW learning rate.

- **`weight_decay`** (float, default `5e-4`)
  - AdamW weight decay.

- **`dropout`** (float, default `0.2`)
  - Applied in multiple places:
    - spectral MLP
    - transformer encoder layer(s)
    - GATv2Conv dropout
    - classifier MLP

**Fixed training behavior (not CLI)**
- LR scheduler: `ReduceLROnPlateau(mode="max", patience=10, factor=0.3)` based on validation accuracy.
- Mixed precision: uses `torch.amp.autocast` + `GradScaler` on CUDA.

---

## 5) Loss Function Controls (Adaptive Imbalance Handling)

Training uses inverse-frequency class weights computed from labels (log-smoothed), and chooses between Weighted CrossEntropy and Focal Loss.

- **`force_wce`** (flag)
  - Forces weighted CrossEntropyLoss.

- **`force_focal`** (flag)
  - Forces FocalLoss.

- **`imbalance_threshold`** (float, default `5`)
  - Used only if neither `force_wce` nor `force_focal` is set.
  - Compute imbalance ratio $\rho$ (from training labels):
    - If $\rho > imbalance_threshold` → FocalLoss
    - Else → weighted CrossEntropyLoss

**Fixed loss details (current code)**
- Weighted CrossEntropy: `label_smoothing=0.05`, `ignore_index=-1`
- FocalLoss: `gamma=2.0`, `ignore_index=-1`

---

## 6) GAST Architecture Hyperparameters

The current GAST is **Spectral MLP + Transformer encoder** (spectral branch) and **GATv2** (spatial branch), fused by a configurable fusion mode.

### 6.1 Spectral branch
- **`embed_dim`** (int, default `64`)
  - Maps to `spec_dim`: spectral embedding dimension.
  - Spectral pipeline:
    - per-node MLP → add learnable positional embedding → Transformer encoder.

- **`transformer_heads`** (int, default `8`)
  - Number of attention heads in the Transformer encoder layer.

- **`transformer_layers`** (int, default `2` in `main.py`)
  - Number of TransformerEncoder layers.
  - (Model default is `1`, but `main.py` explicitly passes the CLI value, so CLI wins.)

### 6.2 Spatial branch (graph attention)
- **`gat_hidden_dim`** (int, default `64`)
  - Maps to `spat_dim`: spatial feature dimension.

- **`gat_heads`** (int, default `8`)
  - Number of heads for each GATv2 layer.

- **`gat_depth`** (int, default `4`)
  - Number of stacked GATv2Conv layers.

### 6.3 Patch graph definition (fixed)
- Nodes: $k^2$ nodes for a `patch_size=k` patch.
- Edges: 8-neighborhood connectivity in the $k \times k$ grid (including diagonals).
- Edge index is cached per `patch_size`.

### 6.4 Ablations / feature fusion
- **`disable_spectral`** (flag)
  - Disables spectral branch.

- **`disable_spatial`** (flag)
  - Disables spatial branch.

- **`fusion_mode`** (`gate` | `concat` | `spatial_only` | `spectral_only`, default `gate`)
  - `gate`:
    - Requires both branches enabled.
    - Uses **vector gating**: a small MLP outputs a per-dimension gate in `[0,1]`.
    - Fuses as: `gate * spatial + (1 - gate) * proj(spectral)`.
  - `concat`:
    - Requires both branches enabled.
    - Concatenates `spatial` and `proj(spectral)` at the center node and classifies.
  - `spatial_only`:
    - Uses only spatial branch output.
  - `spectral_only`:
    - Uses only projected spectral output.

**Compatibility note**
- If you set `fusion_mode=gate` or `fusion_mode=concat` while disabling one branch, the model raises a `ValueError`.

---

## 7) Checkpointing, Testing, and Artifacts

### 7.1 Training checkpoint
- Best checkpoint is saved as:
  - `output_dir/gast_best_<dataset>.pth`

### 7.2 Testing arguments (main entry point)
- **`checkpoint`** (str, required for `--mode test`)
  - Path to `.pth` checkpoint.

Testing also expects:
- `output_dir/splits/train_idx_seed_<seed>.npy`
- `output_dir/splits/test_idx_seed_<seed>.npy`
(and optionally val indices)

### 7.3 Test outputs
Written under:
- `output_dir/test_results/`
Includes:
- `metrics_seed_<seed>.json`
- confusion matrix images/text/npy
- per-class accuracy plots/npy
- `pred_map_seed_<seed>.npy`
- GT vs prediction visualization

---

## 8) Optional t-SNE

- **`run_tsne`** (flag)
  - If set, runs t-SNE after training or testing.

- **`max_samples`** (int, default `3000`)
  - Upper bound on samples used for t-SNE.

If running after training and `checkpoint` is not provided, `main.py` uses:
- `output_dir/gast_best_<dataset>.pth`

---

## 9) Optuna Hyperparameter Tuning (Two-Phase)

Script: `src/training/gast_hyperparameter_tuning.py`

### 9.1 Fixed tuning constants
- `COARSE_TRIALS = 50`
- `FINE_TRIALS   = 20`
- `MAX_EPOCHS    = 500`
- `EARLY_STOP    = 50`
- `FIXED_SEED    = 242`
- `NUM_WORKERS   = 4`
- `TRAIN_RATIO   = 0.05`
- `VAL_RATIO     = 0.05`
- Pruner: `HyperbandPruner(min_resource=1, max_resource=MAX_EPOCHS, reduction_factor=3)`
- Parallel jobs: `study.optimize(..., n_jobs=4)`

### 9.2 Tuned parameter space (current)
Coarse phase samples:
- `epochs`, `batch_size`, `patch_size`, `stride`
- `lr`, `weight_decay`, `dropout`
- `embed_dim`, `gat_hidden_dim`, `gat_heads`, `gat_depth`
- `transformer_heads`, `transformer_layers`

Fine phase narrows around coarse best for:
- `lr`, `weight_decay`, `epochs`, `stride`, `transformer_layers`
and keeps the rest fixed.

### 9.3 Tuning outputs
- Timestamped run outputs:
  - `reports/results/<TS>/`
  - includes `study_<dataset>_<phase>.pkl` and plots, plus best CLI `.sh` scripts.
- Persistent Optuna DB storage:
  - `reports/optuna_db/study_<dataset>_<phase>.db`
- Best CLI scripts point final outputs to:
  - `models/final/gast/<dataset>/`
and expect the best checkpoint at:
- `models/final/gast/<dataset>/gast_best_<dataset>.pth`

---

## 10) Optuna Report Generation

Script: `src/training/generate_optuna_reports.py`

### 10.1 CLI arguments
- **`pkl_path`** (str)
  - Load a `.pkl` study and generate figures.

- **`dataset`** (str)
  - DB mode: dataset name used to find `study_<dataset>_<phase>.db`.

- **`phases`** (list, default `["coarse", "fine"]`)
  - Which study phases to process in DB mode.

- **`db_dir`** (str, default `reports/optuna_db`)
  - Directory containing sqlite DB files.

- **`out_dir`** (str, default `reports/optuna_figures`)
  - Output directory for figures/reports.

- **`top_n`** (int, default `10`)
  - How many top trials to include in summary CSV.

- **`zip_name`** (str, default `optuna_reports.zip`)
  - Zips the generated report folder (DB mode).

### 10.2 Outputs
- Optimization history, parameter importances (if available), parallel coordinate, slice plots
- `*_all_trials.csv`, `*_top_<N>_trials.csv`
- `*_state_summary.json`, `*_robustness_stats.json`
- Optional ZIP of report directory