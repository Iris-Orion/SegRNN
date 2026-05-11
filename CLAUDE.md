# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Time-series forecasting framework comparing RNN-based and transformer-based architectures on FFT-decomposed signal data. Input data is loaded from `fft_decomposition_cleaned.mat` (must contain a `clean_data` variable).

## Running Models

Each model lives in `models/<ModelName>/` and is self-contained. From within a model directory:

```bash
# Train
python train.py

# Evaluate on fft_decomposition_cleaned (train/val/test splits)
python evaluate.py

# Full-signal inference on matlab905(1).mat
python evaluate.py --mode predict
```

### Key Environment Variable Overrides for `train.py`

| Variable | Effect |
|---|---|
| `EXP_H` / `EXP_L` | Override lookback window / forecast horizon |
| `LR_OVERRIDE` | Override learning rate |
| `OPTIMIZER_NAME` | `adam` (default), `adamw`, `sgd` |
| `LOSS_NAME` | `huber` (default), `mse`, `mae` |
| `USE_SCHEDULER=true` | Enable LR scheduler |
| `USE_EARLY_STOPPING=true` | Enable early stopping |
| `GRAD_CLIP_NORM` | Gradient clipping norm |
| `SEQUENCE_STRIDE` | Sliding window stride (default: 20) |
| `USE_SWANLOG=true` | Enable SwanLab experiment logging |

### Key Environment Variable Overrides for `evaluate.py --mode predict`

| Variable | Effect |
|---|---|
| `INPUT_MAT` | Path to input `.mat` file (default: `data/matlab905(1).mat`) |
| `INPUT_VAR` | Variable name inside the `.mat` file (default: `data`) |
| `NORM_SOURCE` | `self` (default) or `train` (use clean_data min/max) |
| `PRED_STRIDE` | Sliding window stride (default: L) |

## Architecture

**Data pipeline** (shared across models via `split_data.py`):
1. Load `.mat` → flatten to 1D → fill NaN with mean → min-max normalize to [0,1]
2. Split 60/20/20 (train/val/test)
3. Sliding window sequences: lookback H=400, horizon L=200, stride=20

**Models:**
- `SegRNN` — Segment-based RNN (RNN/GRU/LSTM), RevIN normalization, RMF or PMF decoding
- `DsFormer` — Dual-stream Transformer with TVA attention blocks
- `PatchTST` — Patch-tokenized Transformer
- `TiDE` — MLP encoder-decoder with residual blocks
- `Seg-Mamba` — State-space model (Mamba) with segment decomposition

**Each model directory contains:**
- `Hyperparameter.py` — All config (H, L, batch_size, epochs, lr, model-specific params)
- `[ModelName].py` — Model architecture
- `train.py` — Training loop with early stopping, metric logging
- `evaluate.py` — Inference + metrics (MAE, MSE, RMSE, R²) + visualization; also handles matlab905 prediction via `--mode predict`
- `datasets.py` — `CleanDataset` and `Matlab905Dataset` dataset classes + `make_loaders` factory
- `split_data.py` — Signal splitting, sliding window sequences, metrics

## Directory Structure (SegRNN)

```
models/SegRNN/
├── data/                          # Input datasets
│   ├── fft_decomposition_cleaned.mat
│   └── matlab905(1).mat
├── outputs/                       # All evaluation outputs
│   └── <ModelName>/
│       └── <dataset_name>/
│           └── <timestamp>/       # One folder per run
│               ├── metrics.txt
│               ├── *.svg
│               └── *.mat
├── tests/                         # pytest test suite
│   ├── conftest.py
│   └── test_datasets.py
├── eda/                           # Exploratory data analysis
│   └── split_visualization.ipynb
├── best_model.pth                 # Best checkpoint (written by train.py)
└── loss_curve.svg                 # Training loss curve
```

## Output Artifacts

`train.py` writes to the model directory:
- `best_model.pth` — Best checkpoint by validation loss
- `loss_curve.svg` — Train/val loss curves
- `best_params_and_metrics.txt` — Training summary

`evaluate.py` writes to `outputs/<ModelName>/<dataset>/<timestamp>/`:
- `metrics.txt` — MAE, MSE, RMSE, R² per split
- `train_compare.svg`, `val_compare.svg`, `test_compare.svg`, `all_compare.svg` — Prediction plots
- `test_error.svg` — Error plot
- `metrics_comparison.svg` — Bar chart across splits
- `model_outputs.mat` — Predictions and ground truth (evaluate mode)
- `compare.svg`, `error.svg`, `predictions.mat` — (predict mode)

## Dependencies

```bash
pip install torch numpy scipy scikit-learn matplotlib h5py tqdm pytest
# Optional (Seg-Mamba):
pip install einops mamba_ssm
# Optional (experiment logging):
pip install swanlab
```

Device is selected automatically (`cuda` if available). Random seed is fixed at 2048.

## 报错纠正思路
遇到pip安装问题时切换为 https://pypi.tuna.tsinghua.edu.cn/simple
保持torch版本为2.7.0,不要随便升级或者降级, 安装mamba需要保持torch 2.7.0的兼容性
