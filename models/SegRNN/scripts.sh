#!/usr/bin/env bash
# Training scripts for SegRNN framework
# Run from anywhere; all paths are relative to this script's directory.

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# SegRNN
# ---------------------------------------------------------------------------

# SegRNN — clean dataset (cuda:0)
python train.py --model segrnn --dataset clean --cuda 0

# SegRNN — matlab905 dataset (cuda:0)
# python train.py --model segrnn --dataset matlab905 --cuda 0

# ---------------------------------------------------------------------------
# SegMamba
# ---------------------------------------------------------------------------

# SegMamba — clean dataset (cuda:0)
# python train.py --model segmamba --dataset clean --cuda 0
# python train.py --model segmamba --dataset clean --cuda 1 --warmup --grad-clip 1.0 --swanlog

# SegMamba — matlab905 dataset (cuda:0)
# python train.py --model segmamba --dataset matlab905 --swanlog --warmup --cuda 0 
# python train.py --model segmamba --dataset matlab905 --swanlog --warmup --cuda 1

# python train.py --model segmamba --dataset clean --swanlog --warmup --cuda 1

# python train.py --model segrnn --dataset clean --swanlog --warmup --cuda 1
# python train.py --model segrnn --dataset matlab905 --swanlog --warmup --cuda 1 --lr 1e-4
# ---------------------------------------------------------------------------
# Common optional flags (append to any command above):
#   --cuda 1                  use GPU 1
#   --swanlog                 enable SwanLab cloud logging
#   --swanlab-mode offline    log locally without uploading
#   --early-stopping          stop when val loss stops improving
#   --warmup                  linear warmup + cosine decay LR schedule
#   --warmup-ratio 0.1        warmup fraction of total steps (default 0.1)
#   --lr 0.001                override default learning rate
#   --grad-clip 1.0           gradient clip norm
#   --H 400 --L 200           lookback / forecast window
#   --stride 20               sliding window stride
#   --loss l1                 loss function: l1 | mse | huber
#   --optimizer adamw         optimizer: adam | adamw | sgd
# ---------------------------------------------------------------------------
