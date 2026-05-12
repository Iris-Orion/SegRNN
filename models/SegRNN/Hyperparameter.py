import os

# ---------------------------------------------------------------------------
# Core hyperparameters
# ---------------------------------------------------------------------------

H = 400   # history / input window
L = 200   # forecast / prediction window

batch_size = 128
epochs     = 50
lr         = 0.0005
stride     = 20

use_early_stopping      = False
early_stopping_patience = 5


class BaseConfigs:
    """Shared hyperparameters for all models. Subclassed by model-specific configs in modelzoo/."""
    def __init__(self):
        self.L          = H        # lookback window length
        self.H          = L        # forecast horizon
        self.enc_in     = 1
        self.num_layer  = 512
        self.dropout    = 0.1
        self.channel_id = False
        self.revin      = True
        # RNN-specific (None for non-RNN models)
        self.rnn_type: str | None = None
        self.dec_way:  str | None = None
        # training (overridden by args)
        self.seed           = 2048
        self.loss_name      = "l1"
        self.optimizer_name = "adamw"


# backward-compat alias used by evaluate.py
Configs = BaseConfigs


def parse_args():
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Train SegRNN")
    parser.add_argument("--model",         default="segrnn",  choices=["segrnn", "segmamba"], help="Model architecture")
    parser.add_argument("--dataset",       default="clean",   choices=["clean", "matlab905"], help="Training dataset")
    parser.add_argument("--H",             type=int,          default=H,        help="Lookback window length")
    parser.add_argument("--L",             type=int,          default=L,        help="Forecast horizon")
    parser.add_argument("--loss",          default="l1",      choices=["l1", "mse", "huber"])
    parser.add_argument("--optimizer",     default="adamw",   choices=["adam", "adamw", "sgd"])
    parser.add_argument("--lr",            type=float,        default=None,     help="Override learning rate")
    parser.add_argument("--warmup",        action="store_true",  help="Enable linear warmup before cosine decay")
    parser.add_argument("--warmup-ratio",  type=float,           default=0.1,  help="Warmup fraction of total epochs")
    parser.add_argument("--early-stopping",action="store_true", default=use_early_stopping)
    parser.add_argument("--grad-clip",     type=float,        default=None,     help="Gradient clip norm")
    parser.add_argument("--stride",        type=int,          default=stride,   help="Sliding window stride")
    parser.add_argument("--seed",          type=int,          default=2048)
    parser.add_argument("--cuda",          default="",        help="CUDA device index, e.g. 0 or 1")
    parser.add_argument("--swanlog",       action="store_true")
    parser.add_argument("--swanlab-mode",  default="cloud",
                        choices=["disabled", "cloud", "local", "offline"])
    args = parser.parse_args()

    if args.cuda:
        args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args
