import os

# ---------------------------------------------------------------------------
# Core hyperparameters
# ---------------------------------------------------------------------------

H = 400   # history / input window
L = 200   # forecast / prediction window

batch_size = 128
epochs     = 100
lr         = 0.0005
stride     = 20

use_early_stopping      = False
early_stopping_patience = 5


class Configs:
    def __init__(self):
        self.L          = H        # model input/history length
        self.H          = L        # model forecast/prediction length
        self.enc_in     = 1        # 输入维度
        self.num_layer  = 512
        self.dropout    = 0.1      # dropout 概率
        self.rnn_type   = 'rnn'    # RNN 类型
        self.dec_way    = 'rmf'    # 解码方式
        self.seg_len    = 10       # 片段长度
        self.channel_id = False    # 是否启用 channel id
        self.revin      = True     # 是否使用 RevIN
        # 训练行为（可被 train.py 的 args 覆盖）
        self.seed           = 2048
        self.loss_name      = "l1"
        self.optimizer_name = "adamw"


def parse_args():
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Train SegRNN")
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
