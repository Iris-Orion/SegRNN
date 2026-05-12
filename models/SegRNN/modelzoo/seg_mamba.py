# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class Model(nn.Module):
    def __init__(
        self,
        H,
        L,
        enc_in,
        num_layer,
        dropout,
        seg_len,
        channel_id=False,
        revin=True,
        D_STATE=32,
        DCONV=4,
        E_FACT=2,
    ):
        super().__init__()
        if H % seg_len != 0 or L % seg_len != 0:
            raise ValueError(f"H({H}) and L({L}) must be divisible by seg_len({seg_len}).")

        # L: input history length, H: forecast horizon.
        self.H = H
        self.L = L
        self.enc_in = enc_in
        self.num_layer = num_layer
        self.dropout = dropout
        self.seg_len = seg_len
        self.channel_id = channel_id
        self.revin = revin

        # Segment counts for input/history and output/forecast.
        self.seg_num_x = L // seg_len
        self.seg_num_y = H // seg_len

        self.value_embedding = nn.Sequential(
            nn.Linear(seg_len, num_layer),
            nn.GELU(),
        )
        self.mamba1 = Mamba(d_model=num_layer, d_state=D_STATE, d_conv=DCONV, expand=E_FACT)
        self.mamba2 = Mamba(d_model=num_layer, d_state=D_STATE, d_conv=DCONV, expand=E_FACT)
        self.norm = nn.LayerNorm(num_layer)

        self.pos_emb_x = nn.Parameter(torch.randn(1, self.seg_num_x, num_layer) * 0.02)
        self.to_y = nn.Linear(self.seg_num_x, self.seg_num_y)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_layer, seg_len),
        )

        if revin:
            self.revin_layer = RevIN(enc_in, affine=False, subtract_last=False)

    def forward(self, x):
        # x: [B, L, C]
        bsz = x.size(0)

        if self.revin:
            x = self.revin_layer(x, "norm").permute(0, 2, 1)  # [B, C, H]
        else:
            seq_last = x[:, -1:, :].detach()
            x = (x - seq_last).permute(0, 2, 1)

        # [B, C, L] -> [B*C, seg_num_x, seg_len]
        x = x.reshape(-1, self.seg_num_x, self.seg_len)
        x = self.value_embedding(x)

        res = x
        x = self.mamba1(x)
        x = self.norm(x + res + self.pos_emb_x)
        x = F.dropout(F.gelu(x), p=self.dropout, training=self.training)

        # project segment axis: seg_num_x -> seg_num_y
        x = self.to_y(x.transpose(1, 2)).transpose(1, 2)
        x = self.mamba2(x)

        # [B*C, seg_num_y, seg_len]
        y = self.predict(x)

        # -> [B, H, C]
        y = y.reshape(bsz, self.enc_in, self.seg_num_y, self.seg_len)
        y = y.permute(0, 1, 3, 2).reshape(bsz, self.enc_in, self.H).permute(0, 2, 1)

        if self.revin:
            y = self.revin_layer(y, "denorm")
        else:
            y = y + seq_last
        return y


import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from Hyperparameter import BaseConfigs


class SegMambaConfigs(BaseConfigs):
    """Default hyperparameters for SegMamba."""
    def __init__(self):
        super().__init__()
        self.seg_len  = 20
        self.d_state  = 16
        self.d_conv   = 4
        self.e_factor = 2
