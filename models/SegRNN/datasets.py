"""
Dataset classes for SegRNN training.

Two concrete datasets:
  CleanDataset    -- fft_decomposition_cleaned.mat  (var: 'clean_data')
  Matlab905Dataset -- matlab905 signal mat          (var: 'signal' or 'data')

Usage:
    from datasets import CleanDataset, Matlab905Dataset, make_loaders
    from Hyperparameter import H, L, batch_size

    # Dataset 1
    train_loader, val_loader, test_loader, norm_stats = make_loaders(
        CleanDataset, "fft_decomposition_cleaned.mat", H=H, L=L, batch_size=batch_size
    )

    # Dataset 2
    train_loader2, val_loader2, test_loader2, norm_stats2 = make_loaders(
        Matlab905Dataset, "matlab905_predictions_<ts>.mat", H=H, L=L, batch_size=batch_size
    )
"""

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

import h5py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_scipy(path, var_name):
    d = sio.loadmat(path)
    available = [k for k in d if not k.startswith("__")]
    if var_name not in d:
        raise KeyError(f"'{var_name}' not in {path}. Available: {available}")
    return np.asarray(d[var_name]).reshape(-1).astype(np.float64)


def _load_hdf5(path, var_name):
    with h5py.File(path, "r") as f:
        if var_name not in f:
            raise KeyError(f"'{var_name}' not in {path}. Available: {list(f.keys())}")
        ds = f[var_name]
        assert isinstance(ds, h5py.Dataset)
        return np.asarray(ds[...]).reshape(-1).astype(np.float64)


def _fill_nan(sig):
    if np.isnan(sig).any():
        sig = np.where(np.isnan(sig), np.nanmean(sig), sig)
    return sig


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _BaseTimeSeriesDataset(Dataset):
    """
    Base: load signal -> fill NaN -> normalize -> split -> sliding windows.

    Args:
        split       : 'train' | 'val' | 'test' | 'all'
        H           : look-back window length (model input)
        L           : forecast horizon (model target)
        stride      : sliding-window step
        train_ratio : fraction used for train
        val_ratio   : fraction used for val
        norm_stats  : (data_min, data_max) from the train split;
                      pass to val/test to prevent leakage
    """

    def _load_raw_signal(self) -> np.ndarray:
        raise NotImplementedError

    def __init__(self, split="train", H=400, L=200, stride=20,
                 train_ratio=0.6, val_ratio=0.2):
        raw = _fill_nan(self._load_raw_signal())
        self.split = split
        self.H, self.L = H, L

        # Normalize full signal first, then split (global min/max)
        self.data_min = float(raw.min())
        self.data_max = float(raw.max())
        scale = self.data_max - self.data_min
        if scale == 0:
            raise ValueError("Constant signal; cannot normalize.")
        normalized = (raw - self.data_min) / scale

        n = len(normalized)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)

        seg = {"train": normalized[:train_end],
               "val":   normalized[train_end:val_end],
               "test":  normalized[val_end:],
               "all":   normalized}.get(split)
        if seg is None:
            raise ValueError(f"split must be 'train'/'val'/'test'/'all', got {split!r}")

        starts = list(range(0, len(seg) - H - L + 1, stride))
        if not starts:
            raise ValueError(
                f"Segment too short ({len(seg)}) for H={H}+L={L}, stride={stride}."
            )
        self.X = np.array([seg[s: s + H] for s in starts], dtype=np.float32)
        self.Y = np.array([seg[s + H: s + H + L] for s in starts], dtype=np.float32)
        self.n_samples = len(self.X)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx]).unsqueeze(-1),   # (H, 1)
                torch.from_numpy(self.Y[idx]).unsqueeze(-1))   # (L, 1)

    def __repr__(self):
        return (f"{self.__class__.__name__}(split={self.split!r}, n={self.n_samples}, "
                f"H={self.H}, L={self.L}, "
                f"norm=[{self.data_min:.4g}, {self.data_max:.4g}])")


# ---------------------------------------------------------------------------
# Dataset 1: fft_decomposition_cleaned.mat  ->  var 'clean_data'
# ---------------------------------------------------------------------------

class CleanDataset(_BaseTimeSeriesDataset):
    """
    Loads fft_decomposition_cleaned.mat (scipy v5 format, var='clean_data').

    Args:
        mat_path : path to fft_decomposition_cleaned.mat
    """

    def __init__(self, mat_path: str, **kwargs):
        self._mat_path = mat_path
        super().__init__(**kwargs)

    def _load_raw_signal(self):
        return _load_scipy(self._mat_path, "clean_data")


# ---------------------------------------------------------------------------
# Dataset 2: matlab905 signal mat  ->  var 'signal' (scipy) or 'data' (hdf5)
# ---------------------------------------------------------------------------

class Matlab905Dataset(_BaseTimeSeriesDataset):
    """
    Loads the matlab905 signal from a .mat file.

    Supports:
      - matlab905_predictions_<ts>.mat  var_name='signal'  (scipy v5)
      - matlab905(1).mat                var_name='data'    (MATLAB v7.3 / hdf5)

    Args:
        mat_path : path to the .mat file
        var_name : variable name inside the file (default 'signal')
        use_hdf5 : True to force h5py (for MATLAB v7.3 files like matlab905(1).mat)
    """

    def __init__(self, mat_path: str, var_name: str = "data",
                 use_hdf5: bool = True, **kwargs):
        self._mat_path = mat_path
        self._var_name = var_name
        self._use_hdf5 = use_hdf5
        super().__init__(**kwargs)

    def _load_raw_signal(self):
        if self._use_hdf5:
            return _load_hdf5(self._mat_path, self._var_name)
        try:
            return _load_scipy(self._mat_path, self._var_name)
        except Exception:
            return _load_hdf5(self._mat_path, self._var_name)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_loaders(dataset_cls, mat_path, H, L, batch_size,
                 stride=20, train_ratio=0.6, val_ratio=0.2, **ds_kwargs):
    """Build (train_loader, val_loader, test_loader, norm_stats)."""
    from torch.utils.data import DataLoader

    common = dict(H=H, L=L, stride=stride,
                  train_ratio=train_ratio, val_ratio=val_ratio, **ds_kwargs)

    train_ds = dataset_cls(mat_path, split="train", **common)
    val_ds   = dataset_cls(mat_path, split="val",   **common)
    test_ds  = dataset_cls(mat_path, split="test",  **common)

    def _loader(ds, shuffle):
        n = (len(ds) // batch_size) * batch_size
        ds.X = ds.X[:n]; ds.Y = ds.Y[:n]; ds.n_samples = n
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          pin_memory=torch.cuda.is_available(),
                          num_workers=4, persistent_workers=True)

    return (
        _loader(train_ds, shuffle=True),
        _loader(val_ds,   shuffle=False),
        _loader(test_ds,  shuffle=False),
        (train_ds.data_min, train_ds.data_max),
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    from Hyperparameter import H, L, batch_size

    data_dir = os.path.join(base, "data")

    print("=== CleanDataset ===")
    for split in ("train", "val", "test"):
        ds = CleanDataset(os.path.join(data_dir, "fft_decomposition_cleaned.mat"),
                          split=split, H=H, L=L, stride=20)
        print(f"  {ds}")

    print("\n=== Matlab905Dataset (var='signal') ===")
    pred_mats = sorted([f for f in os.listdir(base)
                        if f.startswith("matlab905_predictions_") and f.endswith(".mat")])
    if pred_mats:
        pmat = os.path.join(base, pred_mats[-1])
        for split in ("train", "val", "test"):
            ds2 = Matlab905Dataset(pmat, var_name="signal",
                                   split=split, H=H, L=L, stride=20)
            print(f"  {ds2}")

    print("\n=== make_loaders ===")
    tl, vl, tel, ns = make_loaders(
        CleanDataset, os.path.join(data_dir, "fft_decomposition_cleaned.mat"),
        H=H, L=L, batch_size=batch_size,
    )
    print(f"  train={len(tl)} val={len(vl)} test={len(tel)} batches")
    x, y = next(iter(tl))
    print(f"  x={x.shape}  y={y.shape}  norm=[{ns[0]:.4g}, {ns[1]:.4g}]")
