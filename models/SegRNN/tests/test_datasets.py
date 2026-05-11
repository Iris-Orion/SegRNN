import os
import pytest
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_MAT = os.path.join(BASE, "data", "fft_decomposition_cleaned.mat")
M905_MAT  = os.path.join(BASE, "data", "matlab905(1).mat")

from datasets import CleanDataset, Matlab905Dataset, make_loaders
from Hyperparameter import H, L, batch_size


# ---------------------------------------------------------------------------
# CleanDataset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_clean_dataset_split(split):
    ds = CleanDataset(CLEAN_MAT, split=split, H=H, L=L, stride=20)
    assert len(ds) > 0, f"{split} split is empty"
    x, y = ds[0]
    assert x.shape == (H, 1)
    assert y.shape == (L, 1)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_clean_dataset_norm_range():
    ds = CleanDataset(CLEAN_MAT, split="train", H=H, L=L, stride=20)
    x, _ = ds[0]
    assert x.min() >= 0.0 and x.max() <= 1.0, "normalized values outside [0, 1]"


def test_clean_dataset_global_norm():
    # All splits use the same global min/max (normalize before split)
    train_ds = CleanDataset(CLEAN_MAT, split="train", H=H, L=L, stride=20)
    val_ds   = CleanDataset(CLEAN_MAT, split="val",   H=H, L=L, stride=20)
    assert train_ds.data_min == val_ds.data_min
    assert train_ds.data_max == val_ds.data_max


# ---------------------------------------------------------------------------
# Matlab905Dataset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_m905_dataset_split(split):
    ds = Matlab905Dataset(M905_MAT, var_name="data", use_hdf5=True,
                          split=split, H=H, L=L, stride=20)
    assert len(ds) > 0, f"{split} split is empty"
    x, y = ds[0]
    assert x.shape == (H, 1)
    assert y.shape == (L, 1)


def test_m905_dataset_norm_range():
    ds = Matlab905Dataset(M905_MAT, var_name="data", use_hdf5=True,
                          split="train", H=H, L=L, stride=20)
    x, _ = ds[0]
    assert x.min() >= 0.0 and x.max() <= 1.0


# ---------------------------------------------------------------------------
# make_loaders
# ---------------------------------------------------------------------------

def test_make_loaders_clean():
    tl, vl, tel, norm = make_loaders(
        CleanDataset, CLEAN_MAT, H=H, L=L, batch_size=batch_size,
    )
    assert len(tl) > 0 and len(vl) > 0 and len(tel) > 0
    x, y = next(iter(tl))
    assert x.shape == (batch_size, H, 1)
    assert y.shape == (batch_size, L, 1)


def test_make_loaders_m905():
    tl, vl, tel, norm = make_loaders(
        Matlab905Dataset, M905_MAT, H=H, L=L, batch_size=batch_size,
        var_name="data", use_hdf5=True,
    )
    assert len(tl) > 0 and len(vl) > 0 and len(tel) > 0
    x, y = next(iter(tl))
    assert x.shape == (batch_size, H, 1)
    assert y.shape == (batch_size, L, 1)
