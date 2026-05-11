import os
import sys
import time
import datetime
import numpy as np
import torch
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

sns.set_theme(style="ticks", palette="muted", font_scale=1.05)
sns.despine()

from split_data import split_data, create_sequences, calculate_metrics
from SegRNN import Model
from Hyperparameter import Configs, H, L, batch_size


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_device():
    cd = os.getenv("CUDA_DEVICE", "").strip()
    if cd:
        return torch.device(f"cuda:{cd}" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(base_dir, device, model_path=None):
    if model_path is None:
        model_path = os.path.join(base_dir, "best_model.pth")
    configs = Configs()
    model = Model(configs).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device),
        strict=False,
    )
    return model


def _make_out_dir(base_dir, model_name, dataset_name, ts):
    out_dir = os.path.join(base_dir, "outputs", model_name, dataset_name, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _run_inference(model, loader, device):
    """Returns (n_windows, L) array of predictions."""
    model.eval()
    outs = []
    with torch.no_grad():
        for x, _ in loader:
            out = model(x.to(device))              # (batch, L, 1)
            outs.append(out[..., 0].cpu().numpy()) # (batch, L)
    return np.concatenate(outs, axis=0)


# ---------------------------------------------------------------------------
# Color palette  (seaborn "muted" — low saturation, easy on the eyes)
# ---------------------------------------------------------------------------
_muted = sns.color_palette("muted")
_P = {
    "true":   _muted[0],   # muted blue
    "pred":   _muted[1],   # muted orange
    "error":  _muted[4],   # muted purple
    "zero":   "#666666",
    "split1": _muted[2],   # muted green  (train/val)
    "split2": _muted[3],   # muted red    (val/test)
    "mae":    _muted[0],
    "mse":    _muted[1],
    "rmse":   _muted[2],
}


def _style_ax(ax):
    sns.despine(ax=ax)
    ax.grid(True, linewidth=0.6, alpha=0.5)


def save_compare_plot(x, y_true, y_pred, title, path, xlabel="Index", downsample=15000):
    step = max(1, len(x) // downsample)
    xs, yt, yp = x[::step], y_true[::step], y_pred[::step]
    fig, ax = plt.subplots(figsize=(14, 5))
    _style_ax(ax)
    ax.plot(xs, yt, label="True",      color=_P["true"], linewidth=1.4)
    ax.plot(xs, yp, label="Predicted", color=_P["pred"], linewidth=1.4, alpha=0.85)
    ax.fill_between(xs, yt, yp, where=(yp >  yt), color=_P["pred"], alpha=0.12)
    ax.fill_between(xs, yt, yp, where=(yp <= yt), color=_P["true"], alpha=0.12)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_error_plot(x, y_true, y_pred, title, path, xlabel="Index", downsample=15000):
    step = max(1, len(x) // downsample)
    fig, ax = plt.subplots(figsize=(14, 4))
    _style_ax(ax)
    ax.plot(x[::step], (y_pred - y_true)[::step], color=_P["error"], linewidth=1.2, alpha=0.85)
    ax.axhline(0, color=_P["zero"], linewidth=1.2, linestyle="--", alpha=0.7)
    ax.fill_between(x[::step], (y_pred - y_true)[::step], 0,
                    where=((y_pred - y_true)[::step] > 0), color=_P["fill_o"], alpha=0.12)
    ax.fill_between(x[::step], (y_pred - y_true)[::step], 0,
                    where=((y_pred - y_true)[::step] <= 0), color=_P["fill_u"], alpha=0.12)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Error (Pred − True)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_bar_plot(metrics_dict, path):
    labels    = list(metrics_dict.keys())
    mae_vals  = [m["MAE"]  for m in metrics_dict.values()]
    mse_vals  = [m["MSE"]  for m in metrics_dict.values()]
    rmse_vals = [m["RMSE"] for m in metrics_dict.values()]
    x, width = np.arange(len(labels)), 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    _style_ax(ax)
    ax.bar(x - width, mae_vals,  width, label="MAE",  color=_P["mae"],  zorder=3)
    ax.bar(x,         mse_vals,  width, label="MSE",  color=_P["mse"],  zorder=3)
    ax.bar(x + width, rmse_vals, width, label="RMSE", color=_P["rmse"], zorder=3)
    ax.set_xlabel("Split", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title("Metrics Comparison", fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    png_path = os.path.splitext(path)[0] + ".png"
    fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


def _ax_compare(ax, x, y_true, y_pred, title, downsample=15000):
    step = max(1, len(x) // downsample)
    xs, yt, yp = x[::step], y_true[::step], y_pred[::step]
    _style_ax(ax)
    ax.plot(xs, yt, label="True",      color=_P["true"], linewidth=1.2)
    ax.plot(xs, yp, label="Predicted", color=_P["pred"], linewidth=1.2, alpha=0.85)
    ax.fill_between(xs, yt, yp, where=(yp >  yt), color=_P["pred"], alpha=0.12)
    ax.fill_between(xs, yt, yp, where=(yp <= yt), color=_P["true"], alpha=0.12)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Value", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)


def _ax_error(ax, x, y_true, y_pred, title, downsample=15000):
    step = max(1, len(x) // downsample)
    err = (y_pred - y_true)[::step]
    _style_ax(ax)
    ax.plot(x[::step], err, color=_P["error"], linewidth=1.1, alpha=0.85)
    ax.axhline(0, color=_P["zero"], linewidth=1.1, linestyle="--", alpha=0.7)
    ax.fill_between(x[::step], err, 0, where=(err >  0), color=_P["pred"], alpha=0.12)
    ax.fill_between(x[::step], err, 0, where=(err <= 0), color=_P["true"], alpha=0.12)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Error", fontsize=10)


def _print_safe(text):
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(str(text).encode(enc, errors="replace").decode(enc, errors="replace"))


def print_metrics(title, metrics):
    _print_safe(title)
    for k, v in metrics.items():
        _print_safe(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Mode 1: evaluate on fft_decomposition_cleaned (train / val / test splits)
# ---------------------------------------------------------------------------

def run_evaluate(base_dir=None, device=None, out_dir=None, model_path=None, swan_run=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    if device is None:
        device = _build_device()

    sequence_stride = int(os.getenv("SEQUENCE_STRIDE", "20"))
    mat_file       = os.path.join(base_dir, "data", "fft_decomposition_cleaned.mat")

    data = sio.loadmat(mat_file)
    if "clean_data" not in data:
        raise ValueError("clean_data not found in MAT file")
    rawdata  = np.asarray(data["clean_data"]).reshape(-1).astype(np.float64)
    data_min, data_max = rawdata.min(), rawdata.max()
    data1    = (rawdata - data_min) / (data_max - data_min)

    train_data, val_data, test_data = split_data(data1)
    X_train, Y_train = create_sequences(train_data, H, L, stride=sequence_stride)
    X_val,   Y_val   = create_sequences(val_data,   H, L, stride=sequence_stride)
    X_test,  Y_test  = create_sequences(test_data,  H, L, stride=sequence_stride)

    def trim(x, y):
        n = (len(x) // batch_size) * batch_size
        return x[:n], y[:n]

    X_train, Y_train = trim(X_train, Y_train)
    X_val,   Y_val   = trim(X_val,   Y_val)
    X_test,  Y_test  = trim(X_test,  Y_test)

    X_train = X_train.reshape(-1, H, 1);  Y_train = Y_train.reshape(-1, L, 1)
    X_val   = X_val.reshape(-1, H, 1);    Y_val   = Y_val.reshape(-1, L, 1)
    X_test  = X_test.reshape(-1, H, 1);   Y_test  = Y_test.reshape(-1, L, 1)

    def make_loader(X, Y):
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32).to(device),
                          torch.tensor(Y, dtype=torch.float32).to(device)),
            batch_size=batch_size, shuffle=False,
        )

    train_loader = make_loader(X_train, Y_train)
    val_loader   = make_loader(X_val,   Y_val)
    test_loader  = make_loader(X_test,  Y_test)

    ts         = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model      = _load_model(base_dir, device, model_path)
    if out_dir is None:
        out_dir = _make_out_dir(base_dir, "SegRNN",
                                os.path.splitext(os.path.basename(mat_file))[0], ts)
    else:
        os.makedirs(out_dir, exist_ok=True)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in model.buffers())
    print(f"{total_params:,} total parameters.")

    t0 = time.time()
    train_preds = _run_inference(model, train_loader, device)[:, 0]
    val_preds   = _run_inference(model, val_loader,   device)[:, 0]
    test_preds  = _run_inference(model, test_loader,  device)[:, 0]
    testing_time = time.time() - t0

    y_train, y_val, y_test = Y_train[:, 0, 0], Y_val[:, 0, 0], Y_test[:, 0, 0]
    train_metrics = calculate_metrics(y_train, train_preds)
    val_metrics   = calculate_metrics(y_val,   val_preds)
    test_metrics  = calculate_metrics(y_test,  test_preds)

    scale = data_max - data_min
    train_true = y_train * scale + data_min;  train_pred = train_preds * scale + data_min
    val_true   = y_val   * scale + data_min;  val_pred   = val_preds   * scale + data_min
    test_true  = y_test  * scale + data_min;  test_pred  = test_preds  * scale + data_min

    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Time: {ts}\nParams: {total_params}\nEval time: {testing_time:.4f}s\n\n")
        for name, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
            f.write(f"{name}:\n")
            for k, v in m.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    all_true  = np.concatenate([train_true, val_true, test_true])
    all_pred  = np.concatenate([train_pred, val_pred, test_pred])
    all_axis  = np.arange(len(all_true))
    train_end = len(train_true);  val_end = train_end + len(val_true)

    fig, axes = plt.subplot_mosaic(
        [["train", "val"],
         ["test",  "error"],
         ["all",   "all"]],
        figsize=(20, 18),
    )

    _ax_compare(axes["train"], np.arange(len(train_true)), train_true, train_pred, "Train: True vs Pred")
    _ax_compare(axes["val"],   np.arange(len(val_true)),   val_true,   val_pred,   "Validation: True vs Pred")
    _ax_compare(axes["test"],  np.arange(len(test_true)),  test_true,  test_pred,  "Test: True vs Pred")
    _ax_error(  axes["error"], np.arange(len(test_true)),  test_true,  test_pred,  "Test: Prediction Error")

    step_all = max(1, len(all_true) // 15000)
    ax_all = axes["all"]
    _style_ax(ax_all)
    ax_all.plot(all_axis[::step_all], all_true[::step_all], label="True",      color=_P["true"], linewidth=1.2)
    ax_all.plot(all_axis[::step_all], all_pred[::step_all], label="Predicted", color=_P["pred"], linewidth=1.2, alpha=0.85)
    ax_all.axvline(all_axis[train_end - 1], color=_P["split1"], linestyle="--", linewidth=1.5, label="Train/Val")
    ax_all.axvline(all_axis[val_end   - 1], color=_P["split2"], linestyle="--", linewidth=1.5, label="Val/Test")
    ax_all.set_title("Full Signal: True vs Predicted", fontsize=11, fontweight="bold")
    ax_all.set_xlabel("Global Index", fontsize=10); ax_all.set_ylabel("Value", fontsize=10)
    ax_all.legend(fontsize=9, loc="upper right", framealpha=0.9)

    fig.suptitle(
        f"Evaluation Summary  ·  Test MAE={test_metrics['MAE']:.4f}  RMSE={test_metrics['RMSE']:.4f}  R²={test_metrics['R2']:.4f}",
        fontsize=13, fontweight="bold", y=1.005, color="#222222",
    )
    fig.patch.set_facecolor("#FFFFFF")
    fig.tight_layout()
    summary_svg = os.path.join(out_dir, "evaluation_summary.svg")
    summary_png = os.path.join(out_dir, "evaluation_summary.png")
    fig.savefig(summary_svg, dpi=150, bbox_inches="tight")
    fig.savefig(summary_png, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics_png = save_metrics_bar_plot(
        {"Train": train_metrics, "Val": val_metrics, "Test": test_metrics},
        os.path.join(out_dir, "metrics_comparison.svg"),
    )

    if swan_run is not None:
        try:
            import swanlab
            swanlab.log({
                "eval/train_MAE":  train_metrics["MAE"],
                "eval/train_MSE":  train_metrics["MSE"],
                "eval/train_RMSE": train_metrics["RMSE"],
                "eval/train_R2":   train_metrics["R2"],
                "eval/val_MAE":    val_metrics["MAE"],
                "eval/val_MSE":    val_metrics["MSE"],
                "eval/val_RMSE":   val_metrics["RMSE"],
                "eval/val_R2":     val_metrics["R2"],
                "eval/test_MAE":   test_metrics["MAE"],
                "eval/test_MSE":   test_metrics["MSE"],
                "eval/test_RMSE":  test_metrics["RMSE"],
                "eval/test_R2":    test_metrics["R2"],
                "eval/summary":    swanlab.Image(summary_png, caption="Evaluation Summary"),
                "eval/metrics_bar":swanlab.Image(metrics_png,  caption="Metrics Comparison"),
            })
        except Exception as e:
            print(f"swanlab eval log failed: {e}")

    sio.savemat(os.path.join(out_dir, "model_outputs.mat"), {
        "train_true_norm":  y_train.reshape(-1, 1),
        "train_pred_norm":  train_preds.reshape(-1, 1),
        "val_true_norm":    y_val.reshape(-1, 1),
        "val_pred_norm":    val_preds.reshape(-1, 1),
        "test_true_norm":   y_test.reshape(-1, 1),
        "test_pred_norm":   test_preds.reshape(-1, 1),
        "train_true":       train_true.reshape(-1, 1),
        "train_pred":       train_pred.reshape(-1, 1),
        "val_true":         val_true.reshape(-1, 1),
        "val_pred":         val_pred.reshape(-1, 1),
        "test_true":        test_true.reshape(-1, 1),
        "test_pred":        test_pred.reshape(-1, 1),
        "all_true":         all_true.reshape(-1, 1),
        "all_pred":         all_pred.reshape(-1, 1),
        "split_train_end":  np.array([[train_end]], dtype=np.int32),
        "split_val_end":    np.array([[val_end]],   dtype=np.int32),
        "data_min":         np.array([[data_min]]),
        "data_max":         np.array([[data_max]]),
        "all_index":        all_axis.reshape(-1, 1),
    })

    print_metrics("Train", train_metrics)
    print_metrics("Val",   val_metrics)
    print_metrics("Test",  test_metrics)
    return {"train": train_metrics, "val": val_metrics, "test": test_metrics}


# ---------------------------------------------------------------------------
# Mode 2: predict on matlab905(1).mat (full-signal sliding-window inference)
# ---------------------------------------------------------------------------

def _load_hdf5_signal(mat_path, var_name):
    with h5py.File(mat_path, "r") as f:
        if var_name not in f:
            raise KeyError(f"'{var_name}' not in {mat_path}; available: {list(f.keys())}")
        ds = f[var_name]
        assert isinstance(ds, h5py.Dataset)
        sig = np.asarray(ds[...]).reshape(-1).astype(np.float64)
    if np.isnan(sig).any():
        sig = np.where(np.isnan(sig), np.nanmean(sig), sig)
    return sig


def _build_windows(signal, H_len, L_len, stride):
    starts = list(range(0, len(signal) - H_len - L_len + 1, stride))
    if not starts:
        raise ValueError(f"signal too short ({len(signal)}) for H+L={H_len + L_len}")
    X = np.stack([signal[s:s + H_len] for s in starts])
    Y = np.stack([signal[s + H_len:s + H_len + L_len] for s in starts])
    return X, Y, np.array(starts, dtype=np.int64)


def run_predict_matlab905(base_dir=None, device=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    if device is None:
        device = _build_device()

    ts          = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    input_mat   = os.getenv("INPUT_MAT",   os.path.join(base_dir, "data", "matlab905(1).mat"))
    input_var   = os.getenv("INPUT_VAR",   "data")
    norm_source = os.getenv("NORM_SOURCE", "self").lower()
    pred_stride = int(os.getenv("PRED_STRIDE", str(L)))

    print(f"Device: {device}")
    print(f"Input MAT: {input_mat}  var='{input_var}'")
    print(f"Norm source: {norm_source}  stride: {pred_stride}  H={H}  L={L}")

    signal = _load_hdf5_signal(input_mat, input_var)
    print(f"Signal length: {len(signal)}  raw range=[{signal.min():.6g}, {signal.max():.6g}]")

    if norm_source == "train":
        raw = np.asarray(sio.loadmat(os.path.join(base_dir, "data", "fft_decomposition_cleaned.mat"))
                         ["clean_data"]).reshape(-1).astype(np.float64)
        data_min, data_max = float(raw.min()), float(raw.max())
    else:
        data_min, data_max = float(signal.min()), float(signal.max())
    scale = data_max - data_min
    if scale == 0:
        raise ValueError("constant signal; cannot normalize")

    signal_norm = (signal - data_min) / scale
    X, Y, starts = _build_windows(signal_norm, H, L, pred_stride)
    n_trim = (len(X) // batch_size) * batch_size
    if n_trim == 0:
        raise ValueError(f"too few windows ({len(X)}) for batch_size={batch_size}")
    X, Y, starts = X[:n_trim], Y[:n_trim], starts[:n_trim]

    loader = DataLoader(
        TensorDataset(torch.tensor(X.reshape(-1, H, 1), dtype=torch.float32),
                      torch.tensor(Y.reshape(-1, L, 1), dtype=torch.float32)),
        batch_size=batch_size, shuffle=False,
    )

    model   = _load_model(base_dir, device)
    out_dir = _make_out_dir(base_dir, "SegRNN",
                            os.path.splitext(os.path.basename(input_mat))[0], ts)
    print(f"Loaded model  params={sum(p.numel() for p in model.parameters()):,}")

    t0        = time.time()
    pred_norm = _run_inference(model, loader, device)  # (n, L)
    print(f"Inference done in {time.time() - t0:.2f}s  pred shape={pred_norm.shape}")

    pred_denorm = pred_norm * scale + data_min
    true_denorm = Y        * scale + data_min

    series_pred  = np.full_like(signal, np.nan)
    series_count = np.zeros_like(signal)
    for i, s in enumerate(starts):
        sl   = slice(s + H, s + H + L)
        cur  = series_pred[sl]
        mask = np.isnan(cur)
        series_pred[sl]  = np.where(mask, pred_denorm[i],
                                    (cur * series_count[sl] + pred_denorm[i]) / (series_count[sl] + 1))
        series_count[sl] += 1
    valid = ~np.isnan(series_pred)

    yt, yp = signal[valid], series_pred[valid]
    mae  = float(np.mean(np.abs(yt - yp)))
    mse  = float(np.mean((yt - yp) ** 2))
    rmse = float(np.sqrt(mse))
    ss   = float(np.sum((yt - yt.mean()) ** 2))
    r2   = float(1 - np.sum((yt - yp) ** 2) / ss) if ss > 0 else float("nan")
    print(f"MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    idx = np.where(valid)[0]
    save_compare_plot(idx, yt, yp, "matlab905: True vs Predicted",
                      os.path.join(out_dir, "compare.svg"))
    save_error_plot(  idx, yt, yp, "matlab905: Prediction Error",
                      os.path.join(out_dir, "error.svg"))

    sio.savemat(os.path.join(out_dir, "predictions.mat"), {
        "signal":            signal.reshape(-1, 1),
        "series_pred":       series_pred.reshape(-1, 1),
        "valid_mask":        valid.reshape(-1, 1).astype(np.uint8),
        "window_starts":     starts.reshape(-1, 1),
        "windows_true_norm": Y,
        "windows_pred_norm": pred_norm,
        "windows_true":      true_denorm,
        "windows_pred":      pred_denorm,
        "data_min":          np.array([[data_min]]),
        "data_max":          np.array([[data_max]]),
        "H":                 np.array([[H]],           dtype=np.int32),
        "L":                 np.array([[L]],           dtype=np.int32),
        "pred_stride":       np.array([[pred_stride]], dtype=np.int32),
        "metrics_MAE":       np.array([[mae]]),
        "metrics_MSE":       np.array([[mse]]),
        "metrics_RMSE":      np.array([[rmse]]),
        "metrics_R2":        np.array([[r2]]),
    })
    print(f"Output dir: {out_dir}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "out_dir": out_dir}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    _base = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evaluate", "predict"], default="evaluate",
                        help="evaluate: train/val/test on clean data; predict: full inference on matlab905")
    parser.add_argument("--model-path", default=os.path.join(_base, "best_model.pth"),
                        help="Path to .pth checkpoint")
    args = parser.parse_args()

    if args.mode == "predict":
        run_predict_matlab905()
    else:
        _pth_dir  = os.path.dirname(os.path.abspath(args.model_path))
        _out_dir  = os.path.join(_pth_dir, "standalone_eval")
        run_evaluate(out_dir=_out_dir, model_path=args.model_path)
