import time
import datetime
import numpy as np
import torch
import scipy.io as sio
import random
import os
import sys
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

from split_data import split_data, create_sequences, calculate_metrics
from SegRNN import Model
from Hyperparameter import Configs, H, L, batch_size, fs


def run_predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            out = model(x.to(device))
            preds.append(out[:, 0, 0].cpu().numpy())
    return np.concatenate(preds)


def save_compare_plot(x, y_true, y_pred, title, path, xlabel):
    plt.figure(figsize=(14, 6))
    plt.plot(x, y_true, label="True", color='blue', linewidth=1.5)
    plt.plot(x, y_pred, label="Predicted", color='red', linewidth=1.5, alpha=0.8)
    plt.fill_between(x, y_true, y_pred, where=(y_pred > y_true), color='red', alpha=0.1, label='Over-prediction')
    plt.fill_between(x, y_true, y_pred, where=(y_pred <= y_true), color='blue', alpha=0.1, label='Under-prediction')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def save_error_plot(x, y_true, y_pred, title, path, xlabel):
    error = y_pred - y_true
    plt.figure(figsize=(14, 6))
    plt.plot(x, error, label="Prediction Error", color='purple', linewidth=1.5)
    plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.7)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Error (Pred - True)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def save_metrics_bar_plot(metrics_dict, path):
    labels = list(metrics_dict.keys())
    mse_vals = [m['MSE'] for m in metrics_dict.values()]
    mae_vals = [m['MAE'] for m in metrics_dict.values()]
    rmse_vals = [m['RMSE'] for m in metrics_dict.values()]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, mse_vals, width, label='MSE', color='skyblue')
    ax.bar(x, mae_vals, width, label='MAE', color='lightgreen')
    ax.bar(x + width, rmse_vals, width, label='RMSE', color='salmon')

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Error Metrics')
    ax.set_title('Error Metrics Comparison Across Datasets', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def print_console_safe(text):
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_text = str(text).encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe_text)


def print_metrics(title, metrics):
    print_console_safe(title)
    for k, v in metrics.items():
        print_console_safe(f"  {k}: {v}")


def run_evaluate(base_dir=None, device=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mat_file = os.path.join(base_dir, "fft_decomposition_cleaned.mat")
    best_model_path = os.path.join(base_dir, "best_model.pth")
    metrics_file_path = os.path.join(base_dir, "metrics.txt")

    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sequence_stride = int(os.getenv("SEQUENCE_STRIDE", "20"))

    # Load data
    data = sio.loadmat(mat_file)
    if 'clean_data' not in data:
        raise ValueError("clean_data not found in MAT file")

    clean_data = np.asarray(data['clean_data'])
    rawdata = clean_data.reshape(-1)
    rawdata = np.asarray(rawdata, dtype=np.float64)

    data_min, data_max = rawdata.min(), rawdata.max()
    data1 = (rawdata - data_min) / (data_max - data_min)

    train_data, val_data, test_data = split_data(data1)

    X_train, Y_train = create_sequences(train_data, H, L, stride=sequence_stride)
    X_val,   Y_val   = create_sequences(val_data,   H, L, stride=sequence_stride)
    X_test,  Y_test  = create_sequences(test_data,  H, L, stride=sequence_stride)

    def trim_to_batch(x, y):
        n = (len(x) // batch_size) * batch_size
        return x[:n], y[:n]

    X_train, Y_train = trim_to_batch(X_train, Y_train)
    X_val,   Y_val   = trim_to_batch(X_val,   Y_val)
    X_test,  Y_test  = trim_to_batch(X_test,  Y_test)

    X_train = X_train.reshape(-1, H, 1)
    Y_train = Y_train.reshape(-1, L, 1)
    X_val   = X_val.reshape(-1, H, 1)
    Y_val   = Y_val.reshape(-1, L, 1)
    X_test  = X_test.reshape(-1, H, 1)
    Y_test  = Y_test.reshape(-1, L, 1)

    def make_loader(X, Y):
        xt = torch.tensor(X, dtype=torch.float32).to(device)
        yt = torch.tensor(Y, dtype=torch.float32).to(device)
        return DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=False)

    train_loader = make_loader(X_train, Y_train)
    val_loader   = make_loader(X_val,   Y_val)
    test_loader  = make_loader(X_test,  Y_test)

    # Load model
    configs = Configs()
    model = Model(configs).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in model.buffers())
    print(f"{total_params:,} total parameters.")

    # Inference
    start_time = time.time()
    train_pred_norm = run_predict(model, train_loader, device)
    val_pred_norm   = run_predict(model, val_loader,   device)
    test_pred_norm  = run_predict(model, test_loader,  device)
    testing_time = time.time() - start_time

    y_train = Y_train[:, 0, 0]
    y_val   = Y_val[:, 0, 0]
    y_test  = Y_test[:, 0, 0]

    train_metrics = calculate_metrics(y_train, train_pred_norm)
    val_metrics   = calculate_metrics(y_val,   val_pred_norm)
    test_metrics  = calculate_metrics(y_test,  test_pred_norm)

    # Denormalize
    scale = data_max - data_min
    train_true = y_train * scale + data_min
    train_pred = train_pred_norm * scale + data_min
    val_true   = y_val   * scale + data_min
    val_pred   = val_pred_norm   * scale + data_min
    test_true  = y_test  * scale + data_min
    test_pred  = test_pred_norm  * scale + data_min

    # Save metrics
    with open(metrics_file_path, "w", encoding="utf-8") as f:
        f.write(f"Time: {current_date}\n")
        f.write(f"Params: {total_params}\n")
        f.write(f"Eval time: {testing_time:.4f}s\n\n")
        for split_name, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
            f.write(f"{split_name}:\n")
            for k, v in m.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    # Plots
    ts = time.strftime("%Y%m%d-%H%M%S")

    save_compare_plot(np.arange(len(train_true)), train_true, train_pred,
                      "Train: True vs Pred", os.path.join(base_dir, f"dataset_train_compare_{ts}.svg"), "index")
    save_compare_plot(np.arange(len(val_true)), val_true, val_pred,
                      "Validation: True vs Pred", os.path.join(base_dir, f"dataset_val_compare_{ts}.svg"), "index")
    save_compare_plot(np.arange(len(test_true)), test_true, test_pred,
                      "Test: True vs Pred", os.path.join(base_dir, f"dataset_test_compare_{ts}.svg"), "index")

    save_error_plot(np.arange(len(test_true)), test_true, test_pred,
                    "Test: Prediction Error", os.path.join(base_dir, f"dataset_test_error_{ts}.svg"), "index")

    # All-in-one plot
    all_true = np.concatenate([train_true, val_true, test_true])
    all_pred = np.concatenate([train_pred, val_pred, test_pred])
    all_axis = np.arange(len(all_true))
    step_all = max(1, len(all_true) // 15000)
    train_end = len(train_true)
    val_end = train_end + len(val_true)

    plt.figure(figsize=(16, 6))
    plt.plot(all_axis[::step_all], all_true[::step_all], label="True", color='blue', linewidth=1.5)
    plt.plot(all_axis[::step_all], all_pred[::step_all], label="Predicted", color='red', linewidth=1.5, alpha=0.8)
    plt.axvline(all_axis[train_end - 1], color='green', linestyle="--", linewidth=2, label="Train/Val Split")
    plt.axvline(all_axis[val_end - 1],   color='orange', linestyle="--", linewidth=2, label="Val/Test Split")
    plt.title("All Dataset: True vs Predicted", fontsize=18, fontweight='bold')
    plt.xlabel("Global Index", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"dataset_all_compare_{ts}.svg"), dpi=300, bbox_inches='tight')
    plt.close()

    save_metrics_bar_plot(
        {'Train': train_metrics, 'Val': val_metrics, 'Test': test_metrics},
        os.path.join(base_dir, f"metrics_comparison_{ts}.svg")
    )

    # MAT output
    outputs_mat_path = os.path.join(base_dir, f"model_outputs_{ts}.mat")
    sio.savemat(outputs_mat_path, {
        "train_true_norm": y_train.reshape(-1, 1),
        "train_pred_norm": train_pred_norm.reshape(-1, 1),
        "val_true_norm":   y_val.reshape(-1, 1),
        "val_pred_norm":   val_pred_norm.reshape(-1, 1),
        "test_true_norm":  y_test.reshape(-1, 1),
        "test_pred_norm":  test_pred_norm.reshape(-1, 1),
        "train_true": train_true.reshape(-1, 1),
        "train_pred": train_pred.reshape(-1, 1),
        "val_true":   val_true.reshape(-1, 1),
        "val_pred":   val_pred.reshape(-1, 1),
        "test_true":  test_true.reshape(-1, 1),
        "test_pred":  test_pred.reshape(-1, 1),
        "all_true":   all_true.reshape(-1, 1),
        "all_pred":   all_pred.reshape(-1, 1),
        "split_train_end": np.array([[train_end]], dtype=np.int32),
        "split_val_end":   np.array([[val_end]],   dtype=np.int32),
        "data_min": np.array([[data_min]]),
        "data_max": np.array([[data_max]]),
        "all_index": all_axis.reshape(-1, 1),
    })
    print(f"Saved MAT file: {outputs_mat_path}")

    print_metrics("Train", train_metrics)
    print_metrics("Val",   val_metrics)
    print_metrics("Test",  test_metrics)

    return {"train": train_metrics, "val": val_metrics, "test": test_metrics}


if __name__ == "__main__":
    run_evaluate()
