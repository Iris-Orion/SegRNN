import time
import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.nn import LayerNorm
from split_data import split_data,create_sequences,calculate_metrics

import numpy as np
from torch import nn, optim
import traceback 
import os


from SegRNN import Model
from Hyperparameter import Configs, H, L, batch_size, epochs, lr, use_early_stopping, early_stopping_patience
from evaluate import run_evaluate


def _env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _env_float(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


# Optional experiment knobs with safe defaults.
LOSS_NAME = os.getenv("LOSS_NAME", "l1").lower()
LR_OVERRIDE = _env_float("LR_OVERRIDE", None)
OPTIMIZER_NAME = os.getenv("OPTIMIZER_NAME", "adam").lower()
USE_SCHEDULER = _env_flag("USE_SCHEDULER", True)
USE_WARMUP = _env_flag("USE_WARMUP", False)
WARMUP_EPOCHS = _env_int("WARMUP_EPOCHS", 0)
USE_SWANLOG = _env_flag("USE_SWANLOG", False)
TRAIN_SWANLAB_MODE = os.getenv("TRAIN_SWANLAB_MODE", "offline").strip().lower()
USE_EARLY_STOPPING = _env_flag("USE_EARLY_STOPPING", use_early_stopping)
GRAD_CLIP_NORM = _env_float("GRAD_CLIP_NORM", None)
seg_len = getattr(Configs(), "seg_len", None)

try:
    import swanlab  # type: ignore
except Exception as e:
    print(f"swanlab import failed: {e}")
    swanlab = None


_cuda_device = os.getenv("CUDA_DEVICE", "").strip()
if _cuda_device:
    device = torch.device(f"cuda:{_cuda_device}" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sequence_stride = int(os.getenv("SEQUENCE_STRIDE", "20"))


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch(2048)
torch.cuda.empty_cache()

base_dir = os.path.dirname(os.path.abspath(__file__))
mat_file = os.path.join(base_dir, "fft_decomposition_cleaned.mat")
log_file = os.path.join(base_dir, "log.txt")
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
best_model_filename = os.path.join(base_dir, "best_model.pth")
file_path = os.path.join(base_dir, "best_params_and_metrics.txt")
script_dir = os.path.dirname(os.path.abspath(__file__))

try:
    data = sio.loadmat(mat_file)
    print(list(data.keys()))
    if 'clean_data' in data:
        clean_data = np.asarray(data['clean_data'])
        print(f"clean_data shape: {clean_data.shape}")
    else:
        print("clean_data not found in MAT file")
        exit()
except Exception as e:
    print(f"Failed to load .mat file: {e}")
    exit()


rawdata = clean_data.reshape(-1)


print("rawdata.shape", rawdata.shape)
print("Processing signal...")

if np.isnan(rawdata).any():
    print("Data contains NaN, filling with mean...")
    nan_mask = np.isnan(rawdata)
    mean_value = np.nanmean(rawdata)
    rawdata[nan_mask] = mean_value


data_min, data_max = rawdata.min(), rawdata.max()
data1 = (rawdata - data_min) / (data_max - data_min)


train_data, val_data, test_data = split_data(data1)


best_metrics = None
best_model_state = None



start_time = time.time()

X_train, Y_train = create_sequences(train_data, H, L, stride=sequence_stride)
X_val, Y_val = create_sequences(val_data, H, L, stride=sequence_stride)

if __name__ == "__main__":
    num_batches_train = len(X_train) // batch_size
    num_batches_val = len(X_val) // batch_size
    
    X_train = X_train[:num_batches_train * batch_size]
    Y_train = Y_train[:num_batches_train * batch_size]
    X_val = X_val[:num_batches_val * batch_size]
    Y_val = Y_val[:num_batches_val * batch_size]
    
    X_train = X_train.reshape(-1, H, 1)
    Y_train = Y_train.reshape(-1, L, 1)
    X_val = X_val.reshape(-1, H, 1)
    Y_val = Y_val.reshape(-1, L, 1)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(device)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val_tensor, Y_val_tensor), batch_size=batch_size, shuffle=False)
    
    configs = Configs()
    model = Model(configs).to(device)
    
    def build_loss(name):
        key = name.lower()
        if key == "l1":
            return nn.L1Loss()
        if key == "mse":
            return nn.MSELoss()
        if key == "huber":
            return nn.SmoothL1Loss(beta=1.0)
        raise ValueError(f"Unsupported loss: {name}")
    
    
    def build_optimizer(name, params, base_lr):
        key = name.lower()
        if key == "adam":
            return optim.Adam(params, lr=base_lr)
        if key == "adamw":
            return optim.AdamW(params, lr=base_lr, weight_decay=1e-2)
        if key == "sgd":
            return optim.SGD(params, lr=base_lr, momentum=0.9, nesterov=True)
        raise ValueError(f"Unsupported optimizer: {name}")
    
    '''criterion = nn.L1Loss()
    effective_lr = LR_OVERRIDE if LR_OVERRIDE is not None else lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)'''
    criterion = build_loss(LOSS_NAME)
    effective_lr = LR_OVERRIDE if LR_OVERRIDE is not None else lr
    optimizer = build_optimizer(OPTIMIZER_NAME, model.parameters(), effective_lr)
    
    if USE_SCHEDULER:
        eta_min = max(effective_lr * 0.1, 1e-6)
        if USE_WARMUP and WARMUP_EPOCHS > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / max(WARMUP_EPOCHS, 1),
                end_factor=1.0,
                total_iters=WARMUP_EPOCHS,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - WARMUP_EPOCHS),
                eta_min=eta_min,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[WARMUP_EPOCHS],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=eta_min
            )
    else:
        scheduler = None
    
    swan_run = None
    if USE_SWANLOG:
        if swanlab is None:
            print("swanlab is not installed, skipping swanlog.")
        else:
            try:
                swan_run = swanlab.init(
                    project="SegRNN Training",
                    experiment_name=f"optloss_{current_date}",
                    mode=TRAIN_SWANLAB_MODE,
                    config={
                        "optimizer": OPTIMIZER_NAME,
                        "loss": LOSS_NAME,
                        "lr": effective_lr,
                        "scheduler": USE_SCHEDULER,
                        "warmup": USE_WARMUP,
                        "warmup_epochs": WARMUP_EPOCHS,
                        "early_stopping": USE_EARLY_STOPPING,
                        "early_stopping_patience": early_stopping_patience,
                        "grad_clip_norm": GRAD_CLIP_NORM,
                        "batch_size": batch_size,
                        "L": L,
                        "H": H,
                        "seg_len": seg_len,
                    },
                )
                print(f"swanlab initialized (mode={TRAIN_SWANLAB_MODE}).")
            except Exception as e:
                swan_run = None
                print(f"swanlab init failed, disabled logging: {e}")
    
    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in model.buffers())
    print(f'{total_params:,} total parameters.')
    initial_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # convert to MB
    
    
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs} train", leave=False):
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            y = model(inputs)
            loss = criterion(y, targets)
            loss.backward()
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_loss = 0.0
        val_predictions = []
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"  Epoch {epoch+1}/{epochs} val  ", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                y = model(inputs)
                val_predictions.append(y[:, 0, 0].cpu().numpy())
                val_loss += criterion(y, targets).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        if swan_run is not None:
            swanlab.log(
                {
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "lr": float(current_lr),
                },
                step=epoch + 1,
            )

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_model_state = model.state_dict()

        is_best_val = val_loss < best_val_loss
        if is_best_val:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        best_tag = " [best_val]" if is_best_val else ""
        epoch_bar.set_postfix(train=f"{train_loss:.6f}", val=f"{val_loss:.6f}", lr=f"{current_lr:.2e}")
        tqdm.write(
            f"Epoch {epoch+1}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}  lr={current_lr:.8f}{best_tag}"
        )

        if USE_EARLY_STOPPING and patience_counter >= early_stopping_patience:
            tqdm.write(f"Early stopping at epoch {epoch + 1}")
            break
    
    
    # ===========================
    # Save best model
    # ===========================
    if best_val_loss < (best_metrics['val_loss'] if best_metrics else float('inf')):
        best_metrics = {
            'H': H,
            'L': L,
            'batch_size': batch_size,
            'val_loss': best_val_loss
        }
    
        best_model_path = os.path.join(script_dir, best_model_filename)
        torch.save(best_model_state, best_model_path)
        print(f"Saved best model to: {best_model_path}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", color="blue", linestyle="-", marker="o")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", color="red", linestyle="-", marker="s")
    
    plt.title("Training & Validation Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Save to script_dir
    loss_curve_path = os.path.join(script_dir, "loss_curve.svg")
    plt.savefig(loss_curve_path, format="svg", dpi=300)
    # plt.show()
    
    end_time = time.time()
    training_duration = end_time - start_time
    
    # Record final GPU memory usage
    final_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    memory_usage = final_memory - initial_memory
    
    # Save params, time and memory usage
    with open(file_path, "w") as f:
        f.write("Optimized Parameters and Metrics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Training Time: {training_duration:.4f} seconds\n")
        f.write(f"Memory Usage: {memory_usage:.4f} MB\n")
        f.write("-" * 50 + "\n")
    
        if best_metrics is not None:
            f.write(f"IMF 1 Results:\n")
            for key, value in best_metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("-" * 50 + "\n")
    
    print(f"Best metrics saved to {file_path}")

    print("\n=== Running evaluation ===")
    run_evaluate(base_dir=script_dir, device=device)















