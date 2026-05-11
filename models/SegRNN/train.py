import time
import datetime
import os
from contextlib import nullcontext
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import swanlab 
from datasets import CleanDataset, Matlab905Dataset, make_loaders
from SegRNN import Model
from Hyperparameter import (
    Configs, H, L, batch_size, epochs, lr,
    use_early_stopping, early_stopping_patience, parse_args,
)
from evaluate import run_evaluate
from utils import (
    seed_torch, build_loss, build_optimizer,
    build_scheduler, train_one_epoch, eval_one_epoch,
)

if __name__ == "__main__":
    args   = parse_args()
    device = args.device
    H, L   = args.H, args.L   # may override Hyperparameter defaults

    # Configs
    configs = Configs()
    configs.L              = H   # Configs.L = input history length
    configs.H              = L   # Configs.H = forecast horizon
    configs.loss_name      = args.loss
    configs.optimizer_name = args.optimizer
    configs.seed           = args.seed

    seed_torch(configs)
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # AMP
    device_type = device.type
    if device_type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_ctx   = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_dtype = None
        amp_ctx   = nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    effective_lr       = args.lr if args.lr is not None else lr
    use_early_stopping = args.early_stopping
    grad_clip_norm     = args.grad_clip

    base_dir     = os.path.dirname(os.path.abspath(__file__))
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.dataset == "matlab905":
        dataset_cls  = Matlab905Dataset
        mat_file     = os.path.join(base_dir, "data", "matlab905(1).mat")
        dataset_name = "matlab905"
    else:
        dataset_cls  = CleanDataset
        mat_file     = os.path.join(base_dir, "data", "fft_decomposition_cleaned.mat")
        dataset_name = "fft_decomposition_cleaned"

    out_dir = os.path.join(base_dir, "outputs", "SegRNN", dataset_name, current_date)
    os.makedirs(out_dir, exist_ok=True)

    swan_run = None
    if args.swanlog:
        try:
            swan_run = swanlab.init(
                project="SegRNN Training",
                experiment_name=f"optloss_{current_date}",
                mode=args.swanlab_mode,
                config={
                    "optimizer": configs.optimizer_name,
                    "loss":      configs.loss_name,
                    "lr":        effective_lr,
                    "warmup":        args.warmup,
                    "warmup_ratio":  args.warmup_ratio,
                    "early_stopping":       use_early_stopping,
                    "early_stopping_patience": early_stopping_patience,
                    "grad_clip_norm": grad_clip_norm,
                    "batch_size": batch_size,
                    "L": L, "H": H,
                    "seg_len": configs.seg_len,
                },
            )
            print(f"swanlab initialized (mode={args.swanlab_mode}).")
        except Exception as e:
            print(f"swanlab init failed, disabled logging: {e}")

    start_time = time.time()

    # Data
    train_loader, val_loader, _, _ = make_loaders(
        dataset_cls, mat_file, H, L, batch_size, stride=args.stride
    )

    # Model
    model = Model(configs).to(device)

    criterion = build_loss(configs)
    optimizer = build_optimizer(configs, model.parameters(), effective_lr)

    total_steps = epochs * len(train_loader)
    scheduler = build_scheduler(
        optimizer, total_steps, effective_lr, args.warmup, args.warmup_ratio
    )

    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in model.buffers())
    print(f"{total_params:,} total parameters.")
    initial_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)

    # Training loop
    best_val_loss    = float("inf")
    best_train_loss  = float("inf")
    best_model_state = None
    best_metrics     = None
    patience_counter = 0
    train_losses, val_losses = [], []

    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, grad_clip_norm, epoch, epochs, amp_ctx, scaler,
        )
        val_loss = eval_one_epoch(model, val_loader, criterion, device, epoch, epochs, amp_ctx)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        if swan_run is not None:
            swanlab.log(
                {"train_loss": float(train_loss), "val_loss": float(val_loss), "lr": float(current_lr)},
                step=epoch + 1,
            )

        if train_loss < best_train_loss:
            best_train_loss  = train_loss
            best_model_state = model.state_dict()

        is_best_val = val_loss < best_val_loss
        if is_best_val:
            best_val_loss    = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        best_tag = " [best_val]" if is_best_val else ""
        epoch_bar.set_postfix(train=f"{train_loss:.6f}", val=f"{val_loss:.6f}", lr=f"{current_lr:.2e}")
        tqdm.write(
            f"Epoch {epoch+1}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}"
            f"  lr={current_lr:.8f}{best_tag}"
        )

        if use_early_stopping and patience_counter >= early_stopping_patience:
            tqdm.write(f"Early stopping at epoch {epoch + 1}")
            break

    # Save best model
    if best_val_loss < (best_metrics["val_loss"] if best_metrics else float("inf")):
        best_metrics    = {"H": H, "L": L, "batch_size": batch_size, "val_loss": best_val_loss}
        best_model_path = os.path.join(out_dir, "best_model.pth")
        torch.save(best_model_state, best_model_path)
        # keep a fixed-path copy for evaluate.py
        torch.save(best_model_state, os.path.join(base_dir, "best_model.pth"))
        print(f"Saved best model to: {best_model_path}")

    # Loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss",   color="blue", linestyle="-", marker="o")
    plt.plot(range(1, len(val_losses)   + 1), val_losses,   label="Validation Loss", color="red",  linestyle="-", marker="s")
    plt.title("Training & Validation Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_curve.svg"), format="svg", dpi=300)

    end_time         = time.time()
    training_duration = end_time - start_time
    memory_usage     = torch.cuda.max_memory_allocated(device) / (1024 ** 2) - initial_memory

    summary_path = os.path.join(out_dir, "best_params_and_metrics.txt")
    with open(summary_path, "w") as f:
        f.write("Optimized Parameters and Metrics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Training Time: {training_duration:.4f} seconds\n")
        f.write(f"Memory Usage: {memory_usage:.4f} MB\n")
        f.write("-" * 50 + "\n")
        if best_metrics is not None:
            f.write("IMF 1 Results:\n")
            for key, value in best_metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("-" * 50 + "\n")
    print(f"Best metrics saved to {summary_path}")

    print("\n=== Running evaluation ===")
    run_evaluate(base_dir=base_dir, device=device, out_dir=out_dir)
