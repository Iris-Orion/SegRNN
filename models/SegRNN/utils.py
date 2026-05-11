import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def seed_torch(configs):
    random.seed(configs.seed)
    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    torch.cuda.manual_seed(configs.seed)


def build_loss(configs):
    key = configs.loss_name
    if key == "l1":    return nn.L1Loss()
    if key == "mse":   return nn.MSELoss()
    if key == "huber": return nn.SmoothL1Loss(beta=1.0)
    raise ValueError(f"Unsupported loss: {key}")


def build_optimizer(configs, params, base_lr):
    key = configs.optimizer_name
    if key == "adam":  return optim.Adam(params, lr=base_lr)
    if key == "adamw": return optim.AdamW(params, lr=base_lr, weight_decay=1e-2)
    if key == "sgd":   return optim.SGD(params, lr=base_lr, momentum=0.9, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {key}")


def build_scheduler(optimizer, total_steps, effective_lr, use_warmup, warmup_ratio=0.1):
    """
    Step-based scheduler. Call scheduler.step() after every optimizer.step().

    Two modes:
    - use_warmup=False : CosineAnnealingLR over total_steps, lr -> lr * 0.1
    - use_warmup=True  : LinearLR warmup for (warmup_ratio * total_steps) steps,
                         then CosineAnnealingLR for the remaining steps, lr -> lr * 0.1
    """
    eta_min = effective_lr * 0.1

    if use_warmup:
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=eta_min,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=eta_min
    )


def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    device, grad_clip_norm, epoch, epochs, amp_ctx, scaler):
    model.train()
    running_loss = 0.0

    it = iter(loader)
    inputs, targets = next(it)
    inputs  = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    pbar = tqdm(range(len(loader)), desc=f"  Epoch {epoch+1}/{epochs} train", leave=False)
    for _ in pbar:
        try:
            next_x, next_y = next(it)
            next_x = next_x.to(device, non_blocking=True)
            next_y = next_y.to(device, non_blocking=True)
        except StopIteration:
            next_x = next_y = None

        optimizer.zero_grad()
        with amp_ctx:
            loss = criterion(model(inputs), targets)
        scaler.scale(loss).backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item()

        if next_x is not None:
            inputs, targets = next_x, next_y

    return running_loss / len(loader)


def eval_one_epoch(model, loader, criterion, device, epoch, epochs, amp_ctx):
    model.eval()
    total_loss = 0.0

    it = iter(loader)
    inputs, targets = next(it)
    inputs  = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    with torch.no_grad():
        pbar = tqdm(range(len(loader)), desc=f"  Epoch {epoch+1}/{epochs} val  ", leave=False)
        for _ in pbar:
            try:
                next_x, next_y = next(it)
                next_x = next_x.to(device, non_blocking=True)
                next_y = next_y.to(device, non_blocking=True)
            except StopIteration:
                next_x = next_y = None

            with amp_ctx:
                total_loss += criterion(model(inputs), targets).item()

            if next_x is not None:
                inputs, targets = next_x, next_y

    return total_loss / len(loader)
