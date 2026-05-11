import numpy as np


def split_data(data, train_ratio=0.6, val_ratio=0.2):
    train_size = int(train_ratio * len(data))
    val_size = int(val_ratio * len(data))
    return data[:train_size], data[train_size:train_size+val_size], data[train_size+val_size:]


def normalize(data, data_min, data_max):
    return (data - data_min) / (data_max - data_min)


def create_sequences(data, L, H, stride=1):
    if stride <= 0:
        raise ValueError(f"stride must be a positive integer, got {stride}")

    sequences, targets = [], []
    for i in range(0, len(data) - H - L + 1, stride):
        seq = data[i:i + L]
        target = data[i + L:i + H + L]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


def calculate_metrics(y_true, y_pred):
    metrics = {}
    metrics['MAE'] = round(np.mean(np.abs(y_true - y_pred)), 4)
    metrics['MSE'] = round(np.mean((y_true - y_pred) ** 2), 4)
    metrics['RMSE'] = round(np.sqrt(metrics['MSE']), 4)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    metrics['R2'] = round(1 - (ss_residual / ss_total), 4)
    return metrics

