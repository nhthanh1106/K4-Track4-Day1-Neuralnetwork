"""Data loading and train-only preprocessing for the CoverType lab."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_FEATURES = 54
N_NUMERIC = 10
N_CLASSES = 7


def load_split(processed_dir: str | Path = "data/processed",
               include_eval_labels: bool = True) -> dict:
    """Load the fixed train/eval split created by scripts/split_data.py."""
    processed_dir = Path(processed_dir)
    train_path = processed_dir / "train.npz"
    eval_path = processed_dir / "eval.npz"
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {eval_path}. Run scripts/split_data.py from the repository root."
        )

    with np.load(train_path, allow_pickle=False) as train, np.load(eval_path, allow_pickle=False) as ev:
        X_train = train["X"].astype(np.float32, copy=False)
        y_train = train["y"].astype(np.int64, copy=False)
        train_row_id = train["row_id"].astype(np.int64, copy=False)
        feature_names = train["feature_names"].astype(str)
        X_eval = ev["X"].astype(np.float32, copy=False)
        y_eval = ev["y"].astype(np.int64, copy=False) if include_eval_labels else None
        eval_row_id = ev["row_id"].astype(np.int64, copy=False)

    assert X_train.shape == (464_809, N_FEATURES), X_train.shape
    assert X_eval.shape == (116_203, N_FEATURES), X_eval.shape
    assert y_train.shape == (len(X_train),)
    if y_eval is not None:
        assert y_eval.shape == (len(X_eval),)
        assert y_train.min() == y_eval.min() == 0 and y_train.max() == y_eval.max() == 6
    else:
        assert y_train.min() == 0 and y_train.max() == 6
    assert len(np.unique(train_row_id)) == len(train_row_id)
    assert len(np.unique(eval_row_id)) == len(eval_row_id)
    assert not set(train_row_id).intersection(eval_row_id)
    return {
        "X_train_raw": X_train,
        "y_train": y_train,
        "train_row_id": train_row_id,
        "X_eval_raw": X_eval,
        "y_eval": y_eval,
        "eval_row_id": eval_row_id,
        "feature_names": feature_names,
    }


def load_eval_labels(processed_dir: str | Path = "data/processed") -> np.ndarray:
    """Load eval labels only after a final model has been selected and predictions saved."""
    path = Path(processed_dir) / "eval.npz"
    if not path.exists():
        raise FileNotFoundError(f"Expected {path}. Run scripts/split_data.py first.")
    with np.load(path, allow_pickle=False) as ev:
        labels = ev["y"].astype(np.int64, copy=False)
    assert labels.shape == (116_203,) and labels.min() == 0 and labels.max() == 6
    return labels


def make_val_split(X: np.ndarray, y: np.ndarray, val_fraction: float = 0.2,
                   seed: int = 42):
    """Split only the fixed train partition into stratified train/validation data."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr: np.ndarray):
    """Estimate mean and standard deviation from only the training rows."""
    if X_tr.ndim != 2 or X_tr.shape[1] != N_FEATURES:
        raise ValueError(f"Expected an N x {N_FEATURES} feature array")
    mean = X_tr[:, :N_NUMERIC].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X_tr[:, :N_NUMERIC].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize the 10 continuous columns, preserving the 44 binary columns."""
    result = np.array(X, dtype=np.float32, copy=True)
    result[:, :N_NUMERIC] = (result[:, :N_NUMERIC] - mean) / std
    return result


def prepare_data(device: str | torch.device, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str | Path = "data/processed") -> dict:
    """Load fixed splits, create validation, normalize from train statistics, and tensorize."""
    # Do not load y_eval here: the eval labels are opened only after validation has
    # selected the final model and predictions have been saved.
    parts = load_split(processed_dir, include_eval_labels=False)
    X_tr, y_tr, X_val, y_val = make_val_split(
        parts["X_train_raw"], parts["y_train"], val_fraction=val_fraction, seed=seed
    )
    mean, std = fit_standardizer(X_tr)

    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(parts["X_eval_raw"], mean, std)
    device = torch.device(device)

    data = {
        "X_tr": torch.as_tensor(X_tr, dtype=torch.float32, device=device),
        "y_tr": torch.as_tensor(y_tr, dtype=torch.int64, device=device),
        "X_val": torch.as_tensor(X_val, dtype=torch.float32, device=device),
        "y_val": torch.as_tensor(y_val, dtype=torch.int64, device=device),
        "X_eval": torch.as_tensor(X_eval, dtype=torch.float32, device=device),
        "eval_row_id": parts["eval_row_id"],
        "train_row_id": parts["train_row_id"],
        "standardizer_mean": mean,
        "standardizer_std": std,
        "feature_names": parts["feature_names"],
    }
    majority_acc = float(np.bincount(y_tr, minlength=N_CLASSES).max() / len(y_tr))
    data["majority_train_acc"] = majority_acc
    print(f"train/val/eval: {tuple(data['X_tr'].shape)} / {tuple(data['X_val'].shape)} / "
          f"{tuple(data['X_eval'].shape)}")
    print(f"train-only normalization: max |mean|={np.abs(X_tr[:, :N_NUMERIC].mean(0)).max():.5f}; "
          f"max |std-1|={np.abs(X_tr[:, :N_NUMERIC].std(0)-1).max():.5f}")
    print(f"majority-class accuracy on train: {majority_acc:.4f}")
    return data


def iterate_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int,
                    generator: torch.Generator | None = None, shuffle: bool = True):
    """Yield batches; the last batch may be smaller than batch_size."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if shuffle:
        perm = torch.randperm(len(X), generator=generator, device=X.device)
    else:
        perm = torch.arange(len(X), device=X.device)
    for start in range(0, len(X), batch_size):
        idx = perm[start:start + batch_size]
        yield X[idx], y[idx]
