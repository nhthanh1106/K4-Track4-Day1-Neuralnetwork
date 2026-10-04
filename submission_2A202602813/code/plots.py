"""Learning-curve figures for individual runs and controlled comparisons."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _history(result: dict) -> dict:
    history = result.get("history", {})
    if isinstance(history, list):
        keys = ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                "grad_norm", "epoch_time_s")
        return {key: [item.get(key) for item in history] for key in keys}
    return history


def plot_run(result: dict, path: str | Path) -> None:
    cfg = result["cfg"]
    summary = result["summary"]
    history = _history(result)
    epochs = history.get("epoch", [])
    if not epochs:
        raise ValueError(f"{cfg['exp_id']} has no completed epoch to plot")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].plot(epochs, history.get("train_loss", []), label="train loss (eval mode)")
    axes[0].plot(epochs, history.get("val_loss", []), label="validation loss")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(epochs, history.get("val_acc", []), label="validation accuracy")
    axes[1].plot(epochs, history.get("val_macro_f1", []), label="validation macro-F1")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    axes[2].plot(epochs, history.get("grad_norm", []), label="mean pre-clip grad norm")
    axes[2].set(title="Gradient norm", xlabel="Epoch", ylabel="L2 norm (before clipping)")
    axes[2].legend()

    best_epoch = summary.get("best_epoch")
    if best_epoch is not None:
        for axis in axes:
            axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.35,
                         label="best val-loss epoch" if axis is axes[0] else None)
    title = (
        f"{cfg['exp_id']} | {cfg['optimizer']} lr={cfg['lr']} | "
        f"loss={cfg['loss']} hidden={tuple(cfg['hidden'])} "
        f"dropout={cfg['dropout']} init={cfg['init']} seed={cfg['seed']}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str | Path,
                 title: str = "") -> None:
    if not results:
        raise ValueError("at least one result is required")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for result in results:
        history = _history(result)
        values = history.get(metric)
        epochs = history.get("epoch", [])
        if values is None or not epochs:
            continue
        ax.plot(epochs, values, label=result["cfg"]["exp_id"])
    ax.set(
        title=title or f"{metric.replace('_', ' ').title()} comparison",
        xlabel="Epoch",
        ylabel=metric.replace("_", " "),
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
