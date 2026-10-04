"""Reproducible training, validation, and final prediction helpers."""
from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, activation_stats, count_params
from optimizer import build_optimizer, clip_gradients

DEFAULT_CFG = dict(
    exp_id="baseline_s42", group="baseline", description="Baseline M-base",
    hypothesis="He + CE + SGD with momentum should beat the majority-class baseline.",
    loss="ce", optimizer="sgd_momentum", lr=0.05, weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20, hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None, precision="fp32", seed=42, eligible_for_best=True,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return float(f1.mean())


@torch.inference_mode()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    model.eval()
    pieces = [model(X[start:start + batch_size]).argmax(dim=1)
              for start in range(0, len(X), batch_size)]
    if not pieces:
        return torch.empty(0, dtype=torch.int64, device=X.device)
    return torch.cat(pieces).to(dtype=torch.int64)


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        target = F.one_hot(y, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        return F.mse_loss(torch.softmax(logits, dim=1), target)
    raise ValueError("loss must be 'ce' or 'mse'")


@torch.inference_mode()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    model.eval()
    loss_total = 0.0
    seen = 0
    all_preds = []
    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size]
        yb = y[start:start + batch_size]
        logits = model(xb)
        batch_loss = compute_loss(logits, yb, loss_name)
        if not torch.isfinite(batch_loss):
            return {"loss": float("nan"), "acc": float("nan"), "macro_f1": float("nan")}
        batch_size_actual = len(yb)
        loss_total += float(batch_loss.item()) * batch_size_actual
        seen += batch_size_actual
        all_preds.append(logits.argmax(dim=1).to(dtype=torch.int64))
    if seen == 0:
        raise ValueError("cannot evaluate an empty dataset")
    pred = torch.cat(all_preds)
    accuracy = float((pred == y).float().mean().item())
    cm = torch.bincount(y.to(torch.int64) * 7 + pred, minlength=49).reshape(7, 7)
    return {
        "loss": loss_total / seen,
        "acc": accuracy,
        "macro_f1": macro_f1_from_confusion(cm.cpu().numpy()),
    }


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 was requested but CUDA is unavailable")
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    if precision == "bf16":
        dtype = torch.bfloat16
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("this CUDA device does not support BF16")
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)
    raise ValueError("precision must be 'fp32', 'fp16', or 'bf16'")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Train one configuration and keep the checkpoint with lowest validation loss."""
    cfg = {**DEFAULT_CFG, **cfg}
    cfg["hidden"] = tuple(cfg["hidden"])
    set_seed(int(cfg["seed"]))
    device = data["X_tr"].device
    model = MLP(hidden=cfg["hidden"], dropout=float(cfg["dropout"]), init=cfg["init"]).to(device)
    expected = EXPECTED_PARAMS.get(tuple(cfg["hidden"]))
    if expected is not None:
        assert count_params(model) == expected
    optimizer = build_optimizer(
        cfg["optimizer"], model.parameters(), lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        momentum=float(cfg.get("momentum", 0.9)),
    )

    use_scaler = cfg["precision"] == "fp16" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    step0 = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])
    init_stats = activation_stats(model, data["X_val"][:min(4096, len(data["X_val"]))])

    # A fixed 50,000-row subset gives an eval-mode train-loss curve without extra
    # training-time dropout noise or a second full-dataset forward pass per epoch.
    train_eval_n = min(50_000, len(data["X_tr"]))
    X_train_eval = data["X_tr"][:train_eval_n]
    y_train_eval = data["y_tr"][:train_eval_n]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = {key: [] for key in (
        "epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
        "grad_norm", "grad_norm_max", "clip_fraction", "overflow_steps", "epoch_time_s"
    )}
    best_loss = float("inf")
    best_epoch = None
    best_state = None
    best_metrics = None
    diverged = False
    generator = torch.Generator(device=device).manual_seed(int(cfg["seed"]))
    epoch_times = []
    total_start = time.perf_counter()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        grad_norms = []
        clipped_steps = 0
        overflow_steps = 0

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], int(cfg["batch"]),
                                      generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, cfg["precision"]):
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
            if not torch.isfinite(loss):
                diverged = True
                break

            scaler.scale(loss).backward()
            if use_scaler:
                scaler.unscale_(optimizer)
            grad_norm = clip_gradients(model.parameters(), cfg["clip_norm"])
            if not np.isfinite(grad_norm):
                if use_scaler:
                    # Let GradScaler observe the overflow and lower its scale. Its
                    # recorded inf check makes scaler.step skip this update.
                    overflow_steps += 1
                    scaler.step(optimizer)
                    scaler.update()
                    continue
                diverged = True
                optimizer.zero_grad(set_to_none=True)
                break
            grad_norms.append(grad_norm)
            if cfg["clip_norm"] is not None and grad_norm > float(cfg["clip_norm"]):
                clipped_steps += 1
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        if diverged:
            break

        train_metrics = evaluate(model, X_train_eval, y_train_eval, cfg["loss"])
        val_metrics = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])
        if not np.isfinite(val_metrics["loss"]):
            diverged = True
            break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_epoch = time.perf_counter() - epoch_start
        epoch_times.append(elapsed_epoch)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(float(np.mean(grad_norms)) if grad_norms else float("nan"))
        history["grad_norm_max"].append(float(np.max(grad_norms)) if grad_norms else float("nan"))
        history["clip_fraction"].append(
            float(clipped_steps / len(grad_norms)) if grad_norms else 0.0
        )
        history["overflow_steps"].append(int(overflow_steps))
        history["epoch_time_s"].append(elapsed_epoch)

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_metrics = val_metrics
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if epoch in {1, int(cfg["epochs"])}:
            print(f"{cfg['exp_id']:20s} epoch={epoch:02d} "
                  f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                  f"val_macro_f1={val_metrics['macro_f1']:.4f}")

    final_train_loss = history["train_loss"][-1] if history["train_loss"] else None
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else None
    peak_mem = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda" else 0.0)
    summary = {
        "step0_loss": float(step0["loss"]),
        "activation_std_step0": init_stats,
        "best_val_loss": best_loss if best_state is not None else None,
        "best_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "val_acc": best_metrics["acc"] if best_metrics else None,
        "val_macro_f1": best_metrics["macro_f1"] if best_metrics else None,
        "time_per_epoch_s": float(np.mean(epoch_times)) if epoch_times else None,
        "total_time_s": time.perf_counter() - total_start,
        "peak_mem_MB": float(peak_mem),
        "diverged": bool(diverged),
        "param_count": count_params(model),
    }
    result = {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }
    return result


def write_predictions(row_id, preds, path: str | Path) -> None:
    """Write the exact row_id,pred format consumed by scripts/evaluate.py."""
    row_id = np.asarray(row_id, dtype=np.int64)
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    preds = np.asarray(preds, dtype=np.int64)
    if row_id.ndim != 1 or preds.ndim != 1 or len(row_id) != len(preds):
        raise ValueError("row_id and predictions must be one-dimensional arrays of equal length")
    if len(np.unique(row_id)) != len(row_id):
        raise ValueError("row_id values must be unique")
    if ((preds < 0) | (preds > 6)).any():
        raise ValueError("predictions must be integer labels from 0 to 6")
    order = np.argsort(row_id, kind="stable")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "pred"])
        writer.writerows(zip(row_id[order].tolist(), preds[order].tolist()))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str | Path) -> None:
    """Restore the validation-selected checkpoint and write eval predictions only."""
    if result.get("best_state") is None:
        raise ValueError("the selected run has no finite validation checkpoint")
    cfg = {**DEFAULT_CFG, **cfg}
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=float(cfg["dropout"]),
                init=cfg["init"]).to(data["X_eval"].device)
    model.load_state_dict(result["best_state"])
    model.eval()
    predictions = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], predictions, pred_path)
