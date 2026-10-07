"""Training loop with best-validation-loss checkpointing and early stopping."""
import random
import time

import numpy as np
import torch
import torch.nn as nn

from .models import trained_state, load_trained_state


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, 100 * correct / n


def fit(model, train_loader, val_loader, device, groups, epochs, weight_decay=0.0,
        label_smoothing=0.0, patience=None, name="model", verbose=True):
    """Train, keep the epoch with the lowest validation loss, and restore it.

    Validation *loss* picks the checkpoint because accuracy on ~200 images moves in coarse 0.5% steps.
    The validation loss always uses plain cross-entropy, so different setups stay comparable.
    """
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    val_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best = {"val_loss": float("inf")}
    start, stale = time.time(), 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, train_loss_fn, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, val_loss_fn, device)
        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[k].append(v)
        if va_loss < best["val_loss"]:
            best = {"val_loss": va_loss, "val_acc": va_acc, "epoch": epoch, "state": trained_state(model)}
            stale = 0
        else:
            stale += 1
        if verbose:
            print(f"[{name}] epoch {epoch}/{epochs}  train {tr_acc:5.1f}% / {tr_loss:.3f}   val {va_acc:5.1f}% / {va_loss:.3f}")
        if patience is not None and stale >= patience:
            break
    load_trained_state(model, best.pop("state"))
    best["seconds"] = time.time() - start
    if verbose:
        print(f"[{name}] best epoch {best['epoch']}: val {best['val_acc']:.1f}% / {best['val_loss']:.3f}  ({best['seconds']:.0f}s)")
    return history, best


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, probs = [], []
    for x, y in loader:
        probs.append(model(x.to(device)).softmax(1).cpu())
        y_true.extend(y.tolist())
    probs = torch.cat(probs).numpy()
    return np.array(y_true), probs.argmax(1), probs
