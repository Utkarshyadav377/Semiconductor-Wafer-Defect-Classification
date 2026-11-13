from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from .config import Config


def macro_f1(y_true, y_pred) -> float:
    return precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]


@torch.no_grad()
def predict(model: nn.Module, dl: DataLoader, device: torch.device):
    """Return (predictions, labels, logits) as numpy / cpu tensors."""
    model.eval()
    preds, ys, logits = [], [], []
    for x, y in dl:
        out = model(x.to(device))
        preds.append(out.argmax(1).cpu())
        ys.append(y)
        logits.append(out.cpu())
    return torch.cat(preds).numpy(), torch.cat(ys).numpy(), torch.cat(logits)


def train_model(model: nn.Module, name: str, train_dl: DataLoader, val_dl: DataLoader,
                class_weights: torch.Tensor, cfg: Config, device: torch.device,
                ckpt_dir: Path) -> Tuple[nn.Module, pd.DataFrame]:
    """Weighted cross-entropy, AdamW + one-cycle LR, AMP on CUDA, early stopping on val macro-F1."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda"
    model.to(device)
    cw = class_weights.to(device)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=len(train_dl)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_f1, best_state, bad, t0 = [], -1.0, None, 0, time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        tl = tc = tn = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
            tn += len(y)

        p, yv, lv = predict(model, val_dl, device)
        vloss = F.cross_entropy(lv, torch.tensor(yv), weight=class_weights.cpu()).item()
        vacc, vf1 = accuracy_score(yv, p), macro_f1(yv, p)
        history.append(dict(epoch=ep, train_loss=tl / tn, train_acc=tc / tn,
                            val_loss=vloss, val_acc=vacc, val_f1=vf1))

        if vf1 > best_f1:
            best_f1, bad, best_state = vf1, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if ep == 1 or ep % 5 == 0:
            print(f"[{name}] ep {ep:3d} | train loss {tl/tn:.3f} acc {tc/tn:.3f} "
                  f"| val loss {vloss:.3f} acc {vacc:.3f} F1 {vf1:.3f}")
        if bad >= cfg.patience:
            print(f"[{name}] early stop at epoch {ep}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, ckpt_dir / f"{name}_best.pt")
    print(f"[{name}] best val macro-F1 {best_f1:.4f} ({time.time() - t0:.0f}s)")
    return model, pd.DataFrame(history)
