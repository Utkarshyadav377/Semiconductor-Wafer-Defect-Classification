from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader

from .data import WaferDataset
from .engine import macro_f1, predict

matplotlib.rcParams["figure.dpi"] = 110

ROTFLIP_TRANSFORMS = {
    "original": lambda x: x,
    "rot90": lambda x: torch.rot90(x, 1, (1, 2)),
    "rot180": lambda x: torch.rot90(x, 2, (1, 2)),
    "rot270": lambda x: torch.rot90(x, 3, (1, 2)),
    "flip_h": lambda x: x.flip(2),
    "flip_v": lambda x: x.flip(1),
}


def evaluate_model(model, dl: DataLoader, classes: List[str], device) -> Dict:
    p, y, _ = predict(model, dl, device)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(
        preds=p, labels=y,
        metrics=dict(accuracy=accuracy_score(y, p), precision_macro=pr, recall_macro=rc, f1_macro=f1),
        report=classification_report(y, p, target_names=classes, digits=3, zero_division=0),
        confusion=confusion_matrix(y, p, labels=range(len(classes))),
    )


def plot_confusion_matrices(results: Dict[str, Dict], classes: List[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6), squeeze=False)
    for ax, (name, r) in zip(axes[0], results.items()):
        sns.heatmap(r["confusion"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(f'{name} (acc {r["metrics"]["accuracy"]:.3f})')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(histories: Dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for name, h in histories.items():
        ax[0].plot(h.epoch, h.train_loss, label=f"{name} train")
        ax[0].plot(h.epoch, h.val_loss, "--", label=f"{name} val")
        ax[1].plot(h.epoch, h.train_acc, label=f"{name} train")
        ax[1].plot(h.epoch, h.val_acc, "--", label=f"{name} val")
        ax[2].plot(h.epoch, h.val_f1, label=name)
    for a, t in zip(ax, ["Loss", "Accuracy", "Validation macro-F1"]):
        a.set_title(t)
        a.set_xlabel("epoch")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def robustness_table(models: Dict[str, torch.nn.Module], test_df: pd.DataFrame, img_size: int,
                     device, batch_size: int = 128) -> pd.DataFrame:
    """Macro-F1 on the test set under fixed rotations/flips (out-of-distribution check)."""
    table = {}
    for name, m in models.items():
        table[name] = {}
        for tname, tf in ROTFLIP_TRANSFORMS.items():
            ds = WaferDataset(test_df, img_size, train=False, transform=tf)
            p, y, _ = predict(m, DataLoader(ds, batch_size), device)
            table[name][tname] = macro_f1(y, p)
    return pd.DataFrame(table).round(4)


def save_reports(results: Dict[str, Dict], out_dir: Path) -> pd.DataFrame:
    summary = pd.DataFrame({k: v["metrics"] for k, v in results.items()}).T.round(4)
    summary.to_csv(out_dir / "test_summary.csv")
    with open(out_dir / "classification_reports.txt", "w") as f:
        for name, r in results.items():
            f.write(f"===== {name} =====\n{r['report']}\n")
    return summary
