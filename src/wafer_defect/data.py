from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

MEAN, STD = 0.5, 0.5  # scale to [-1, 1]; wafer maps have no natural ImageNet statistics
IMG_EXT = (".jpg", ".jpeg", ".png")
DATASET_FOLDER = "WM811k_Dataset"


def find_dataset_root(data_root: Optional[str] = None) -> Path:
    """Locate the class-per-folder dataset directory."""
    bases = [data_root] if data_root else []
    bases += ["/kaggle/input", "data", "."]
    for base in bases:
        if base and os.path.isdir(base):
            for dirpath, _, _ in os.walk(base):
                if os.path.basename(dirpath) == DATASET_FOLDER:
                    return Path(dirpath)
    if data_root and os.path.isdir(data_root):
        return Path(data_root)  # user pointed directly at the class folders
    raise FileNotFoundError(
        f"'{DATASET_FOLDER}' not found. Pass --data-root or place it under ./data."
    )


def build_dataframe(root: Path, drop_classes: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Return a DataFrame(path, label, y) and the sorted class list."""
    rows = []
    for cls in sorted(os.listdir(root)):
        d = Path(root) / cls
        if not d.is_dir():
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMG_EXT):
                rows.append((str(d / f), cls))
    df = pd.DataFrame(rows, columns=["path", "label"])
    if drop_classes:
        df = df[~df.label.isin(drop_classes)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No images found under {root}")
    classes = sorted(df.label.unique())
    df["y"] = df.label.map({c: i for i, c in enumerate(classes)})
    return df, classes


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop unreadable files and exact duplicate images."""
    bad, seen, dup = [], {}, []
    for p in df.path:
        try:
            with Image.open(p) as im:
                im.load()
                h = hashlib.md5(np.asarray(im).tobytes()).hexdigest()
        except Exception:
            bad.append(p)
            continue
        if h in seen:
            dup.append(p)
        else:
            seen[h] = p
    out = df[~df.path.isin(bad + dup)].reset_index(drop=True)
    return out, {"unreadable": len(bad), "duplicates": len(dup), "kept": len(out)}


def split_dataframe(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int):
    """Stratified train/val/test split (deterministic for a given seed)."""
    train_df, tmp = train_test_split(df, test_size=val_frac + test_frac, stratify=df.y, random_state=seed)
    val_df, test_df = train_test_split(
        tmp, test_size=test_frac / (val_frac + test_frac), stratify=tmp.y, random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_tensor(path: str, img_size: int) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((img_size, img_size), Image.NEAREST)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div(255)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Label-invariant geometric augmentation: exact rotations/flips plus light noise."""
    x = torch.rot90(x, random.randint(0, 3), dims=(1, 2))
    if random.random() < 0.5:
        x = x.flip(2)
    if random.random() < 0.5:
        x = x.flip(1)
    if random.random() < 0.3:
        x = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)
    return x


class WaferDataset(Dataset):
    """In-memory dataset (the images are tiny). Returns normalised 3xHxW tensors."""

    def __init__(self, frame: pd.DataFrame, img_size: int = 64, train: bool = False,
                 transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.x = torch.stack([load_tensor(p, img_size) for p in frame.path])
        self.y = torch.tensor(frame.y.values, dtype=torch.long)
        self.train, self.transform = train, transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = self.x[i]
        if self.train:
            x = augment(x)
        if self.transform is not None:
            x = self.transform(x)
        return (x - MEAN) / STD, self.y[i]


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights (sample-weighted mean = 1). Uniform on balanced data."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    return torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return x * STD + MEAN
