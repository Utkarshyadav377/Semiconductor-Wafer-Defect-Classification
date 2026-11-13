"""Shared data preparation used by all scripts, so splits are identical across train/eval/Grad-CAM."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import (WaferDataset, build_dataframe, clean_dataframe, compute_class_weights,
                   find_dataset_root, split_dataframe)


@dataclass
class DataBundle:
    classes: List[str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_ds: WaferDataset
    val_ds: WaferDataset
    test_ds: WaferDataset
    train_dl: DataLoader
    val_dl: DataLoader
    test_dl: DataLoader
    class_weights: torch.Tensor


def prepare_data(cfg: Config, verbose: bool = True) -> DataBundle:
    root = find_dataset_root(cfg.data_root)
    df, classes = build_dataframe(root, cfg.drop_classes)
    df, report = clean_dataframe(df)
    if verbose:
        print(f"Dataset root: {root}")
        print(f"Cleaning: {report}")
        print(df.label.value_counts().to_string())

    train_df, val_df, test_df = split_dataframe(df, cfg.val_frac, cfg.test_frac, cfg.seed)
    if verbose:
        print(f"Split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_ds = WaferDataset(train_df, cfg.img_size, train=cfg.augment)
    val_ds = WaferDataset(val_df, cfg.img_size)
    test_ds = WaferDataset(test_df, cfg.img_size)
    pin = torch.cuda.is_available()
    mk = lambda ds, shuffle: DataLoader(ds, cfg.batch_size, shuffle=shuffle,
                                        num_workers=cfg.num_workers, pin_memory=pin)

    out = cfg.out_path
    (out / "classes.json").write_text(json.dumps(classes, indent=2))
    test_df.to_csv(out / "test_split.csv", index=False)

    return DataBundle(
        classes=classes, train_df=train_df, val_df=val_df, test_df=test_df,
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        train_dl=mk(train_ds, True), val_dl=mk(val_ds, False), test_dl=mk(test_ds, False),
        class_weights=compute_class_weights(train_df.y.values, len(classes)),
    )
