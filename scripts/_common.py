"""Small helpers shared by the CLI scripts."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from wafer_defect.config import Config, load_config
from wafer_defect.models import MODEL_NAMES, build_model


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data-root", default=None, help="Folder containing WM811k_Dataset")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    return p


def config_from_args(args) -> Config:
    overrides = dict(data_root=args.data_root, out_dir=args.out_dir, seed=args.seed)
    for extra in ("epochs", "batch_size", "lr"):
        overrides[extra] = getattr(args, extra, None)
    return load_config(args.config, **overrides)


def load_trained(names, n_classes: int, cfg: Config, device) -> dict:
    """Load best checkpoints saved by scripts/train.py."""
    models = {}
    for name in names:
        ckpt = Path(cfg.out_dir) / "checkpoints" / f"{name}_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} not found. Run scripts/train.py first.")
        m = build_model(name, n_classes, pretrained=False if name == "custom_cnn" else cfg.resnet_pretrained)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        models[name] = m.to(device).eval()
    return models
