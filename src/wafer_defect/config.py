from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class Config:
    data_root: Optional[str] = None
    out_dir: str = "outputs"
    img_size: int = 64
    drop_classes: List[str] = field(default_factory=list)
    val_frac: float = 0.15
    test_frac: float = 0.15
    batch_size: int = 64
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    patience: int = 12
    num_workers: int = 2
    seed: int = 42
    augment: bool = True
    resnet_pretrained: bool = False

    @property
    def out_path(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | None = None, **overrides) -> Config:
    """Load a YAML config; keyword overrides (when not None) take precedence."""
    data = {}
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**data)
