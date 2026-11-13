import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def toy_dataset(tmp_path):
    """3 classes x 24 random 32x32 RGB images in class folders."""
    root = tmp_path / "WM811k_Dataset"
    rng = np.random.default_rng(0)
    for cls in ["Center", "Edge Ring", "none"]:
        (root / cls).mkdir(parents=True)
        for i in range(24):
            arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(root / cls / f"{i}.png")
    return root
