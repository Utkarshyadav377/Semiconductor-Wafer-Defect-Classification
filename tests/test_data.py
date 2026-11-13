import numpy as np
import torch

from wafer_defect.data import (WaferDataset, build_dataframe, clean_dataframe, compute_class_weights,
                               find_dataset_root, split_dataframe)


def test_find_and_build(toy_dataset):
    root = find_dataset_root(str(toy_dataset.parent))
    df, classes = build_dataframe(root)
    assert classes == ["Center", "Edge Ring", "none"]
    assert len(df) == 72 and set(df.y) == {0, 1, 2}


def test_drop_classes(toy_dataset):
    df, classes = build_dataframe(toy_dataset, drop_classes=["none"])
    assert classes == ["Center", "Edge Ring"] and len(df) == 48


def test_clean_removes_duplicates(toy_dataset):
    import shutil
    shutil.copy(toy_dataset / "Center" / "0.png", toy_dataset / "Center" / "dup.png")
    df, _ = build_dataframe(toy_dataset)
    cleaned, rep = clean_dataframe(df)
    assert rep["duplicates"] == 1 and len(cleaned) == 72


def test_split_is_stratified_and_disjoint(toy_dataset):
    df, _ = build_dataframe(toy_dataset)
    tr, va, te = split_dataframe(df, 0.15, 0.15, seed=1)
    assert not (set(tr.path) & set(va.path)) and not (set(tr.path) & set(te.path)) and not (set(va.path) & set(te.path))
    assert len(tr) + len(va) + len(te) == len(df)
    for part in (tr, va, te):
        assert set(part.y) == {0, 1, 2}
    # deterministic
    tr2, _, _ = split_dataframe(df, 0.15, 0.15, seed=1)
    assert list(tr.path) == list(tr2.path)


def test_class_weights():
    y = np.array([0] * 90 + [1] * 10)
    w = compute_class_weights(y, 2)
    assert w[1] > w[0]
    assert abs(float((w[torch.tensor(y)]).mean()) - 1.0) < 1e-5  # sample-weighted mean is 1
    assert torch.allclose(compute_class_weights(np.array([0, 1, 2] * 5), 3), torch.ones(3))


def test_dataset_shapes_and_augmentation(toy_dataset):
    df, _ = build_dataframe(toy_dataset)
    ds = WaferDataset(df, img_size=64, train=True)
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and x.min() >= -1.01 and x.max() <= 1.01 and y.dtype == torch.long
