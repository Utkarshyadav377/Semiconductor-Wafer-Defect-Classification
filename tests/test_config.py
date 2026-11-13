import pytest

from wafer_defect.config import load_config


def test_defaults_and_overrides(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("epochs: 5\nseed: 7\n")
    cfg = load_config(str(f), epochs=None, seed=9)
    assert cfg.epochs == 5 and cfg.seed == 9 and cfg.img_size == 64


def test_unknown_key(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("bogus: 1\n")
    with pytest.raises(ValueError):
        load_config(str(f))
