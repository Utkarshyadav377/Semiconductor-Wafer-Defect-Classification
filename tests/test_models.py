import pytest
import torch

from wafer_defect.models import MODEL_NAMES, build_model, count_parameters


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_shape(name):
    m = build_model(name, n_classes=9).eval()
    out = m(torch.randn(4, 3, 64, 64))
    assert out.shape == (4, 9)
    assert count_parameters(m) > 0


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("vgg", 9)
