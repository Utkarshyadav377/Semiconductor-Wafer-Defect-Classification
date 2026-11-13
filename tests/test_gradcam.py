import pytest
import torch

from wafer_defect.gradcam import GradCAM, get_target_layer
from wafer_defect.models import MODEL_NAMES, build_model


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_gradcam_output(name):
    model = build_model(name, 5).eval()
    cam = GradCAM(model, get_target_layer(name, model))
    heat, cls, probs = cam(torch.randn(3, 64, 64))
    cam.remove()
    assert heat.shape == (64, 64) and 0.0 <= heat.min() and heat.max() <= 1.0
    assert 0 <= cls < 5 and abs(float(probs.sum()) - 1.0) < 1e-4
