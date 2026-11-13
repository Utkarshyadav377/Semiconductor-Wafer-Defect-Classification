from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import WaferDataset, denormalize


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) for a chosen convolutional layer."""

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        self._handles = [
            layer.register_forward_hook(lambda m, i, o: setattr(self, "activations", o)),
            layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "gradients", go[0])),
        ]

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def __call__(self, x: torch.Tensor, target: int | None = None):
        """x: normalised CxHxW tensor. Returns (cam in [0,1] HxW, predicted class, probabilities)."""
        device = next(self.model.parameters()).device
        x = x.unsqueeze(0).to(device).requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            target = logits.argmax(1).item() if target is None else target
            self.model.zero_grad()
            logits[0, target].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = F.softmax(logits, 1)[0].detach().cpu()
        return cam.detach().cpu().numpy(), target, probs


def get_target_layer(name: str, model: nn.Module) -> nn.Module:
    """Last convolutional stage of each architecture."""
    if name == "custom_cnn":
        return [m for m in model.features if isinstance(m, nn.ReLU)][-1]
    if name == "resnet18":
        return model.layer4[-1]
    raise ValueError(name)


def plot_gradcam_grid(models: Dict[str, nn.Module], dataset: WaferDataset, indices: List[int],
                      classes: List[str], title: str, path: Path) -> None:
    cams = {n: GradCAM(m, get_target_layer(n, m)) for n, m in models.items()}
    ncols = 1 + 2 * len(cams)
    fig, axes = plt.subplots(len(indices), ncols, figsize=(2.2 * ncols, 2.3 * len(indices)), squeeze=False)
    for r, i in enumerate(indices):
        x, y = dataset[i]
        img = denormalize(x).permute(1, 2, 0).numpy()
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"True: {classes[int(y)]}", fontsize=8)
        for k, (name, g) in enumerate(cams.items()):
            cam, c, probs = g(x)
            axes[r, 1 + 2 * k].imshow(cam, cmap="jet")
            axes[r, 1 + 2 * k].set_title(f"{name}\nGrad-CAM", fontsize=8)
            axes[r, 2 + 2 * k].imshow(img)
            axes[r, 2 + 2 * k].imshow(cam, cmap="jet", alpha=0.5)
            axes[r, 2 + 2 * k].set_title(f"pred {classes[c]} ({probs[c]:.2f})", fontsize=8)
        for a in axes[r]:
            a.axis("off")
    for g in cams.values():
        g.remove()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
