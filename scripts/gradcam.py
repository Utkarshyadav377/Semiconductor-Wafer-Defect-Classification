"""Generate Grad-CAM figures: one correct example per class, plus misclassified examples."""
from __future__ import annotations

import numpy as np

from _common import base_parser, config_from_args, load_trained
from wafer_defect.engine import predict
from wafer_defect.gradcam import plot_gradcam_grid
from wafer_defect.pipeline import prepare_data
from wafer_defect.utils import get_device, seed_everything
from torch.utils.data import DataLoader


def main():
    ap = base_parser("Grad-CAM visualisations")
    ap.add_argument("--max-errors", type=int, default=6)
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = get_device()
    seed_everything(cfg.seed)
    data = prepare_data(cfg, verbose=False)
    models = load_trained(args.models, len(data.classes), cfg, device)
    out = cfg.out_path

    ref = list(models)[-1]  # pick examples using the last model's predictions
    preds, y, _ = predict(models[ref], DataLoader(data.test_ds, 128), device)

    correct = [int(np.where((y == k) & (preds == k))[0][0])
               for k in range(len(data.classes)) if ((y == k) & (preds == k)).any()]
    plot_gradcam_grid(models, data.test_ds, correct, data.classes,
                      f"Grad-CAM: correct predictions ({ref})", out / "gradcam_correct.png")

    wrong = [int(i) for i in np.where(preds != y)[0][: args.max_errors]]
    if wrong:
        plot_gradcam_grid(models, data.test_ds, wrong, data.classes,
                          f"Grad-CAM: misclassified ({ref})", out / "gradcam_errors.png")
    print(f"Saved figures to {out.resolve()}")


if __name__ == "__main__":
    main()
