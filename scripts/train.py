"""Train the custom CNN and/or ResNet-18, then evaluate on the held-out test split."""
from __future__ import annotations

from _common import base_parser, config_from_args
from wafer_defect.engine import train_model
from wafer_defect.evaluation import (evaluate_model, plot_confusion_matrices, plot_training_curves,
                                     robustness_table, save_reports)
from wafer_defect.models import build_model, count_parameters
from wafer_defect.pipeline import prepare_data
from wafer_defect.utils import get_device, seed_everything


def main():
    ap = base_parser("Train wafer defect classifiers")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    cfg = config_from_args(args)
    device = get_device()
    print(f"Device: {device}")
    seed_everything(cfg.seed)
    data = prepare_data(cfg)
    n = len(data.classes)
    out = cfg.out_path

    trained, histories = {}, {}
    for name in args.models:
        seed_everything(cfg.seed)
        model = build_model(name, n, cfg.resnet_pretrained if name == "resnet18" else False)
        print(f"\n=== {name} ({count_parameters(model):,} parameters) ===")
        model, hist = train_model(model, name, data.train_dl, data.val_dl,
                                  data.class_weights, cfg, device, out / "checkpoints")
        hist.to_csv(out / f"history_{name}.csv", index=False)
        trained[name], histories[name] = model, hist

    results = {name: evaluate_model(m, data.test_dl, data.classes, device) for name, m in trained.items()}
    for name, r in results.items():
        print(f"\n===== {name} =====\n{r['report']}")
    print(save_reports(results, out).to_string())
    plot_training_curves(histories, out / "training_curves.png")
    plot_confusion_matrices(results, data.classes, out / "confusion_matrices.png")

    rob = robustness_table(trained, data.test_df, cfg.img_size, device)
    rob.to_csv(out / "robustness_f1.csv")
    print("\nRotation/flip robustness (macro-F1):\n", rob.to_string())
    print(f"\nArtifacts written to {out.resolve()}")


if __name__ == "__main__":
    main()
