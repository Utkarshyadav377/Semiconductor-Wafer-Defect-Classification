"""Evaluate saved checkpoints on the test split (same seed => same split as training)."""
from __future__ import annotations

from _common import base_parser, config_from_args, load_trained
from wafer_defect.evaluation import (evaluate_model, plot_confusion_matrices, robustness_table,
                                     save_reports)
from wafer_defect.pipeline import prepare_data
from wafer_defect.utils import get_device, seed_everything


def main():
    args = base_parser("Evaluate trained wafer defect classifiers").parse_args()
    cfg = config_from_args(args)
    device = get_device()
    seed_everything(cfg.seed)
    data = prepare_data(cfg)
    models = load_trained(args.models, len(data.classes), cfg, device)
    out = cfg.out_path

    results = {n: evaluate_model(m, data.test_dl, data.classes, device) for n, m in models.items()}
    for n, r in results.items():
        print(f"\n===== {n} =====\n{r['report']}")
    print(save_reports(results, out).to_string())
    plot_confusion_matrices(results, data.classes, out / "confusion_matrices.png")
    rob = robustness_table(models, data.test_df, cfg.img_size, device)
    rob.to_csv(out / "robustness_f1.csv")
    print(rob.to_string())


if __name__ == "__main__":
    main()
