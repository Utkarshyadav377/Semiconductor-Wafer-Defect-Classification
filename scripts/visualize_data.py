"""Plot a sample grid and class distribution (no training required)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from _common import base_parser, config_from_args
from wafer_defect.data import build_dataframe, clean_dataframe, find_dataset_root


def main():
    args = base_parser("Visualise the dataset").parse_args()
    cfg = config_from_args(args)
    df, classes = build_dataframe(find_dataset_root(cfg.data_root), cfg.drop_classes)
    df, report = clean_dataframe(df)
    print(report)
    out = cfg.out_path

    fig, axes = plt.subplots(len(classes), 8, figsize=(12, 1.5 * len(classes)), squeeze=False)
    for r, c in enumerate(classes):
        for k, p in enumerate(df[df.label == c].sample(8, random_state=0).path):
            axes[r, k].imshow(Image.open(p))
            axes[r, k].axis("off")
            if k == 0:
                axes[r, k].set_title(c, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=150)

    plt.figure(figsize=(7, 3))
    sns.countplot(data=df, y="label", order=classes)
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=150)
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
