# Semiconductor Wafer Defect Classification (WM-811K)

Deep-learning pipeline that classifies semiconductor wafer-map defect patterns, comparing a **custom CNN** with **ResNet-18**, using class-weighted loss, label-invariant geometric augmentation, full metric reporting and **Grad-CAM** interpretability.

> Status: training code complete; the Streamlit app is deferred. Results below are placeholders until you run training.

## Why this problem
Wafer-map defect patterns (center, donut, edge-ring, scratch, ...) point to specific process faults. Manual labelling by engineers is slow and inconsistent, so automatic classification speeds up root-cause analysis and protects yield.

## Dataset
WM-811K wafer maps (MIR Lab, NTU). The export used here has 9 class folders (Center, Donut, Edge Local, Edge Ring, Local, Scratch, near full, none, random), about 100 32x32 RGB JPEGs per class (~900 images). It is nearly balanced, so class weights have little effect on this subset. The weighted loss is implemented so the code stays valid for the full, heavily imbalanced WM-811K. See [`data/README.md`](data/README.md). The dataset is not committed.

## Method
| Step | Implementation |
|---|---|
| Cleaning | drop unreadable files and exact duplicates (`data.clean_dataframe`) |
| Preprocessing | resize to 64x64 (nearest), scale to [-1, 1] |
| Split | stratified 70/15/15, seeded and reproducible |
| Imbalance | inverse-frequency class weights in cross-entropy |
| Augmentation | exact 90-degree rotations, flips, light noise (a wafer's label does not depend on orientation) |
| Models | custom 4-block CNN (BN, GAP head) and ResNet-18 (small-image stem) |
| Training | AdamW, one-cycle LR, label smoothing, AMP on GPU, early stopping on validation macro-F1 |
| Evaluation | accuracy, macro precision/recall/F1, per-class report, confusion matrices, rotation/flip robustness |
| Interpretability | Grad-CAM on the last conv stage of each model |

## Project structure
```
wafer-defect-classification/
├── configs/default.yaml          # all hyper-parameters
├── data/README.md               # where to put the dataset (data itself is git-ignored)
├── notebooks/
│   └── 01_standalone_kaggle_notebook.ipynb   # single-file version for quick Kaggle runs
├── scripts/
│   ├── train.py                 # train + evaluate + plots
│   ├── evaluate.py              # evaluate saved checkpoints
│   ├── gradcam.py               # Grad-CAM figures
│   └── visualize_data.py        # sample grid + class distribution
├── src/wafer_defect/
│   ├── config.py  data.py  models.py  engine.py
│   ├── evaluation.py  gradcam.py  pipeline.py  utils.py
├── tests/                       # pytest unit tests
├── outputs/                     # generated artifacts (git-ignored)
├── .github/workflows/ci.yml     # lint + tests
├── Makefile  pyproject.toml  requirements*.txt  LICENSE
```

## Setup
```bash
git clone https://github.com/Utkarshyadav377/Semiconductor-Wafer-Defect-Classification.git
cd wafer-defect-classification
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements-dev.txt
```

## Usage
```bash
python scripts/visualize_data.py --data-root data
python scripts/train.py --data-root data                 # both models
python scripts/train.py --models resnet18 --epochs 40    # one model
python scripts/evaluate.py --data-root data
python scripts/gradcam.py --data-root data
pytest -q
```
Outputs (checkpoints, `test_summary.csv`, `classification_reports.txt`, `confusion_matrices.png`, `training_curves.png`, `robustness_f1.csv`, Grad-CAM figures) are written to `outputs/`.

### Running on Kaggle
1. Upload the dataset zip as a Kaggle dataset and attach it to a notebook; enable the GPU.
2. In the notebook:
```python
!git clone https://github.com/Utkarshyadav377/Semiconductor-Wafer-Defect-Classification.git
%cd wafer-defect-classification
!pip install -q -e .
!python scripts/train.py --out-dir /kaggle/working/outputs
!python scripts/gradcam.py --out-dir /kaggle/working/outputs
```
The dataset folder is auto-detected under `/kaggle/input`. Or skip the clone and use `notebooks/01_standalone_kaggle_notebook.ipynb`, which contains the whole pipeline in one file.

## Results
Fill in after training (`outputs/test_summary.csv`):

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| Custom CNN | | | | |
| ResNet-18 | | | | |

With about 135 test images, one image is roughly 0.7% accuracy. Report mean and standard deviation over several seeds (`--seed`) before claiming one model is better.

## Limitations and next steps
- Small, near-balanced JPEG subset: JPEG compression blurs the discrete die states, and conclusions about class imbalance need the full WM-811K (`LSWMD.pkl`).
- "near full" and "none" are kept; the reference paper excludes them. Use `drop_classes: ["near full", "none"]` to compare.
- Possible extension: Radon-transform features plus a kernel-flip module for built-in rotation/flip invariance (Jeong et al., 2023).
- Streamlit app (upload a wafer map, show prediction, confidence and Grad-CAM) is planned.

## Reference
Jeong, I., Lee, S., Park, K., Kim, I., Huh, H. & Lee, S. *Wafer map failure pattern classification using geometric transformation-invariant convolutional neural network.* Scientific Reports 13, 8127 (2023). https://doi.org/10.1038/s41598-023-34147-2 (CC BY 4.0)

## License
MIT. See [LICENSE](LICENSE).
