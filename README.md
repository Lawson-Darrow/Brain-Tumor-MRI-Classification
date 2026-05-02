# Brain Tumor MRI Classification

Four-class brain tumor MRI classification project using:

- Classical machine learning on handcrafted image features
- A custom convolutional neural network (CNN)
- Transfer learning with pretrained vision backbones

The goal is to compare modeling strategies under the same evaluation policy and report results using consistent metrics (accuracy, precision, recall, and macro-F1).

## Project Summary

This repository implements an end-to-end classification pipeline for the Kaggle dataset `masoudnickparvar/brain-tumor-mri-dataset`. The code covers:

1. Data loading and split preparation
2. Feature extraction for baseline models
3. Training and evaluation for baseline, CNN, and transfer approaches
4. Artifact packaging for final reporting and presentation

Classes:

- `glioma`
- `meningioma`
- `pituitary`
- `notumor`

## Dataset and Evaluation Policy

Expected local dataset layout:

```text
data/
  Training/
    glioma/
    meningioma/
    pituitary/
    notumor/
  Testing/
    glioma/
    meningioma/
    pituitary/
    notumor/
```

Evaluation policy used throughout the project:

- `Training/` is split into train and validation sets using stratified sampling.
- `Testing/` is treated as a strict held-out set for final evaluation only.
- Model and hyperparameter decisions are made using validation performance.

Dataset counts used in the final package:

- `Training`: 5600 images (1400 per class)
- `Testing`: 1600 images (400 per class)
- `Total`: 7200 images

The proposal text cited 7023 images; this mismatch is documented in `results/final_submission/canonical_split_report.json`.

## Methods

### 1) Baseline ML (Handcrafted Features)

- Grayscale conversion and resize to `128x128`
- HOG (Histogram of Oriented Gradients) feature extraction
- Optional PCA dimensionality reduction (default 200 components)
- Models:
  - Logistic Regression
  - RBF SVM
  - Random Forest

### 2) Custom CNN

- From-scratch CNN implemented in `src/cnn.py`
- Data augmentation in training pipeline:
  - Random horizontal flip
  - Random rotation
  - Mild color jitter
- Optimization with early stopping and learning-rate scheduling

### 3) Transfer Learning

- Backbones supported:
  - `resnet50`
  - `efficientnet_b0`
  - `vgg16`
- Two-stage training strategy:
  1. Freeze backbone, train classifier head
  2. Unfreeze last feature block(s), then fine-tune

## Final Results (Held-Out Test Set)

From `results/final_submission/comparison/final_model_comparison.csv`:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| `baseline::svm_rbf` | `0.9069` | `0.9047` |
| `transfer::resnet50_tuned` | `0.8438` | `0.8403` |
| `transfer::resnet50_original` | `0.7488` | `0.7405` |
| `custom_cnn` | `0.7375` | `0.7333` |

Key observations:

- Best overall model in this run is `baseline::svm_rbf`.
- Tuned transfer learning substantially improves over the original transfer run.
- `notumor` is generally the easiest class, while `glioma` remains comparatively harder.

## Repository Structure

```text
src/
  data.py                # dataset scanning, stratified split, dataloaders
  features.py            # HOG and PCA feature pipeline
  baselines.py           # logistic regression / SVM / random forest branch
  cnn.py                 # custom CNN architecture
  transfer.py            # transfer model builders and unfreeze logic
  train.py               # training loops, early stopping, prediction
  evaluate.py            # metrics + confusion matrix plotting
  main.py                # command-line entry point
  finalize_artifacts.py  # final package assembly + report assets

results/final_submission/
  canonical_split_report.json
  comparison/
  models/
  presentation_assets/
  run_manifest.json
```

## Setup

Recommended Python version: `3.10` to `3.12`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt`.

## How to Run

### Baselines Only

```bash
python -m src.main --dataset-root data --task baselines --output-dir results/real_run
```

### Custom CNN Only

```bash
python -m src.main --dataset-root data --task cnn --epochs 3 --batch-size 32 --output-dir results/real_run
```

### Transfer Learning Only

```bash
python -m src.main --dataset-root data --task transfer --epochs 3 --batch-size 32 --transfer-model resnet50 --output-dir results/real_run
```

### Run All Tracks

```bash
python -m src.main --dataset-root data --task all --epochs 3 --batch-size 32 --transfer-model resnet50 --output-dir results/real_run
```

## Build Final Submission Artifacts

```bash
python -m src.finalize_artifacts --dataset-root data --results-root results --output-root results/final_submission
```

This assembles consolidated outputs including:

- split report
- comparison tables
- model metrics and confusion matrices
- presentation-ready figures
- run manifest

## Colab Notebook Version

A notebook version of the project is included at:

- `Brain_Tumor_MRI_Classification.ipynb`

It mirrors the core pipeline in a single Colab workflow for easier course submission.

## Reproducibility Notes

- Global seeds are set for Python, NumPy, and PyTorch.
- Validation splitting is stratified and seeded.
- Run provenance is documented in `results/final_submission/run_manifest.json`.

## Acknowledgment

Dataset source: Kaggle, `masoudnickparvar/brain-tumor-mri-dataset`.
