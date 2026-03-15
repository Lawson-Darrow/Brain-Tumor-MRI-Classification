# Brain Tumor MRI Classification

This project compares three approaches for four-class brain tumor MRI classification on Kaggle's `masoudnickparvar/brain-tumor-mri-dataset`:

- Baseline ML on extracted features (Logistic Regression, SVM, Random Forest)
- Custom CNN trained from scratch
- Transfer learning (`resnet50`, `efficientnet_b0`, or `vgg16`)

## Dataset Policy

Evaluation policy:

- `Training/` is split into train/validation
- `Testing/` is held out for final test metrics only
- Hyperparameter decisions are made on validation, not on `Testing/`

Expected local layout:

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

Current local dataset snapshot used for final artifacts is balanced:

- `Training`: 5600 images (1400/class)
- `Testing`: 1600 images (400/class)
- `Total`: 7200 images

The proposal text cites 7023 images; the final package documents this mismatch in `results/final_submission/canonical_split_report.json`.

## Setup

Recommended Python version: `3.10` to `3.12`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository Contents

This public repository contains:

- `src/` pipeline code for data loading, baselines, CNN training, transfer learning, evaluation, and artifact packaging
- `requirements.txt` for the Python environment
- `results/final_submission/` with the final comparison tables, selected model metrics, confusion matrices, and presentation figures

This repository does not include the raw MRI dataset. Place the Kaggle dataset under `data/Training` and `data/Testing` locally before running the pipeline.

## Main Run Commands

Run baselines only:

```bash
python -m src.main --dataset-root data --task baselines --output-dir results/real_run
```

Run custom CNN only:

```bash
python -m src.main --dataset-root data --task cnn --epochs 3 --batch-size 32 --output-dir results/real_run
```

Run transfer only:

```bash
python -m src.main --dataset-root data --task transfer --epochs 3 --batch-size 32 --transfer-model resnet50 --output-dir results/real_run
```

Run all three tracks (reference command used for official run):

```bash
python -m src.main --dataset-root data --task all --epochs 3 --batch-size 32 --transfer-model resnet50 --output-dir results/real_run
```

## Final Submission Package

Build the consolidated, presentation-ready package:

```bash
python -m src.finalize_artifacts --dataset-root data --results-root results --output-root results/final_submission
```

This creates:

- `results/final_submission/canonical_split_report.json`
- `results/final_submission/comparison/final_model_comparison.csv`
- `results/final_submission/models/...` (selected model bundles and confusion matrices)
- `results/final_submission/presentation_assets/...` (plots and image gallery)
- `results/final_submission/key_findings.md`
- `results/final_submission/run_manifest.json`

## Final Result Summary

From the official final comparison (`results/final_submission/comparison/final_model_comparison.csv`):

- `baseline::svm_rbf`: accuracy `0.9069`, macro-F1 `0.9047` (best overall)
- `custom_cnn`: accuracy `0.7375`, macro-F1 `0.7333`
- `transfer::resnet50_original`: accuracy `0.7488`, macro-F1 `0.7405`
- `transfer::resnet50_tuned`: accuracy `0.8438`, macro-F1 `0.8403` (best deep model)

## Reproducibility

- Fixed seed is used for Python, NumPy, and PyTorch
- Validation is stratified from `Training/` only
- Official run details and commands are recorded in `results/final_submission/run_manifest.json`
