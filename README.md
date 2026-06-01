# Brain Tumor MRI Classification — Classical vs CNN vs Vision Transformer

Four-class brain-tumor MRI classification (`glioma`, `meningioma`, `pituitary`,
`notumor`) comparing classical models on handcrafted features, a from-scratch CNN,
and transfer learning with **ResNet50, EfficientNet-B0, and a Vision Transformer
(ViT-B/16)** — trained to convergence, reported across 3 seeds with ROC-AUC,
calibration error, and bootstrap confidence intervals.

The headline contribution is methodological: this dataset's official train/test
split **leaks**, and we measure and correct for it.

## Headline findings

1. **The official split leaks badly.** A perceptual-hash audit
   (`scripts/audit_leakage.py`) finds **23% of test images are exact duplicates of
   training images, and 63% have a near-duplicate in training.** Every published
   accuracy on this split — including this project's original course submission — is
   therefore optimistic.
2. **The leakage inflates scores by ~7–10 accuracy points.** Evaluating on a
   **deduplicated** test set (images with no near-twin in training) drops every model
   substantially: e.g. ViT-B/16 falls from **0.949 → 0.878** accuracy
   (macro-F1 0.948 → 0.832).
3. **At convergence, the Vision Transformer wins** (0.949 official / 0.878 clean),
   narrowly ahead of ResNet50, and both beat the classical SVM (0.906). This corrects
   an earlier under-trained result in which the SVM appeared to beat the CNNs.

## Results

Single training/validation split (group-aware so near-duplicates never straddle
train/val), evaluated on **both** the official test set (comparable to published
numbers) and the **deduplicated** test set (honest generalization). Deep models are
mean ± std over 3 seeds.

| Model | Official acc | Official macro-F1 | ROC-AUC (OvR) | ECE | **Clean acc** | **Clean macro-F1** |
|---|---:|---:|---:|---:|---:|---:|
| **ViT-B/16** | **0.949 ± 0.009** | **0.948** | **0.990** | 0.032 | **0.878** | **0.832** |
| ResNet50 | 0.940 ± 0.004 | 0.938 | 0.988 | 0.041 | 0.864 | 0.797 |
| EfficientNet-B0 | 0.838 ± 0.023 | 0.833 | 0.958 | 0.023 | 0.763 | 0.696 |
| Custom CNN | 0.833 ± 0.012 | 0.826 | 0.954 | 0.047 | 0.732 | 0.662 |
| SVM-RBF (HOG, classical) | 0.906 | 0.904 | — | — | — | — |

- **Leakage gap** (official − clean accuracy): ViT 0.071, ResNet50 0.076, EfficientNet
  0.075, CNN 0.101. The clean numbers are the ones to trust as generalization estimates.
- Strong pretrained backbones (ViT, ResNet50) clearly beat the classical SVM; the
  weaker deep models (EfficientNet-B0, the from-scratch CNN) do **not** — a fair nuance,
  not "deep always wins."
- Per-class F1 and confusion matrices are saved per run under `results/research_grade/`;
  `glioma`/`meningioma` are the harder, more-confused pair.

## The data-leakage audit (why two test columns)

The masoudnickparvar dataset is assembled from several public sources and contains many
near-duplicate MRI slices. `scripts/audit_leakage.py` computes a 64-bit perceptual hash
(dHash) for every image and reports duplicates that cross the train/test boundary:

| | Count | % of test |
|---|---:|---:|
| Test images with an **exact** duplicate in Training | 372 | 23.3% |
| Test images with a **near**-duplicate (Hamming ≤ 5) | 1004 | 62.8% |

`src/dedup_split.py` acts on this: it groups near-duplicate images (union-find over the
hash graph), builds a **group-aware** train/val split so duplicates can't leak into model
selection, and derives a **deduplicated test set** (569 of 1600 official test images whose
hash-group has no member in Training). We report both.

**Honest caveats:** the dataset ships no patient IDs, so true patient-level leakage cannot
be fully removed — dHash near-duplication is a proxy. One cross-class exact duplicate was
also found (a `glioma` test image identical to a `meningioma` training image), indicating
some label noise. Treat official-split numbers as benchmark-comparable, not as clinical
generalization, and note this is a research/education project, **not a diagnostic device.**

## Interpretability

Grad-CAM overlays for each model (CNN/ResNet/EfficientNet via the final conv stage,
ViT via a reshape transform on the last encoder block) are saved at
`results/research_grade/<model>/seed0/gradcam/`. Saliency is qualitative context for where
each model attends — it is **not** evidence of clinical reasoning.

## Methods

- **Classical:** grayscale + HOG features, optional PCA, then LogReg / RBF-SVM / RandomForest.
- **Custom CNN:** from-scratch conv net (`src/cnn.py`), trained with augmentation + early stopping.
- **Transfer (ResNet50 / EfficientNet-B0 / ViT-B/16):** two-stage — freeze backbone and train the
  head, then unfreeze the last block and fine-tune. 224×224, ImageNet normalization.
- **Evaluation:** validation (group-aware) for model selection; metrics on official + clean test;
  ROC-AUC (one-vs-rest macro), expected calibration error, bootstrap 95% CIs; 3 seeds (split fixed,
  training seed varied).

## Reproduce

```bash
# Dataset: masoudnickparvar/brain-tumor-mri-dataset (Training/ + Testing/ with 4 class folders)
python scripts/audit_leakage.py            # perceptual-hash leakage audit -> results/leakage_audit.json
python scripts/run_brain_experiments.py    # full matrix: 4 deep models x 3 seeds + baselines, dual eval
python scripts/gradcam_figures.py --model vit_b_16 \
    --weights results/research_grade/vit_b_16/seed0/model_seed0.pt
```

Environment: Python 3.12, PyTorch 2.6 (CUDA), `timm`, `pytorch-grad-cam`, scikit-learn. Trained on
a single RTX 4090. The original single-seed CLI pipeline (`python -m src.main ...`) still works.

## Original course submission (preserved)

This started as an MTH/CSE course project. Its original results
(`results/final_submission/`) were produced with deep models trained for only ~3 epochs, which
is why the classical SVM (0.906) appeared to beat an under-trained ResNet50 (0.844). Training to
convergence (above) reverses that and adds the ViT. The `Brain_Tumor_MRI_Classification.ipynb`
notebook mirrors the original single-pass pipeline.

## Repository structure

```text
src/
  data.py            # dataset scan, stratified split, dataloaders
  dedup_split.py     # leakage-aware group split + deduplicated test set  [new]
  features.py        # HOG + PCA pipeline
  baselines.py       # classical models
  cnn.py             # custom CNN
  transfer.py        # backbone builders incl. vit_b_16                    [updated]
  train.py           # training loop, predict, predict_proba              [updated]
  evaluate.py        # metrics + confusion matrices
  main.py            # original single-run CLI
scripts/
  audit_leakage.py           # perceptual-hash leakage audit              [new]
  run_brain_experiments.py   # research-grade matrix + dual eval          [new]
  gradcam_figures.py         # Grad-CAM interpretability                  [new]
results/
  leakage_audit.json
  research_grade/            # per-model metrics, weights, gradcam (gitignored bulk)
```

## Acknowledgment

Dataset: Kaggle, `masoudnickparvar/brain-tumor-mri-dataset`.
