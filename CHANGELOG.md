# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses
[SemVer](https://semver.org/) (pre-1.0: minor = breaking allowed).

## [Unreleased]

## [0.1.0] - 2026-06-05

Research-grade release. Four-class brain-tumor MRI classification comparing
classical models, a from-scratch CNN, and transfer learning (ResNet50,
EfficientNet-B0, ViT-B/16), with a data-leakage audit.

### Added
- Perceptual-hash (dHash) leakage audit: 23% of official-test images are exact
  duplicates of training images and 63% have a near-duplicate.
- Group-aware train/val split plus a derived deduplicated test set (union-find
  over the hash graph) so near-duplicates cannot leak into model selection.
- Dual evaluation: every model scored on both the official test set
  (benchmark-comparable) and the deduplicated test set (honest generalization).
- Deep models trained to convergence, reported as mean over 3 seeds with ROC-AUC,
  expected calibration error, and bootstrap 95% CIs.
- Grad-CAM interpretability overlays per model.

### Findings
- The official split leaks; the leakage inflates accuracy by about 7 to 10 points
  (ViT-B/16 0.949 official to 0.878 clean).
- At convergence the Vision Transformer wins, ahead of ResNet50, and both beat the
  classical SVM. This reverses the original under-trained course result in which
  the SVM appeared to beat the CNNs.

### Preserved
- The original MTH/CSE course submission (`results/final_submission/`) and the
  single-pass notebook are kept for comparison.
