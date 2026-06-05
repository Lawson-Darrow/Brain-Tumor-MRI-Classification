# Contributing

This is a research and education project. Issues, ideas, and PRs are welcome,
especially around the leakage audit, additional backbones, and the evaluation
protocol.

## Dev setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -r requirements.txt
```

Download the dataset (`masoudnickparvar/brain-tumor-mri-dataset`) into the layout
the scripts expect, then:

```bash
python scripts/audit_leakage.py            # perceptual-hash leakage audit
python scripts/run_brain_experiments.py    # full matrix: 4 deep models x 3 seeds + baselines, dual eval
```

A single RTX 4090 (or any CUDA GPU) is assumed for the deep models; Python 3.12,
PyTorch 2.6.

## Bar for changes

- **Do not break the leakage controls.** The train/val split is group-aware
  (near-duplicates never straddle the split) and the deduplicated test set is
  derived from the perceptual-hash graph. Changes here need to preserve that.
- **Report both test columns.** Every headline gets the official-split number
  (benchmark-comparable) and the deduplicated-test number (honest generalization).
  Do not drop the clean column.
- **Keep the seeds and CIs.** Deep models are reported as mean over 3 seeds with
  bootstrap confidence intervals.
- Keep the framing honest: this is research and education, not a diagnostic device,
  and saliency maps are qualitative context, not clinical reasoning.

## Scope

New backbones, a stronger leakage proxy, or additional interpretability are all
welcome. The original single-pass notebook is preserved on purpose; leave it intact.
