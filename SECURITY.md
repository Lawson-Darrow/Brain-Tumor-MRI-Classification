# Security Policy

## Supported versions

This is a research and education project. Only the latest `main` is supported;
there are no backported fixes.

## Reporting a vulnerability

Please report security issues privately via
[GitHub Security Advisories](https://github.com/Lawson-Darrow/Brain-Tumor-MRI-Classification/security/advisories/new)
rather than a public issue. We will acknowledge and respond as soon as we can.

## Scope notes

Everything here runs locally: the pipeline trains and evaluates on a dataset you
download yourself (Kaggle, `masoudnickparvar/brain-tumor-mri-dataset`) and sends
nothing to external services. Trained weights and raw data are not committed.

This is a research and education project for studying classification and
evaluation methodology. It is not a medical device and must not be used for
diagnosis or patient care.
