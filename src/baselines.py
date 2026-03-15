from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .evaluate import save_confusion_matrix_plot
from .features import FeatureConfig, extract_feature_matrix, maybe_apply_pca
from .utils import save_json


def _macro_f1(metrics: dict[str, Any]) -> float:
    # Macro-F1 treats each class equally, which is important for medical-class balance.
    return float(metrics["macro avg"]["f1-score"])


def run_baseline_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: list[str],
    output_dir: str | Path,
    feature_cfg: FeatureConfig | None = None,
) -> dict[str, Any]:
    """Train and evaluate classical baselines on a shared split."""
    feature_cfg = feature_cfg or FeatureConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract features once per split so all baseline models share identical inputs.
    x_train, y_train = extract_feature_matrix(train_df, feature_cfg)
    x_val, y_val = extract_feature_matrix(val_df, feature_cfg)
    x_test, y_test = extract_feature_matrix(test_df, feature_cfg)

    x_train, x_val, x_test, pca = maybe_apply_pca(
        x_train, x_val, x_test, pca_components=feature_cfg.pca_components
    )

    models = {
        # Pipelines standardize features where appropriate before model fitting.
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=500)),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=5.0, gamma="scale")),
            ]
        ),
        "random_forest": RandomForestClassifier(n_estimators=250, random_state=42),
    }

    summary: dict[str, Any] = {
        "feature_config": asdict(feature_cfg),
        "pca_components_used": None if pca is None else int(pca.n_components_),
        "models": {},
    }

    from sklearn.metrics import classification_report

    best_model_name = ""
    best_val_f1 = -1.0
    best_test_metrics: dict[str, Any] = {}

    for model_name, model in models.items():
        # Each model is fit on train only; validation decides "best" model.
        model.fit(x_train, y_train)

        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)

        val_metrics = classification_report(
            y_val, val_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        test_metrics = classification_report(
            y_test, test_pred, target_names=class_names, output_dict=True, zero_division=0
        )

        val_f1 = _macro_f1(val_metrics)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_name = model_name
            best_test_metrics = test_metrics

        # Store test confusion matrix per model for presentation and error analysis.
        save_confusion_matrix_plot(
            y_test.tolist(),
            test_pred.tolist(),
            class_names,
            output_dir / f"{model_name}_test_confusion_matrix.png",
            title=f"{model_name} Test Confusion Matrix",
        )

        summary["models"][model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
        }

    summary["best_model_by_validation_macro_f1"] = best_model_name
    summary["best_model_test_metrics"] = best_test_metrics

    save_json(summary, output_dir / "baseline_metrics.json")
    return summary

