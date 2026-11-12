"""Shared development-only evaluation for Part E."""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from .data_preparation import TrainingFoldPreprocessor


def top_fraction_predictions(probability, fraction=0.10):
    """Flag ceil(fraction × rows), resolving ties in the supplied row order."""
    count = int(np.ceil(fraction * len(probability)))
    order = np.argsort(-np.asarray(probability), kind="stable")
    flagged = np.zeros(len(probability), dtype=int)
    flagged[order[:count]] = 1
    return flagged


def evaluate_models(X, y, folds, models, reference, seed=42):
    """Run matched training-fold experiments and collect validation evidence."""
    conditions = {
        "Raw": {},
        "Log": {"log_income": True},
        "Weighted": {"balanced": True},
    }
    models_with_reference = {"Majority reference": reference, **models}
    records, coefficients, importances, screens = [], [], [], []
    oof = pd.DataFrame({"status": y.to_numpy(), "fold": np.zeros(len(y), dtype=int)})
    for fold_number, (fit_rows, validation_rows) in enumerate(folds, start=1):
        if np.intersect1d(fit_rows, validation_rows).size or oof.loc[validation_rows, "fold"].any():
            raise ValueError("Validation folds must be disjoint from fit rows and each other.")
        oof.loc[validation_rows, "fold"] = fold_number
        X_fit, X_val = X.iloc[fit_rows], X.iloc[validation_rows]
        y_fit, y_val = y.iloc[fit_rows], y.iloc[validation_rows]
        sample_rows, _ = train_test_split(
            np.arange(len(y_val)), train_size=min(2000, len(y_val) - 1),
            stratify=y_val, random_state=seed,
        )
        for name, estimator in models_with_reference.items():
            variants = {"Raw": {}} if name == "Majority reference" else conditions.copy()
            if name == "Logistic Regression":
                variants["PCA 95%"] = {"pca": True}
            selected_for_fold = None if name == "Majority reference" else raw_selected
            for variant, options in variants.items():
                steps = [("prepare", TrainingFoldPreprocessor(log_income=options.get("log_income", False)))]
                if options.get("pca"):
                    steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
                steps.append(("model", clone(estimator)))
                pipeline = Pipeline(steps)
                fit_options = {}
                if options.get("balanced"):
                    fit_options["model__sample_weight"] = compute_sample_weight("balanced", y_fit)

                start = perf_counter()
                pipeline.fit(X_fit, y_fit, **fit_options)
                fit_seconds = perf_counter() - start
                prepare = pipeline.named_steps["prepare"]
                feature_names = prepare.get_feature_names_out()
                if name == "Majority reference":
                    raw_selected = prepare.selected_features_
                    screens.append(prepare.screen_audit_.assign(fold=fold_number))
                else:
                    assert prepare.selected_features_ == selected_for_fold
                assert np.array_equal(pipeline.classes_, [0, 1])

                # Warm up once, then time complete preprocessing and prediction.
                probability = pipeline.predict_proba(X_val)[:, 1]
                elapsed = []
                for _ in range(3):
                    start = perf_counter()
                    pipeline.predict_proba(X_val)
                    elapsed.append(perf_counter() - start)
                assert np.isfinite(probability).all() and ((0 <= probability) & (probability <= 1)).all()
                predicted = (probability >= 0.50).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val, predicted, labels=[0, 1]).ravel()
                priority = top_fraction_predictions(probability)
                row = {
                    "model": name, "variant": variant, "fold": fold_number,
                    "ap": average_precision_score(y_val, probability),
                    "train_ap": average_precision_score(y_fit, pipeline.predict_proba(X_fit)[:, 1]),
                    "roc_auc": roc_auc_score(y_val, probability),
                    "accuracy": accuracy_score(y_val, predicted),
                    "precision": precision_score(y_val, predicted, zero_division=0),
                    "recall": recall_score(y_val, predicted),
                    "f1": f1_score(y_val, predicted, zero_division=0),
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    "top10_flagged": priority.sum(), "top10_tp": int(np.dot(priority, y_val)),
                    "validation_defaults": int(y_val.sum()),
                    "encoded_features": len(feature_names),
                    "model_features": pipeline.named_steps["pca"].n_components_ if "pca" in pipeline.named_steps else len(feature_names),
                    "fit_seconds": fit_seconds,
                    "prediction_ms_per_1000": np.median(elapsed) * 1_000_000 / len(X_val),
                }
                records.append(row)

                if variant != "Raw":
                    continue
                oof.loc[validation_rows, name] = probability
                if name == "Logistic Regression":
                    coefficients.append(pd.DataFrame({
                        "feature": feature_names,
                        "coefficient": pipeline.named_steps["model"].coef_[0],
                        "fold": fold_number,
                    }))
                if name in {"Random Forest", "XGBoost"}:
                    # Permute original columns before preprocessing, preserving one-hot groups.
                    X_sample = X_val.iloc[sample_rows][prepare.selected_features_]
                    importance = permutation_importance(
                        pipeline, X_sample, y_val.iloc[sample_rows], scoring="average_precision",
                        n_repeats=3, random_state=seed, n_jobs=1,
                    )
                    importances.append(pd.DataFrame({
                        "model": name, "feature": prepare.selected_features_, "fold": fold_number,
                        "mean_ap_drop": importance.importances_mean, "validation_rows": len(X_sample),
                    }))
        print(f"Completed fold {fold_number}", flush=True)

    assert oof["fold"].gt(0).all() and oof.notna().all().all()
    assert oof[list(models_with_reference)].apply(lambda col: col.between(0, 1).all()).all()
    oof.index.name = "development_position"
    return (
        pd.DataFrame(records), oof, pd.concat(coefficients, ignore_index=True),
        pd.concat(importances, ignore_index=True), pd.concat(screens, ignore_index=True),
    )
