"""Metrics and robustness checks for the frozen Part H candidates."""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .model_tuning import SEED
from .model_validation import top_fraction_predictions


def select_f1_threshold(y, probabilities):
    """Choose maximum validation F1; ties favour the higher threshold."""
    y, p = np.asarray(y), np.asarray(probabilities)
    if y.ndim != 1 or p.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Aligned binary labels containing both classes are required.")
    if not np.isfinite(p).all() or not np.all((p >= 0) & (p <= 1)):
        raise ValueError("Probabilities must be finite and in [0, 1].")
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) != 0)
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best])


def tune_validation_thresholds(models, X_dev, y_dev, *, n_splits=5):
    """Tune fixed candidates on development OOF predictions, never holdout labels.

    All preprocessing is refitted inside each stratified fold. These are tuning
    scores, not unbiased performance estimates: settings also used development
    data. The reserved split is used only after thresholds have been frozen.
    """
    if not X_dev.index.equals(y_dev.index):
        raise ValueError("Feature rows and labels must be aligned.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    results, probabilities = [], pd.DataFrame(index=X_dev.index)
    for name, model in models.items():
        p = cross_val_predict(model, X_dev, y_dev, cv=cv,
                              method="predict_proba", n_jobs=1)[:, 1]
        threshold = select_f1_threshold(y_dev, p)
        predicted = p >= threshold
        probabilities[name] = p
        results.append({
            "model": name, "threshold": threshold, "objective": "F1",
            "validation_f1": f1_score(y_dev, predicted),
            "validation_precision": precision_score(y_dev, predicted, zero_division=0),
            "validation_recall": recall_score(y_dev, predicted),
            "validation_accuracy": accuracy_score(y_dev, predicted),
            "validation_f1_050": f1_score(y_dev, p >= 0.5),
            "folds": n_splits,
        })
    return pd.DataFrame(results).set_index("model"), probabilities


def _default_scores(model, X):
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Expected 1 to be the default class.")
    p = model.predict_proba(X)[:, 1]
    if not np.isfinite(p).all() or not np.all((p >= 0) & (p <= 1)):
        raise ValueError("Expected finite default probabilities in [0, 1].")
    return p


def _review_metrics(y, p):
    queue = top_fraction_predictions(p).astype(bool)
    caught = int(y[queue].sum())
    return {
        "reviewed": int(queue.sum()),
        "defaults_caught": caught,
        "review_precision": caught / queue.sum(),
        "review_recall": caught / y.sum(),
    }


def evaluate_holdout(models, X_dev, y_dev, X_holdout, y_holdout, *, thresholds=None):
    """Fit fixed pipelines on development data and score the reserved split."""
    if not X_dev.index.is_unique or not X_holdout.index.is_unique:
        raise ValueError("Rows must have unique split indexes.")
    if not X_dev.index.intersection(X_holdout.index).empty:
        raise ValueError("Development and holdout rows overlap.")
    if not X_dev.index.equals(y_dev.index) or not X_holdout.index.equals(y_holdout.index):
        raise ValueError("Feature rows and labels must be aligned.")
    if thresholds is not None:
        if set(thresholds) != set(models):
            raise ValueError("Supply one frozen validation threshold per model.")
        if any(not np.isfinite(t) or not 0 <= t <= 1 for t in thresholds.values()):
            raise ValueError("Thresholds must be finite and in [0, 1].")

    y = y_holdout.to_numpy()
    # Valid complete-case stress: replace income by the DEVELOPMENT median.
    # Missing inputs are rejected by the revised preprocessing contract.
    stress_inputs = X_holdout.assign(income=float(X_dev["income"].median()))
    results, stress_results, predictions = [], [], {}
    for name, model in models.items():
        start = perf_counter()
        model.fit(X_dev, y_dev)
        fit_seconds = perf_counter() - start
        p = _default_scores(model, X_holdout)
        predictions[name] = p

        # The first holdout prediction warms up the pipeline. Time three further calls.
        timings = []
        for _ in range(3):
            start = perf_counter()
            model.predict_proba(X_holdout)
            timings.append(perf_counter() - start)

        train_ap = average_precision_score(y_dev, _default_scores(model, X_dev))
        ap = average_precision_score(y, p)
        predicted = p >= 0.50
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        review = _review_metrics(y, p)
        results.append({
            "model": name, "ap": ap, "roc_auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p),
            "accuracy_050": accuracy_score(y, predicted),
            "precision_050": precision_score(y, predicted, zero_division=0),
            "recall_050": recall_score(y, predicted, zero_division=0),
            "f1_050": f1_score(y, predicted, zero_division=0),
            "tn_050": int(tn), "fp_050": int(fp), "fn_050": int(fn), "tp_050": int(tp),
            **review, "train_ap": train_ap, "train_holdout_ap_gap": train_ap - ap,
            "fit_seconds": fit_seconds,
            "batch_predict_ms_per_1000": np.median(timings) * 1_000_000 / len(y),
        })
        if thresholds is not None:
            threshold = float(thresholds[name])
            tuned = p >= threshold
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y, tuned, labels=[0, 1]).ravel()
            results[-1].update({
                "threshold": threshold,
                "accuracy_tuned": accuracy_score(y, tuned),
                "precision_tuned": precision_score(y, tuned, zero_division=0),
                "recall_tuned": recall_score(y, tuned, zero_division=0),
                "f1_tuned": f1_score(y, tuned, zero_division=0),
                "tn_tuned": int(tn_t), "fp_tuned": int(fp_t),
                "fn_tuned": int(fn_t), "tp_tuned": int(tp_t),
            })

        stress_p = _default_scores(model, stress_inputs)
        stress_ap = average_precision_score(y, stress_p)
        stress_review = _review_metrics(y, stress_p)
        stress_results.append({
            "model": name, "original_ap": ap, "income_median_ap": stress_ap,
            "ap_change_pp": 100 * (stress_ap - ap), "original_defaults_caught": review["defaults_caught"],
            "income_median_defaults_caught": stress_review["defaults_caught"],
        })
        print(f"{name}: fitted and scored.", flush=True)
    return (
        pd.DataFrame(results).set_index("model"),
        pd.DataFrame(stress_results).set_index("model"),
        predictions,
    )


def paired_ap_interval(y, candidate, reference, *, repeats=1000, seed=SEED):
    """Percentile interval for the difference between fixed predictions on the same rows."""
    y = np.asarray(y)
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if y.ndim != 1 or candidate.shape != y.shape or reference.shape != y.shape:
        raise ValueError("Labels and prediction arrays must be aligned one-dimensional arrays.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("The holdout must contain both classes.")
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    deltas = np.empty(repeats)
    for i in range(repeats):
        sampled = np.concatenate([
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ])
        deltas[i] = (
            average_precision_score(y[sampled], candidate[sampled])
            - average_precision_score(y[sampled], reference[sampled])
        )
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "ap_difference_pp": 100 * (average_precision_score(y, candidate) - average_precision_score(y, reference)),
        "ci_low_pp": 100 * low, "ci_high_pp": 100 * high, "resamples": repeats,
    }
