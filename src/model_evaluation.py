"""Metrics and robustness checks for the frozen Part H candidates."""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from .model_tuning import SEED
from .model_validation import top_fraction_predictions


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


def evaluate_holdout(models, X_dev, y_dev, X_holdout, y_holdout):
    """Fit fixed pipelines on development data and score the reserved split."""
    if not X_dev.index.is_unique or not X_holdout.index.is_unique:
        raise ValueError("Rows must have unique split indexes.")
    if not X_dev.index.intersection(X_holdout.index).empty:
        raise ValueError("Development and holdout rows overlap.")
    if not X_dev.index.equals(y_dev.index) or not X_holdout.index.equals(y_holdout.index):
        raise ValueError("Feature rows and labels must be aligned.")

    y = y_holdout.to_numpy()
    stress_inputs = X_holdout.assign(income=np.nan)
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

        stress_p = _default_scores(model, stress_inputs)
        stress_ap = average_precision_score(y, stress_p)
        stress_review = _review_metrics(y, stress_p)
        stress_results.append({
            "model": name, "original_ap": ap, "income_missing_ap": stress_ap,
            "ap_change_pp": 100 * (stress_ap - ap), "original_defaults_caught": review["defaults_caught"],
            "income_missing_defaults_caught": stress_review["defaults_caught"],
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
