"""Metrics and robustness checks for the frozen Part H candidates."""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    roc_auc_score, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .model_tuning import SEED
from .model_validation import top_fraction_predictions


RECALL_TARGETS = (0.50, 0.60, 0.70, 0.80)


def select_thresholds(y, probabilities):
    """Choose cutoffs using only development labels and out-of-fold scores."""
    y, p = np.asarray(y), np.asarray(probabilities)
    if y.ndim != 1 or p.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Aligned binary labels containing both classes are required.")
    if not (np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))):
        raise ValueError("Probabilities must be finite and in [0, 1].")
    precision, recall, cuts = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    chosen = {"fixed_050": 0.5}
    for beta in (1, 2):
        denominator = beta**2 * precision + recall
        score = np.divide((1+beta**2) * precision * recall, denominator,
                          out=np.zeros_like(denominator), where=denominator > 0)
        chosen[f"max_f{beta}"] = float(cuts[np.flatnonzero(score == score.max())[-1]])
    for goal in RECALL_TARGETS:
        candidates = np.flatnonzero(recall >= goal)
        if len(candidates):
            best = candidates[precision[candidates] == precision[candidates].max()][-1]
            chosen[f"recall_{round(goal * 100)}"] = float(cuts[best])
    return chosen


def classification_metrics(y, predicted):
    tn, fp, fn, tp = (int(n) for n in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    positive = tp + fp
    rows = tn + fp + fn + tp
    return {
        "rows": rows, "defaults": tp + fn, "tn": tn, "false_positives": fp, "missed_defaults": fn,
        "defaults_caught": tp, "accuracy": (tp + tn) / rows,
        "precision": tp / positive if positive else 0.0, "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn), "f2": 5 * tp / (5 * tp + fp + 4 * fn),
        "flagged": positive, "flagged_pct": 100 * positive / rows,
    }


def ranking_metrics(y, probabilities):
    return {"ap": average_precision_score(y, probabilities), "roc_auc": roc_auc_score(y, probabilities),
            "brier": brier_score_loss(y, probabilities)}


def tune_validation_thresholds(models, X_dev, y_dev, *, n_splits=3):
    """Tune fixed candidates on stratified OOF predictions. Settings also used development data."""
    if not X_dev.index.equals(y_dev.index):
        raise ValueError("Feature rows and labels must be aligned.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    results, probabilities = [], pd.DataFrame(index=X_dev.index)
    for name, model in models.items():
        p = cross_val_predict(model, X_dev, y_dev, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        probabilities[name] = p
        ranking = ranking_metrics(y_dev, p)
        for strategy, threshold in select_thresholds(y_dev, p).items():
            results.append({"model": name, "strategy": strategy, "threshold": threshold,
                            **ranking, **classification_metrics(y_dev, p >= threshold)})
        print(f"{name}: development thresholds frozen.", flush=True)
    return pd.DataFrame(results), probabilities


def _default_scores(model, X):
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Expected 1 to be the default class.")
    p = model.predict_proba(X)[:, 1]
    if not (np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))):
        raise ValueError("Expected finite default probabilities in [0, 1].")
    return p


def evaluate_holdout(models, X_dev, y_dev, X_holdout, y_holdout, *, thresholds=None):
    """Apply frozen cutoffs to the reserved split after fitting each model on development."""
    if not X_dev.index.is_unique or not X_holdout.index.is_unique:
        raise ValueError("Rows must have unique split indexes.")
    if not X_dev.index.intersection(X_holdout.index).empty:
        raise ValueError("Development and holdout rows overlap.")
    if not X_dev.index.equals(y_dev.index) or not X_holdout.index.equals(y_holdout.index):
        raise ValueError("Feature rows and labels must be aligned.")
    if thresholds is None or set(thresholds) != set(models):
        raise ValueError("Supply frozen development thresholds for every model.")
    if any(not np.isfinite(value) or not 0 <= value <= 1
           for rules in thresholds.values() for value in rules.values()):
        raise ValueError("Thresholds must be finite probabilities.")

    y = y_holdout.to_numpy()
    rare_features = [name for name in ("construction_type", "secured_by", "security_type") if name in X_holdout]
    scenarios = {
        "income replaced with development median": X_holdout.assign(income=float(X_dev["income"].median())),
        "income unavailable": X_holdout.assign(income=np.nan),
        "security and construction categories unrecognised":
            X_holdout.assign(**{name: "__unknown__" for name in rare_features}),
    }
    rare_levels = {"construction_type": "mh", "secured_by": "land", "security_type": "Indirect"}
    rare_rows = X_holdout[rare_features].eq(pd.Series(rare_levels).reindex(rare_features)).any(axis=1)
    common = {name: X_holdout[name].mask(rare_rows, X_dev[name].mode().iloc[0]) for name in rare_features}
    scenarios["rare security and construction profiles replaced with common categories"] = X_holdout.assign(**common)
    ranking_rows, threshold_rows, stress_rows, predictions = [], [], [], {}
    for name, model in models.items():
        start = perf_counter()
        model.fit(X_dev, y_dev)
        fit_seconds = perf_counter() - start
        p = _default_scores(model, X_holdout)
        predictions[name] = p
        timings = []
        for _ in range(3):
            start = perf_counter()
            model.predict_proba(X_holdout)
            timings.append(perf_counter() - start)

        ranking = ranking_metrics(y, p)
        train_ap = average_precision_score(y_dev, _default_scores(model, X_dev))
        ranking_rows.append({"model": name, **ranking, "train_ap": train_ap,
                             "train_holdout_ap_gap": train_ap - ranking["ap"],
                             "fit_seconds": fit_seconds,
                             "batch_predict_ms_per_1000": np.median(timings) * 1_000_000 / len(y)})
        decisions = {rule: (cut, p >= cut) for rule, cut in thresholds[name].items()}
        decisions["top_10_pct"] = (np.nan, top_fraction_predictions(p))
        for rule, (cut, decisions_at_cut) in decisions.items():
            threshold_rows.append({"model": name, "strategy": rule, "threshold": cut, **ranking,
                                   **classification_metrics(y, decisions_at_cut)})

        original_top10 = classification_metrics(y, top_fraction_predictions(p))
        for scenario, inputs in scenarios.items():
            stress_p = _default_scores(model, inputs)
            stressed = ranking_metrics(y, stress_p)
            stressed_top10 = classification_metrics(y, top_fraction_predictions(stress_p))
            stress_rows.append({"model": name, "scenario": scenario, **stressed,
                                "rows_with_changed_probability": int((np.abs(p - stress_p) > 1e-10).sum()),
                                "ap_change_pp": 100 * (stressed["ap"] - ranking["ap"]),
                                "top10_defaults_caught": stressed_top10["defaults_caught"],
                                "top10_change_in_defaults_caught": stressed_top10["defaults_caught"] -
                                original_top10["defaults_caught"]})
        print(f"{name}: fitted and evaluated on reserved rows.", flush=True)
    return (pd.DataFrame(ranking_rows), pd.DataFrame(threshold_rows),
            pd.DataFrame(stress_rows), predictions)


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


def bootstrap_summary(y, predictions, reference, thresholds, *, repeats=1000):
    """Reuse class-stratified resamples for all AP differences and fixed-cutoff recall intervals."""
    y = np.asarray(y)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(SEED)
    ap = {name: np.empty(repeats) for name in predictions}
    recall = {name: np.empty(repeats) for name in predictions}
    f1_recall = {name: np.empty(repeats) for name in predictions}
    for i in range(repeats):
        sampled_positive = rng.choice(positive, len(positive), replace=True)
        sampled = np.concatenate([sampled_positive, rng.choice(negative, len(negative), replace=True)])
        for name, p in predictions.items():
            ap[name][i] = average_precision_score(y[sampled], p[sampled])
            recall[name][i] = np.mean(p[sampled_positive] >= thresholds[name]["recall_50"])
            f1_recall[name][i] = np.mean(p[sampled_positive] >= thresholds[name]["max_f1"])
    rows = []
    for name in predictions:
        low, high = np.percentile(ap[name], [2.5, 97.5])
        delta_low, delta_high = np.percentile(100 * (ap[name] - ap[reference]), [2.5, 97.5])
        recall_low, recall_high = np.percentile(recall[name], [2.5, 97.5])
        f1_low, f1_high = np.percentile(f1_recall[name], [2.5, 97.5])
        rows.append({"model": name, "f1_recall_ci_low": f1_low, "f1_recall_ci_high": f1_high, "reference": reference, "ap_ci_low": low, "ap_ci_high": high,
                     "ap_difference_ci_low_pp": delta_low, "ap_difference_ci_high_pp": delta_high,
                     "recall_50_ci_low": recall_low, "recall_50_ci_high": recall_high, "resamples": repeats})
    return pd.DataFrame(rows)
