"""Build and evaluate the Part G voting and stacking classifiers."""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .model_tuning import MODEL_NAMES, SEED, make_pipeline
from .model_validation import top_fraction_predictions


MEMBERS = {"xgb": "XGBoost", "rf": "Random Forest"}
METHODS = ("Soft voting", "Stacking")


def configure_members(names):
    """Select the best two development models; keep imported mapping references live."""
    names = list(names)
    if len(names) != 2 or len(set(names)) != 2 or not set(names).issubset(MODEL_NAMES):
        raise ValueError("Expected two distinct supported base models.")
    MEMBERS.clear()
    MEMBERS.update(zip(("member_a", "member_b"), names))


def make_ensemble(method, settings, *, workers=4):
    """Build a fresh ensemble from Part F settings for development or Part H."""
    estimators = [
        (alias, make_pipeline(name, settings[name], workers=workers))
        for alias, name in MEMBERS.items()
    ]
    if method == "Soft voting":
        return VotingClassifier(estimators, voting="soft", n_jobs=1)
    if method == "Stacking":
        return StackingClassifier(
            estimators, final_estimator=LogisticRegression(max_iter=1000),
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
            stack_method="predict_proba", passthrough=False, n_jobs=1,
        )
    raise ValueError(f"Unknown ensemble method: {method}")


def _default_probability(estimator, X):
    if not np.array_equal(estimator.classes_, [0, 1]):
        raise ValueError("Expected class 1 to represent default.")
    probabilities = estimator.predict_proba(X)[:, 1]
    if not np.isfinite(probabilities).all() or not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("Default probabilities must be finite and within [0, 1].")
    return probabilities


def evaluate_ensembles(X_dev, y_dev, folds, part_f_scores, *, workers=4):
    """Use fold-specific Part F winners and compare on the same outer rows."""
    tuned = part_f_scores.query("variant == 'Nested tuned'").set_index(["fold", "model"], verify_integrity=True)
    records, overlaps = [], []
    coverage = np.zeros(len(y_dev), dtype=int)
    for fold, (train, validation) in enumerate(folds, start=1):
        if np.intersect1d(train, validation).size or coverage[validation].any():
            raise ValueError("Outer training and validation must be separate and cover each validation row once.")
        np.add.at(coverage, validation, 1)
        X_fit, X_val = X_dev.iloc[train], X_dev.iloc[validation]
        y_fit, y_val = y_dev.iloc[train], y_dev.iloc[validation]
        settings = {name: json.loads(tuned.at[(fold, name), "parameters"]) for name in MODEL_NAMES}

        stack = make_ensemble("Stacking", settings, workers=workers).fit(X_fit, y_fit)
        scores = {
            name: _default_probability(stack.named_estimators_[alias], X_val)
            for alias, name in MEMBERS.items()
        }
        # Equal-weight soft voting reuses the same fitted base models as the stack.
        # This is equivalent to fitting an independent VotingClassifier on this fold.
        scores["Soft voting"] = np.mean([scores[name] for name in MEMBERS.values()], axis=0)
        scores["Stacking"] = _default_probability(stack, X_val)
        for name in MODEL_NAMES:
            if name not in scores:
                other = make_pipeline(name, settings[name], workers=workers)
                scores[name] = _default_probability(other.fit(X_fit, y_fit), X_val)

        y = y_val.to_numpy()
        queues = {name: top_fraction_predictions(p).astype(bool) for name, p in scores.items()}
        for name, p in scores.items():
            queue = queues[name]
            caught = int(y[queue].sum())
            ap = average_precision_score(y, p)
            if name in MODEL_NAMES:
                np.testing.assert_allclose(ap, tuned.at[(fold, name), "ap"], rtol=0, atol=1e-5)
            records.append({
                "fold": fold, "model": name, "ap": ap, "roc_auc": roc_auc_score(y, p),
                "brier": brier_score_loss(y, p), "reviewed": int(queue.sum()),
                "defaults_in_fold": int(y.sum()), "defaults_caught": caught,
                "top10_precision": caught / queue.sum(), "top10_recall": caught / y.sum(),
            })

        first, second = MEMBERS.values()
        xgb_queue, rf_queue = queues[first], queues[second]
        both = xgb_queue & rf_queue
        either = xgb_queue | rf_queue
        overlaps.append({
            "fold": fold,
            "member_a": first, "member_b": second,
            "spearman_rank_correlation": pd.Series(scores[first]).corr(pd.Series(scores[second]), method="spearman"),
            "queue_jaccard": both.sum() / either.sum(),
            "defaults_caught_by_both": int(y[both].sum()),
            "defaults_caught_only_by_member_a": int(y[xgb_queue & ~rf_queue].sum()),
            "defaults_caught_only_by_member_b": int(y[rf_queue & ~xgb_queue].sum()),
            "defaults_caught_by_neither": int(y[~either].sum()),
            "stack_member_a_coefficient": stack.final_estimator_.coef_[0, 0],
            "stack_member_b_coefficient": stack.final_estimator_.coef_[0, 1],
        })
        print(f"Outer fold {fold}: five candidates scored; Part F baselines reproduced.", flush=True)
    if not np.all(coverage == 1):
        raise ValueError("Outer validation folds must cover every development row exactly once.")
    return pd.DataFrame(records), pd.DataFrame(overlaps)
