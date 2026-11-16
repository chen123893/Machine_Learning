"""Part F model configurations and nested development-only evaluation."""

import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .data_preparation import TrainingFoldPreprocessor
from .model_validation import top_fraction_predictions


SEED = 42
MODEL_NAMES = ("Logistic Regression", "Random Forest", "XGBoost")
SEARCH_GRIDS = {
    "Logistic Regression": {"model__C": [1.0, 0.01, 0.1, 10.0], "prepare__log_income": [False, True], "balanced": [False, True]},
    "Random Forest": [
        {"model__n_estimators": [100], "model__max_depth": [12, 18], "model__min_samples_leaf": [10, 5], "balanced": [False, True]},
        {"model__n_estimators": [160], "model__max_depth": [12, 18], "model__min_samples_leaf": [10], "balanced": [False, True]},
        {"model__n_estimators": [100], "model__max_depth": [8], "model__min_samples_leaf": [20], "balanced": [False, True]},
    ],
    "XGBoost": {
        "model__n_estimators": [120, 220],
        "model__max_depth": [4, 6],
        "model__learning_rate": [0.1, 0.05],
        "balanced": [False, True],
    },
}


class BalancedTrainingPipeline(Pipeline):
    """Compute optional inverse-frequency weights from each fit's labels only.

    Computing weights here (rather than before GridSearchCV or stacking) keeps
    inner validation labels out of training weights. Validation remains natural.
    """
    def __init__(self, steps, *, balanced=False, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)
        self.balanced = balanced

    def fit(self, X, y=None, **params):
        self.class_counts_ = dict(zip(*np.unique(y, return_counts=True)))
        if self.balanced:
            if "model__sample_weight" in params:
                raise ValueError("Do not supply weights twice.")
            params["model__sample_weight"] = compute_sample_weight("balanced", y)
        return super().fit(X, y, **params)


def make_pipeline(model_name, params=None, *, workers=4):
    """Recreate an unfitted Part E baseline or apply saved Part F parameters."""
    if model_name == "Logistic Regression":
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=10,
            random_state=SEED, n_jobs=workers,
        )
    elif model_name == "XGBoost":
        model = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_estimators=120, max_depth=4, learning_rate=0.1,
            random_state=SEED, n_jobs=workers,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    pipeline = BalancedTrainingPipeline([("prepare", TrainingFoldPreprocessor()), ("model", model)])
    return pipeline.set_params(**(params or {}))


def _candidate_rows(model_name, scope, search):
    results = search.cv_results_
    fold_fields = [key for key in results if key.startswith("split") and key.endswith("_test_score")]
    return [
        {
            "model": model_name, "scope": scope, "parameters": json.dumps(params, sort_keys=True),
            "rank": results["rank_test_score"][i],
            "selected": i == search.best_index_,
            "mean_ap": results["mean_test_score"][i],
            "sd_ap": results["std_test_score"][i],
            "mean_fit_seconds": results["mean_fit_time"][i],
            **{field.replace("_test_score", "_ap"): results[field][i] for field in fold_fields},
        }
        for i, params in enumerate(results["params"])
    ]


def _outer_scores(pipeline, X, y):
    assert np.array_equal(pipeline.classes_, [0, 1])
    probabilities = pipeline.predict_proba(X)[:, 1]
    assert np.isfinite(probabilities).all()
    priority = top_fraction_predictions(probabilities)
    defaults_caught = int(np.dot(priority, y))
    return {
        "ap": average_precision_score(y, probabilities),
        "roc_auc": roc_auc_score(y, probabilities),
        "top10_precision": defaults_caught / priority.sum(),
        "top10_recall": defaults_caught / y.sum(),
    }


def tune_model(model_name, X_dev, y_dev, outer_folds, *, workers=4):
    """Evaluate the search on outer folds and separately select full-dev settings."""
    grid = SEARCH_GRIDS[model_name]
    scores, candidates = [], []
    for fold, (fit_rows, val_rows) in enumerate(outer_folds, start=1):
        if np.intersect1d(fit_rows, val_rows).size:
            raise ValueError("Training and validation rows overlap.")
        X_fit, X_val = X_dev.iloc[fit_rows], X_dev.iloc[val_rows]
        y_fit, y_val = y_dev.iloc[fit_rows], y_dev.iloc[val_rows]
        inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED + fold)
        search = GridSearchCV(
            make_pipeline(model_name, workers=workers), grid, scoring="average_precision",
            cv=inner_cv, refit=True, error_score="raise", n_jobs=1,
        )
        start = perf_counter()
        search.fit(X_fit, y_fit)
        search_seconds = perf_counter() - start
        candidates.extend(_candidate_rows(model_name, f"outer_{fold}_inner_search", search))

        references = [("Part E raw", {})]
        if model_name == "Logistic Regression":
            references.append(("Part E log", {"prepare__log_income": True}))
        for label, settings in references:
            baseline = make_pipeline(model_name, settings, workers=workers).fit(X_fit, y_fit)
            scores.append({
                "model": model_name, "fold": fold, "variant": label,
                "parameters": json.dumps(settings, sort_keys=True),
                **_outer_scores(baseline, X_val, y_val),
            })
        scores.append({
            "model": model_name, "fold": fold, "variant": "Nested tuned",
            "parameters": json.dumps(search.best_params_, sort_keys=True),
            "search_seconds": search_seconds,
            **_outer_scores(search.best_estimator_, X_val, y_val),
        })
        print(f"{model_name} | outer fold {fold} complete", flush=True)

    final_search = GridSearchCV(
        make_pipeline(model_name, workers=workers), grid, scoring="average_precision",
        cv=outer_folds, refit=False, error_score="raise", n_jobs=1,
    )
    final_search.fit(X_dev, y_dev)
    candidates.extend(_candidate_rows(model_name, "final_development_search", final_search))
    return pd.DataFrame(scores), pd.DataFrame(candidates), final_search.best_params_
