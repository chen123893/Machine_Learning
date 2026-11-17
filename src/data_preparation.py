"""Part D feature selection and preprocessing for the loan-default baseline."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .eda_preprocessing import TARGET, PRICING_EXCLUSIONS


AGE_BANDS = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]

BASELINE_EXCLUSIONS = {
    "id": "Application identifier.",
    "year": "Single recorded year; no time variation in this dataset.",
    **{name: "Pricing field with outcome-linked missingness." for name in PRICING_EXCLUSIONS},
    "approv_in_adv": "Approval timing is unverified.",
    "property_value": "Valuation availability and missingness require verification.",
    "ltv": "Depends on the valuation; missingness is outcome-linked.",
    "dtir1": "Availability of the recorded underwriting ratio is unverified.",
    "submission_of_application": "Workflow field with outcome-linked missingness.",
    "credit_type": "The EQUI category has a near-deterministic association with default.",

}


def split_loan_data(df: pd.DataFrame):
    """Reserve 20% of the records for final evaluation with a reproducible split."""
    y = df[TARGET]
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Expected a non-missing binary status column.")
    if df["id"].isna().any() or not df["id"].is_unique:
        raise ValueError("Expected a unique, non-missing ID for each application.")

    X = df.drop(columns=[TARGET, *BASELINE_EXCLUSIONS], errors="ignore")
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def screen_target_proxies(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Flag near-deterministic categorical levels or numeric missingness."""
    if not X.index.equals(y.index):
        raise ValueError("X and y must have aligned row indexes.")
    rows = []
    min_support = 50
    positive_count = int(y.sum())
    for column in X.columns:
        values = X[column]
        if pd.api.types.is_numeric_dtype(values):
            groups = {"<missing>": values.isna()}
        else:
            groups = {str(level): values.eq(level).fillna(False) for level in values.dropna().unique()}
            groups["<missing>"] = values.isna()
        for level, mask in groups.items():
            count = int(mask.sum())
            if count < min_support or len(y) - count < min_support:
                continue
            positives = int(y[mask].sum())
            rate = positives / count
            other_rate = (positive_count - positives) / (len(y) - count)
            rows.append({
                "feature": column, "group": level, "support": count,
                "default_rate": rate, "other_default_rate": other_rate,
                "flagged": (rate >= 0.98 or rate <= 0.02) and abs(rate - other_rate) >= 0.50,
            })
    return pd.DataFrame(rows, columns=[
        "feature", "group", "support", "default_rate", "other_default_rate", "flagged",
    ]).astype({"flagged": bool})


class TrainingFoldPreprocessor(TransformerMixin, BaseEstimator):
    """Fit fixed eligible inputs and audit potential proxies on training labels."""

    def __init__(self, *, log_income: bool = False):
        self.log_income = log_income

    def fit(self, X: pd.DataFrame, y):
        labels = y if isinstance(y, pd.Series) else pd.Series(y, index=X.index)
        self.screen_audit_ = screen_target_proxies(X, labels)
        self.selected_features_ = list(X.columns)
        self.transformer_ = build_preprocessor(X[self.selected_features_], log_income=self.log_income)
        # Use explicit zeros consistently for scikit-learn and XGBoost.
        self.transformer_.set_params(sparse_threshold=0.0)
        self.transformer_.fit(X[self.selected_features_])
        return self

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, "transformer_")
        return self.transformer_.transform(X[self.selected_features_])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "transformer_")
        return self.transformer_.get_feature_names_out()


def _log_income_names(transformer, input_features):
    return ["log_income"]


def build_preprocessor(X_train: pd.DataFrame, *, log_income: bool = False):
    """Build a fresh transformer to fit inside each training fold in Part E."""
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    # Age is ordered, while gender remains a nominal one-hot category.
    numeric = [column for column in numeric if column != "age"]
    categorical = [column for column in X_train.columns.difference(numeric, sort=False) if column != "age"]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                                  ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], remainder="drop", verbose_feature_names_out=False)

    if "age" in X_train:
        preprocessor.transformers.append(("age", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=False)),
            ("ordinal", OrdinalEncoder(categories=[AGE_BANDS], handle_unknown="error")),
        ]), ["age"]))

    if log_income and "income" in numeric:
        income_pipeline = Pipeline([
            ("log1p", FunctionTransformer(np.log1p, feature_names_out=_log_income_names)),
            ("prepare", clone(preprocessor.get_params()["numeric"])),
        ])
        preprocessor.transformers = [
            (name, transformer, [column for column in columns if column != "income"])
            for name, transformer, columns in preprocessor.transformers
        ]
        preprocessor.transformers.append(("log_income", income_pipeline, ["income"]))

    return preprocessor
