"""EDA and leakage-aware preprocessing utilities.

This module intentionally contains no model estimator and performs no model
training. Its purpose is to make the data audit reproducible and to define an
unfitted preprocessing object for a later modelling phase.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "status"
IDENTIFIER_COLUMNS = ["id"]
CONSTANT_COLUMNS = ["year"]

# These fields are quarantined from the first baseline because their missingness
# or categories almost determine the target in this dataset. They can only be
# reconsidered after the business confirms that they are available at decision
# time and are not consequences of the loan outcome.
LEAKAGE_QUARANTINE = [
    "rate_of_interest",
    "interest_rate_spread",
    "upfront_charges",
    "property_value",
    "ltv",
    "dtir1",
    "age",
    "submission_of_application",
    "credit_type",
]

EXPECTED_RAW_COLUMNS = {
    "ID",
    "year",
    "loan_limit",
    "Gender",
    "approv_in_adv",
    "loan_type",
    "loan_purpose",
    "Credit_Worthiness",
    "open_credit",
    "business_or_commercial",
    "loan_amount",
    "rate_of_interest",
    "Interest_rate_spread",
    "Upfront_charges",
    "term",
    "Neg_ammortization",
    "interest_only",
    "lump_sum_payment",
    "property_value",
    "construction_type",
    "occupancy_type",
    "Secured_by",
    "total_units",
    "income",
    "credit_type",
    "Credit_Score",
    "co-applicant_credit_type",
    "age",
    "submission_of_application",
    "LTV",
    "Region",
    "Security_Type",
    "Status",
    "dtir1",
}


FEATURE_DESCRIPTIONS = {
    "id": "Unique loan application identifier; excluded from modelling.",
    "year": "Calendar year of the application record.",
    "loan_limit": "Whether the loan conforms to the applicable loan limit.",
    "gender": "Applicant gender category as supplied by the source dataset.",
    "approv_in_adv": "Whether approval was obtained in advance.",
    "loan_type": "Source-coded loan product type.",
    "loan_purpose": "Source-coded purpose of the loan.",
    "credit_worthiness": "Source-coded credit-worthiness category.",
    "open_credit": "Whether the application is an open-credit transaction.",
    "business_or_commercial": "Whether the loan is business/commercial rather than personal.",
    "loan_amount": "Requested or originated loan principal amount.",
    "rate_of_interest": "Recorded annual interest rate; availability timing requires verification.",
    "interest_rate_spread": "Difference between the loan rate and a benchmark; leakage-risk field.",
    "upfront_charges": "Recorded upfront fees or charges; availability timing requires verification.",
    "term": "Loan term in months.",
    "neg_amortization": "Whether the loan permits negative amortisation.",
    "interest_only": "Whether the loan has an interest-only feature.",
    "lump_sum_payment": "Whether the loan contains a lump-sum or balloon payment feature.",
    "property_value": "Recorded value of the secured property.",
    "construction_type": "Construction type of the secured property.",
    "occupancy_type": "Intended occupancy category.",
    "secured_by": "Asset category securing the loan.",
    "total_units": "Number of property units, supplied as ordered labels (1U-4U).",
    "income": "Recorded applicant income in the source dataset's units.",
    "credit_type": "Credit bureau/reporting source category; leakage risk requires investigation.",
    "credit_score": "Applicant credit score.",
    "co_applicant_credit_type": "Co-applicant credit bureau/reporting source category.",
    "age": "Applicant age band.",
    "submission_of_application": "Application submission channel/status category.",
    "ltv": "Loan-to-value ratio as a percentage.",
    "region": "Geographic region category.",
    "security_type": "Direct or indirect security type.",
    "status": "Binary target: 1 denotes default and 0 denotes non-default in this project.",
    "dtir1": "Debt-to-income ratio measure as supplied by the source dataset.",
}


def _snake_case(name: str) -> str:
    """Convert a raw column name to stable snake_case."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", name.strip())
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return name.strip("_").lower()


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the source CSV without mutating it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_csv(path)
    missing = EXPECTED_RAW_COLUMNS - set(df.columns)
    unexpected = set(df.columns) - EXPECTED_RAW_COLUMNS
    if missing or unexpected:
        raise ValueError(
            f"Schema mismatch. Missing={sorted(missing)}; unexpected={sorted(unexpected)}"
        )
    if len(df) < 5_000 or df.shape[1] < 10:
        raise ValueError("Dataset does not satisfy the assignment size requirements.")
    if not set(df["Status"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Status must be a binary 0/1 target.")
    if df["Status"].isna().any():
        raise ValueError("Status contains missing values; target treatment must be explicit.")
    return df


def deterministic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply defensible, outcome-independent cleaning only.

    Statistical transformations such as imputation and scaling are deliberately
    excluded because they must later be fitted on training folds only.
    """
    clean = df.copy()
    clean.columns = [_snake_case(column) for column in clean.columns]
    clean = clean.rename(columns={"neg_ammortization": "neg_amortization"})

    object_columns = clean.select_dtypes(include="object").columns
    for column in object_columns:
        clean[column] = clean[column].str.strip()

    clean["region"] = clean["region"].replace(
        {"south": "South", "central": "Central"}
    )
    clean["security_type"] = clean["security_type"].replace(
        {"Indriect": "Indirect"}
    )

    # Remove exact duplicates only if they exist; never deduplicate on ID alone.
    clean = clean.drop_duplicates().reset_index(drop=True)
    return clean


def feature_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Describe every cleaned feature and its analytical role."""
    rows = []
    for column in df.columns:
        if column == TARGET:
            role = "target"
        elif column in IDENTIFIER_COLUMNS:
            role = "identifier"
        elif column in CONSTANT_COLUMNS:
            role = "constant / exclude"
        elif column in LEAKAGE_QUARANTINE:
            role = "candidate predictor / leakage review"
        else:
            role = "candidate predictor"
        rows.append(
            {
                "feature": column,
                "description": FEATURE_DESCRIPTIONS.get(
                    column, "Description requires confirmation from the data publisher."
                ),
                "dtype": str(df[column].dtype),
                "analytical_role": role,
            }
        )
    return pd.DataFrame(rows)


def cleaning_change_log(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Return a transparent, count-based record of deterministic cleaning."""
    standardized = [_snake_case(column) for column in raw.columns]
    standardized = [
        "neg_amortization" if column == "neg_ammortization" else column
        for column in standardized
    ]
    renamed_columns = sum(old != new for old, new in zip(raw.columns, standardized))

    trimmed_cells = 0
    for column in raw.select_dtypes(include="object").columns:
        original = raw[column].astype("string")
        stripped = original.str.strip()
        trimmed_cells += int((original.fillna("<NA>") != stripped.fillna("<NA>")).sum())

    region_changes = int(raw["Region"].isin(["south", "central"]).sum())
    security_changes = int((raw["Security_Type"] == "Indriect").sum())
    duplicates_removed = int(len(raw) - len(clean))

    return pd.DataFrame(
        [
            {
                "cleaning_action": "Standardise column names to snake_case and correct neg_amortization spelling",
                "items_changed": renamed_columns,
                "reason": "Consistent, reproducible feature naming",
                "risk_or_limitation": "Names change, but values do not",
            },
            {
                "cleaning_action": "Strip leading/trailing whitespace from categorical values",
                "items_changed": trimmed_cells,
                "reason": "Prevent accidental duplicate categories",
                "risk_or_limitation": "Internal whitespace is intentionally preserved",
            },
            {
                "cleaning_action": "Standardise south/central region capitalisation",
                "items_changed": region_changes,
                "reason": "Merge labels that differ only by case",
                "risk_or_limitation": "Assumes case has no semantic meaning",
            },
            {
                "cleaning_action": "Correct Indriect security label to Indirect",
                "items_changed": security_changes,
                "reason": "Treat apparent spelling error as one category",
                "risk_or_limitation": "Assumption should be confirmed against source documentation",
            },
            {
                "cleaning_action": "Remove exact duplicate rows",
                "items_changed": duplicates_removed,
                "reason": "Prevent repeated identical observations",
                "risk_or_limitation": "No deduplication is performed on ID alone",
            },
        ]
    )


def post_clean_validation(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Verify that deterministic cleaning preserved key data invariants."""
    string_columns = clean.select_dtypes(include="object").columns
    whitespace_cells = sum(
        int(
            (
                clean[column].astype("string").fillna("<NA>")
                != clean[column].astype("string").str.strip().fillna("<NA>")
            ).sum()
        )
        for column in string_columns
    )
    expected_rows = len(raw) - int(raw.duplicated().sum())
    checks = [
        ("Expected row count after exact deduplication", len(clean) == expected_rows, expected_rows, len(clean)),
        ("Column names are unique", clean.columns.is_unique, True, clean.columns.is_unique),
        ("Target has no missing values", clean[TARGET].isna().sum() == 0, 0, int(clean[TARGET].isna().sum())),
        ("Target remains binary 0/1", set(clean[TARGET].unique()) == {0, 1}, "{0, 1}", sorted(clean[TARGET].unique().tolist())),
        ("Identifier has no missing values", clean["id"].isna().sum() == 0, 0, int(clean["id"].isna().sum())),
        ("Identifier remains unique", clean["id"].is_unique, True, clean["id"].is_unique),
        ("No exact duplicates remain", clean.duplicated().sum() == 0, 0, int(clean.duplicated().sum())),
        ("No leading/trailing categorical whitespace remains", whitespace_cells == 0, 0, whitespace_cells),
        ("Inconsistent region labels removed", not clean["region"].isin(["south", "central"]).any(), 0, int(clean["region"].isin(["south", "central"]).sum())),
        ("Misspelled security label removed", not (clean["security_type"] == "Indriect").any(), 0, int((clean["security_type"] == "Indriect").sum())),
        ("Target counts preserved", raw["Status"].value_counts().sort_index().equals(clean[TARGET].value_counts().sort_index()), True, raw["Status"].value_counts().sort_index().equals(clean[TARGET].value_counts().sort_index())),
    ]
    return pd.DataFrame(
        checks, columns=["validation_check", "passed", "expected", "observed"]
    )


def data_validity_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Flag impossible or suspicious values without silently changing them."""
    numeric = df.select_dtypes(include=np.number)
    rules = [
        ("numeric_infinite", np.isinf(numeric).any(axis=1), "Impossible", "Investigate source or set to missing only with evidence"),
        ("loan_amount_non_positive", df["loan_amount"].le(0), "Impossible", "Investigate source; loan amount should be positive"),
        ("term_non_positive", df["term"].notna() & df["term"].le(0), "Impossible", "Investigate source; term should be positive"),
        ("credit_score_outside_300_900", ~df["credit_score"].between(300, 900), "Potentially impossible", "Confirm scoring scale with source"),
        ("negative_income", df["income"].notna() & df["income"].lt(0), "Potentially impossible", "Confirm units and definition"),
        ("zero_income", df["income"].eq(0), "Suspicious but plausible", "Retain; examine missingness/zero semantics"),
        ("non_positive_interest_rate", df["rate_of_interest"].notna() & df["rate_of_interest"].le(0), "Suspicious", "Retain pending domain verification"),
        ("non_positive_property_value", df["property_value"].notna() & df["property_value"].le(0), "Potentially impossible", "Confirm source; do not repair automatically"),
        ("ltv_above_200_pct", df["ltv"].notna() & df["ltv"].gt(200), "Extreme / possible data issue", "Investigate individually; do not remove by IQR alone"),
        ("dti_outside_0_100", df["dtir1"].notna() & ~df["dtir1"].between(0, 100), "Potentially impossible", "Confirm ratio definition and range"),
    ]
    rows = []
    for rule, mask, severity, action in rules:
        count = int(mask.sum())
        rows.append(
            {
                "rule": rule,
                "severity": severity,
                "flagged_count": count,
                "flagged_pct": round(count / len(df) * 100, 3),
                "recommended_action": action,
            }
        )
    return pd.DataFrame(rows)


def categorical_association_with_target(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Cramer's V between each categorical feature and the target."""
    rows = []
    for column in df.select_dtypes(exclude=np.number).columns:
        contingency = pd.crosstab(df[column].fillna("Missing"), df[TARGET])
        chi2, p_value, _, _ = chi2_contingency(contingency)
        n = contingency.to_numpy().sum()
        denominator = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * denominator)) if denominator > 0 else np.nan
        rows.append(
            {
                "feature": column,
                "cramers_v": float(cramers_v),
                "chi_square_p_value": float(p_value),
                "categories_including_missing": int(contingency.shape[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).round(6)


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Return high-level data-contract and class-balance metrics."""
    target = df[TARGET]
    return pd.DataFrame(
        {
            "metric": [
                "observations",
                "attributes",
                "numeric_attributes",
                "categorical_attributes",
                "exact_duplicates",
                "unique_ids",
                "default_cases",
                "non_default_cases",
                "default_rate_pct",
            ],
            "value": [
                len(df),
                df.shape[1],
                df.select_dtypes(include=np.number).shape[1],
                df.select_dtypes(exclude=np.number).shape[1],
                int(df.duplicated().sum()),
                int(df["id"].nunique()),
                int((target == 1).sum()),
                int((target == 0).sum()),
                round(float(target.mean() * 100), 3),
            ],
        }
    )


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Profile type, completeness, cardinality, and skewness by column."""
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean() * 100), 3),
                "unique_including_missing": int(series.nunique(dropna=False)),
                "skewness": (
                    round(float(series.skew()), 3)
                    if pd.api.types.is_numeric_dtype(series)
                    and series.nunique(dropna=True) > 2
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["missing_pct", "feature"], ascending=[False, True]
    )


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics with robust percentiles."""
    excluded = [TARGET, *IDENTIFIER_COLUMNS, *CONSTANT_COLUMNS]
    numeric = df.select_dtypes(include=np.number).drop(columns=excluded, errors="ignore")
    result = numeric.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    result["missing_pct"] = numeric.isna().mean() * 100
    result["skewness"] = numeric.skew()
    return result.round(3)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return frequency and target-rate evidence for every category."""
    rows = []
    for column in df.select_dtypes(exclude=np.number).columns:
        grouped = (
            df.groupby(column, dropna=False, observed=False)[TARGET]
            .agg(count="size", default_rate="mean")
            .reset_index()
        )
        grouped["feature"] = column
        grouped = grouped.rename(columns={column: "category"})
        grouped["category"] = grouped["category"].astype(str)
        rows.append(grouped[["feature", "category", "count", "default_rate"]])
    result = pd.concat(rows, ignore_index=True)
    result["default_rate_pct"] = result.pop("default_rate") * 100
    return result.sort_values(["feature", "count"], ascending=[True, False]).round(3)


def missingness_target_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Measure how strongly feature missingness is associated with the target."""
    rows = []
    for column in df.columns[df.isna().any()]:
        is_missing = df[column].isna()
        present_rate = df.loc[~is_missing, TARGET].mean()
        missing_rate = df.loc[is_missing, TARGET].mean()
        rows.append(
            {
                "feature": column,
                "missing_count": int(is_missing.sum()),
                "missing_pct": float(is_missing.mean() * 100),
                "default_rate_when_present_pct": float(present_rate * 100),
                "default_rate_when_missing_pct": float(missing_rate * 100),
                "absolute_rate_gap_pp": float(abs(missing_rate - present_rate) * 100),
                "perfect_separation_warning": bool(
                    present_rate in {0.0, 1.0} or missing_rate in {0.0, 1.0}
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "absolute_rate_gap_pp", ascending=False
    ).round(3)


def iqr_outlier_summary(
    df: pd.DataFrame, exclude: Iterable[str] = (TARGET, "id", "year")
) -> pd.DataFrame:
    """Flag potential numeric outliers without deleting them."""
    rows = []
    numeric_columns = df.select_dtypes(include=np.number).columns.difference(exclude)
    for column in numeric_columns:
        series = df[column].dropna()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flagged = (series < lower) | (series > upper)
        rows.append(
            {
                "feature": column,
                "iqr_lower_bound": lower,
                "iqr_upper_bound": upper,
                "flagged_count": int(flagged.sum()),
                "flagged_pct_of_non_missing": float(flagged.mean() * 100),
                "observed_min": series.min(),
                "observed_max": series.max(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "flagged_pct_of_non_missing", ascending=False
    ).round(3)


def feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return conservative baseline, numeric, and categorical feature lists."""
    excluded = set(IDENTIFIER_COLUMNS + CONSTANT_COLUMNS + [TARGET] + LEAKAGE_QUARANTINE)
    baseline_features = [column for column in df.columns if column not in excluded]
    numeric_features = [
        column
        for column in baseline_features
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_features = [
        column for column in baseline_features if column not in numeric_features
    ]
    return baseline_features, numeric_features, categorical_features


def build_unfitted_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Define, but do not fit, the later training-fold preprocessing pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def save_table(table: pd.DataFrame, path: str | Path) -> Path:
    """Save a table with stable UTF-8 CSV settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8")
    return path


def configure_plots() -> None:
    """Apply one restrained visual theme across all figures."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "figure.dpi": 120,
        }
    )


def save_figure(path: str | Path) -> Path:
    """Save the active figure and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path
