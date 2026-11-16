# Loan Default Prediction

6006CEM group project comparing Logistic Regression, Random Forest, XGBoost,
soft voting and stacking, with a supplementary deep MLP comparison.

## Current cleaning policy

1. Standardise names and category labels; remove exact duplicates if present.
2. Drop `rate_of_interest`, `interest_rate_spread` and `upfront_charges` BEFORE
   filtering incomplete records. Their raw missingness is outcome-linked. Every
   local default has missing interest-rate spread: `dropna()` first erases defaults.
3. Drop rows with any missing value in the retained columns.
4. Keep only Female/Male records, as requested; encode Female=0 and Male=1.

The cohort has 56,934 rows: 47,102 non-defaults and 9,832 defaults (17.269%).
It represents complete Female/Male applications, not every applicant. The raw
CSV remains unchanged. See `outputs/tables/cleaning/cohort_attrition.csv`.

Only `id`, constant `year`, the target and the three removed pricing fields are
excluded from predictors. All 28 remaining input fields are retained, including
gender, approval, property value, LTV, DTI, credit bureau, age and submission
channel. Proxy screening is audit-only. Field availability before approval remains
unverified: this is a retrospective internal experiment, not a lending service.

## Training and imbalance

Stratified 80:20 development/holdout split, seed 42. Numeric scaling and one-hot
encoding fit inside training folds; no imputation. Missing future inputs are
rejected. Raw/log income, PCA and balanced weights are compared on development
data. Nested grid search tests weighted and unweighted fits for every base model.
`BalancedTrainingPipeline` computes inverse-frequency weights separately for each
fit, including inner CV and stacking folds. Validation and holdout retain natural
prevalence; no SMOTE is used. Ensemble members are selected from development
rankings, never test scores.

## Reproduce

Create `.venv` with Python 3.12, then run in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/run_notebooks.py --start 1 --end 8
```

Open notebooks 01 through 08 in numerical order. The runner saves actual output
cells and logs completion under `outputs/tables/execution`. Training may take
several minutes. No API keys or cloud services are needed.

## Outputs and provenance

- `data/raw/Loan_Default.csv`: original unchanged source; see `data/README.md`.
- `data/processed/loan_default_cleaned.csv`: current filtered cohort.
- `src/`: reusable cleaning, preparation, validation, tuning and ensemble code.
- `outputs/tables/` and `outputs/figures/`: current experiment evidence.
- `outputs/reports/`: use the latest complete-case revision of the Word documents.
- `outputs/archive/before_complete_case_revision_20261116.zip`: previous source,
  notebooks, cleaned data, tables, figures and reports preserved for comparison.

The pricing exclusion was prompted by a Kaggle discussion excerpt supplied by the
project author. Notebook 01 and the report explain its reasoning and verify it
against local data. No author name or exact comment URL was supplied; neither is
invented. The excerpt contains inconsistent class-direction wording, so local
Status cross-tabs determine the reported direction. Report charts come directly
from executed notebooks.

### Validation-selected decision thresholds

Notebook 07 now selects a separate threshold for each fixed candidate by maximising F1 on five-fold stratified development out-of-fold probabilities. Preprocessing is fitted inside each fold; holdout labels are not used to choose thresholds. Exact ties select the higher threshold. F1 is a balanced precision/recall objective, not an estimated financial cost. Development scores are tuning scores, and the repeatedly inspected reserved split remains internal validation.

The notebook saves `outputs/tables/part_h/selected_thresholds.json`, `validation_thresholds.csv` and `threshold_oof_predictions.csv`. `holdout_metrics.csv` retains the original `_050` metrics and adds `_tuned` metrics. Use `probability >= thresholds[model_name]` for the tuned decision; the estimator's default `.predict()` still uses its original decision rule. Ranking metrics and top-10% review selection are unchanged. Existing Word documents describe the previous 0.50 results and have not been regenerated for this code-only update.
