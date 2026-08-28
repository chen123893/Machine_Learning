# Loan Default Prediction - Cleaning and Exploratory Data Analysis

This repository contains the first, pre-modelling phase of the 6006CEM group project. It uses the public [Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset) (CC0) and deliberately stops before train/test splitting, fitted preprocessing, model training, model testing, or hyperparameter optimisation.

## Current scope

- `01_data_cleaning.ipynb`: raw-data contract, quality audit, deterministic cleaning, before/after evidence, validation, and cleaned-data export
- `02_exploratory_data_analysis.ipynb`: feature descriptions, descriptive statistics, distributions, skewness, missing values, duplicates, outliers, class balance, noise, inconsistencies, numeric correlation, categorical association, leakage warnings, and performance-impact discussion
- No train/test split, learned preprocessing, estimator, or model test is run in this phase
- Part D model-dependent preprocessing remains a later phase because imputation, encoding, scaling, feature selection, PCA, and SMOTE must be evaluated without contaminating the test data

## Important audit finding

The dataset contains severe target-leakage warning signs. For example, `Interest_rate_spread` is missing for every default case and present for every non-default case. Other variables also have missingness or categories that nearly determine `Status`. A high-scoring submission should show this evidence, explain why random imputation would not solve it, and quarantine the suspicious fields until their collection time is verified.

## Run in Visual Studio Code

1. Open this folder in VS Code.
2. Open a terminal and create the environment:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   .venv/bin/python -m ipykernel install --user --name loan-default-6006cem --display-name "Python (Loan Default 6006CEM)"
   ```

3. Open `notebooks/01_data_cleaning.ipynb`.
4. Select the project Python kernel and choose **Run All**. Confirm every validation check passes and that `data/processed/loan_default_cleaned.csv` is created.
5. Open `notebooks/02_exploratory_data_analysis.ipynb` and choose **Run All**.
6. Stop after Notebook 02. Do not run modelling or testing yet.

Notebook 01 writes cleaning evidence to `outputs/tables/cleaning/`. Notebook 02 writes EDA evidence to `outputs/tables/eda/` and `outputs/figures/eda/`.

## Project structure

```text
.
|-- data/
|   |-- raw/Loan_Default.csv
|   `-- processed/                  # generated cleaned data
|-- notebooks/
|   |-- 01_data_cleaning.ipynb
|   `-- 02_exploratory_data_analysis.ipynb
|-- outputs/
|   |-- figures/eda/                # generated EDA figures
|   `-- tables/
|       |-- cleaning/               # cleaning audit and validation evidence
|       `-- eda/                    # EDA evidence tables
|-- src/eda_preprocessing.py
|-- requirements.txt
`-- README.md
```

## Methodological decisions

- `Status` is treated as a labelled binary target, so the appropriate primary task is supervised binary classification.
- `ID` is an identifier and `year` is constant; neither should enter a model.
- Exact duplicates are checked before any removal. Duplicate removal is conditional, not automatic.
- Category labels are stripped and a few spelling/case inconsistencies are standardised; raw data is never overwritten.
- Statistical imputation, encoding, scaling, feature selection, PCA, and resampling are not executed in these notebooks. They must later be fitted only on training folds to prevent leakage.
- SMOTE is not applied during EDA. If later justified experimentally, it belongs inside the training-only cross-validation pipeline, never before splitting.
- PCA is not automatically appropriate for this mixed-type dataset and would reduce interpretability. It should only be tested later against a non-PCA baseline.
- Outliers are documented rather than silently deleted. Values such as very high LTV require domain validation before treatment.

## Distinction checklist for this phase

- [x] Public, real-world dataset with 148,670 observations and 34 attributes
- [x] Clear business and ML objectives
- [x] Classification choice justified against regression and clustering
- [x] Full data-quality audit with numerical evidence and visualisations
- [x] Leakage and collection-timestamp risks explicitly investigated
- [x] Reproducible, modular cleaning and EDA workflow
- [x] Each cleaning action includes evidence, a reason, and a limitation
- [x] EDA issues are connected to expected model impact and later treatment
- [x] Raw data preserved; generated outputs separated
- [ ] Complete Part B's critical literature comparison in the final report
- [ ] Design and evaluate Part D preprocessing during the later modelling phase
- [ ] Add APA 7 citations to the final report in your own words
- [ ] Record each group member's contribution and keep Git history
- [ ] Declare AI assistance as required by the brief

## Academic integrity

Use this project as an auditable starting point. Interpret every result yourself, retain evidence of your own development, and write the submitted discussion in your group's own voice. The assignment brief requires AI-tool use to be declared and limits AI-generated report content.
