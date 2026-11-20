# Loan default prediction with age and gender

The main notebooks 01–08 now use 22 original predictors, including ordinal age and nominal one-hot gender. Age blanks use the training-fold most frequent band without a missingness indicator. All four gender categories and all 148,670 applications are retained. Other feature exclusions remain in src/data_preparation.py.

Run `python -m unittest discover -s tests -v` and `python run_notebooks.py --start 1 --end 8`. Each notebook runs in a fresh kernel and saves its actual outputs. GridSearchCV tunes Logistic Regression, CatBoost and XGBoost. Equal voting, Optuna-weighted soft voting and OOF stacking combine the two strongest development models. The final ensemble and maximum-F1 cutoff are chosen on development data.

Main results are under outputs/tables/part_d through part_h, with execution evidence under outputs/tables/execution. The final configuration and recommendation are in outputs/tables/part_h/frozen_configuration.json and recommendation.json. Demographic subgroup errors are exported beside them.

Model evaluation remains internal because the reserved split was inspected during earlier revisions. Feature timing and subgroup performance require verification before any operational use.

## Completed integrated run

Notebooks 04–07 completed in fresh kernels after replacing Random Forest with CatBoost, and all 5 integration tests passed. Cleaning, EDA, feature selection, encoding, the stratified split, CV folds and the MLP training code are unchanged. Notebook 08 remains supplementary; rerun it after Notebook 06 to refresh its cleared ensemble comparison.

The final model is Weighted soft voting using XGBoost and CatBoost, with maximum-F1 cutoff 0.28709111. Reserved accuracy is 77.45%, precision 53.97%, recall 57.85%, F1 55.84%, AP 0.6359 and ROC AUC 0.7821. Use `model.predict_proba(X)[:, 1] >= threshold` with the frozen cutoff.

## Reproduce on Windows

Use Python 3.12. Create and activate a virtual environment, install the tested dependency versions, then execute the checks and notebooks:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_notebooks.py --start 1 --end 8
```

The reserved split has been inspected in earlier revisions; these results remain internal validation. Older guides and archived outputs describe their historical versions. The report was not edited. Update its model names and affected numbers using the current 04–07 outputs.
