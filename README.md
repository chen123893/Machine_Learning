# Loan default prediction with age and gender

The main notebooks 01–08 now use 22 original predictors, including ordinal age and nominal one-hot gender. Age blanks use the training-fold most frequent band without a missingness indicator. All four gender categories and all 148,670 applications are retained. Other feature exclusions remain in src/data_preparation.py.

Run `python -m unittest discover -s tests -v` and `python run_notebooks.py --start 1 --end 8`. Each notebook runs in a fresh kernel and saves its actual outputs. Nested search retunes the models for the new features; ensembles and F1 thresholds are then regenerated. The project choice is XGBoost with a development-selected maximum-F1 cutoff. Stacking and voting remain comparison models.

Main results are under outputs/tables/part_d through part_h, with execution evidence under outputs/tables/execution. The final configuration and recommendation are in outputs/tables/part_h/frozen_configuration.json and recommendation.json. Demographic subgroup errors are exported beside them.

Model evaluation remains internal because the reserved split was inspected during earlier revisions. Feature timing and subgroup performance require verification before any operational use.

## Completed integrated run

All 8 main notebooks completed in fresh kernels (52 executed code cells), and all 13 tests passed. Run notebooks 01-08 to regenerate the tables, figures and processed data in this checkout. Existing generated outputs may describe an earlier run until regenerated.

The final XGBoost F1 cutoff is 0.27655298. Reserved accuracy is 76.88%, precision 52.74%, recall 59.43%, F1 55.89%, AP 0.6343 and ROC AUC 0.7812. Use `model.predict_proba(X)[:, 1] >= threshold` with the frozen cutoff; the estimator's ordinary `.predict()` retains its default rule.

## Reproduce on Windows

Use Python 3.12. Create and activate a virtual environment, install the tested dependency versions, then execute the checks and notebooks:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
python run_notebooks.py --start 1 --end 8
```

The reserved split has been inspected in earlier revisions; these results remain internal validation. Older guides and archived outputs describe their historical versions. The integrated report and main 01–08 outputs describe the current version.
