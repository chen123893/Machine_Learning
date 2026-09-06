# Loan Default Prediction - Cleaning and Exploratory Data Analysis

This repository contains the first, pre-modelling phase of the 6006CEM group project. It uses the public [Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset) (CC0) and deliberately stops before train/test splitting, fitted preprocessing, model training, model testing, or hyperparameter optimisation.


## Important audit finding

The dataset contains severe target-leakage warning signs. 


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

