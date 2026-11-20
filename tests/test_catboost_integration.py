"""Checks for the CatBoost replacement and training-only voting weights."""
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.model_selection import StratifiedKFold

from src.model_ensemble import OOFWeightedVotingClassifier, configure_members, make_ensemble
from src.model_tuning import MODEL_NAMES, make_pipeline


class AuditedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, direction=1):
        self.direction = direction

    def fit(self, X, y):
        self.fit_rows_ = set(X.index)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        if self.fit_rows_.intersection(X.index):
            raise AssertionError('OOF prediction included a training row.')
        p = 1 / (1 + np.exp(-self.direction * X.value.to_numpy()))
        return np.column_stack([1-p, p])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class CatBoostIntegrationTests(unittest.TestCase):
    def setUp(self):
        configure_members(['XGBoost', 'CatBoost'])

    def test_three_supported_models(self):
        self.assertEqual(set(MODEL_NAMES), {'Logistic Regression', 'CatBoost', 'XGBoost'})

    def test_catboost_balanced_pipeline_and_clone(self):
        X = pd.DataFrame({'income': [1200, 1800, np.nan, 4200] * 8,
                          'loan_amount': np.arange(32) + 100,
                          'gender': ['Male', 'Female'] * 16,
                          'age': ['25-34', '35-44'] * 16})
        y = pd.Series([0, 0, 0, 1] * 8)
        model = make_pipeline('CatBoost', {'balanced': True, 'model__iterations': 5}, workers=1)
        self.assertTrue(is_classifier(model))
        fitted = clone(model).fit(X.iloc[:24], y.iloc[:24])
        p = fitted.predict_proba(X.iloc[24:])
        self.assertEqual(p.shape, (8, 2))
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(axis=1), 1)
        self.assertEqual(fitted.class_counts_, {0: 18, 1: 6})

    def test_weight_search_uses_oof_rows(self):
        X = pd.DataFrame({'value': np.linspace(-3, 3, 60)})
        y = (X.value > 0).astype(int)
        model = OOFWeightedVotingClassifier(
            [('good', AuditedClassifier(1)), ('bad', AuditedClassifier(-1))], n_trials=5)
        model.fit(X, y)
        self.assertGreater(model.weights_[0], .5)
        self.assertAlmostEqual(float(model.weights_.sum()), 1)
        p = model.predict_proba(pd.DataFrame({'value': [-1, 1]}, index=[100, 101]))
        self.assertLess(p[0, 1], p[1, 1])

    def test_weight_index_and_seed_are_reproducible(self):
        X = pd.DataFrame({'value': np.linspace(-3, 3, 60)})
        y = (X.value > 0).astype(int)
        a = OOFWeightedVotingClassifier(
            [('bad', AuditedClassifier(-1)), ('good', AuditedClassifier(1))], weight_index=1, n_trials=5)
        first, second = clone(a).fit(X, y), clone(a).fit(X, y)
        self.assertGreater(first.weights_[1], .5)
        np.testing.assert_array_equal(first.weights_, second.weights_)

    def test_stacking_keeps_oof_training(self):
        stack = make_ensemble('Stacking', {'XGBoost': {}, 'CatBoost': {}})
        self.assertIsInstance(stack.cv, StratifiedKFold)
        self.assertEqual(stack.cv.n_splits, 3)
        self.assertEqual(stack.stack_method, 'predict_proba')
        self.assertFalse(stack.passthrough)


if __name__ == '__main__':
    unittest.main()
