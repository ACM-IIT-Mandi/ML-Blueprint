"""Tests for SVM classificationimplementation."""

import random

import numpy as np
from sklearn.svm import SVC

from mlblueprint.svm import SvmClassification, SvmClassificationScratch


class TestSVM:
    """Test cases for SVM."""

    def test_fit_simple(self):
        """Test that model can learn a linearly separable dataset."""
        np.random.seed(42)

        X = np.array([[1, 1], [2, 2], [2, 3], [6, 5], [7, 7], [8, 6]])
        y = np.array([-1, -1, -1, 1, 1, 1])

        model = SvmClassification()
        model.fit(X, y, lr=0.01, n_iters=2000, C=1.0)

        predictions = model.predict(X)
        assert np.mean(predictions == y) >= 0.9

    def test_scratch_vs_numpy(self):
        """Test that scratch and numpy implementations give similar predictions."""
        random.seed(42)
        np.random.seed(42)

        X = [[1, 1], [2, 2], [2, 3], [6, 5], [7, 7], [8, 6]]
        y = [-1, -1, -1, 1, 1, 1]

        model_scratch = SvmClassificationScratch()
        model_scratch.fit(X, y, lr=0.01, n_iters=2000, C=1.0)

        model_numpy = SvmClassification()
        model_numpy.fit(np.array(X), np.array(y), lr=0.01, n_iters=2000, C=1.0)

        pred_scratch = model_scratch.predict(X)
        pred_numpy = model_numpy.predict(X)

        assert np.array_equal(np.array(pred_scratch), pred_numpy)

    def test_against_sklearn(self):
        """Test that our implementation gives similar predictions to sklearn."""
        np.random.seed(42)
        random.seed(42)

        X = np.array([[1, 1], [2, 2], [2, 3], [6, 5], [7, 7], [8, 6]])
        y = np.array([-1, -1, -1, 1, 1, 1])

        our_model = SvmClassification()
        our_model.fit(X, y, lr=0.01, n_iters=2000, C=1.0)

        sklearn_model = SVC(kernel="linear", C=1.0)
        sklearn_model.fit(X, y)

        our_preds = our_model.predict(X)
        sklearn_preds = sklearn_model.predict(X)

        assert np.mean(our_preds == sklearn_preds) >= 0.9


test = TestSVM()
test.test_fit_simple()
test.test_scratch_vs_numpy()
test.test_against_sklearn()
