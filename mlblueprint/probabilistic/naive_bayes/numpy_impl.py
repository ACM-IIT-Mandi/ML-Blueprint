"""Vectorized Gaussian, Categorical, and Multinomial Naive Bayes."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mlblueprint.core.base import Predictor
from mlblueprint.core.validation import check_array, check_X_y


class GaussianNB(Predictor):
    """Gaussian class likelihoods with feature independence and smoothing."""

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing

    def fit(self, X: ArrayLike, y: ArrayLike) -> GaussianNB:
        """Learn class frequencies and per-class feature distributions."""
        if not np.isfinite(self.var_smoothing) or self.var_smoothing < 0:
            raise ValueError("var_smoothing must be finite and nonnegative")
        X, y = check_X_y(X, y, y_numeric=False)
        classes, counts = np.unique(y, return_counts=True)
        class_prior = counts / len(y)
        theta = np.vstack([X[y == label].mean(axis=0) for label in classes])
        epsilon = self.var_smoothing * np.var(X, axis=0).max()
        variance = np.vstack([X[y == label].var(axis=0) for label in classes]) + epsilon
        if np.any(variance <= 0):
            raise ValueError(
                "Zero variance: remove constant features or use "
                "training data with variation."
            )
        self.classes_ = classes
        self.class_prior_ = class_prior
        self.theta_ = theta
        self.var_ = variance
        self.n_features_in_ = X.shape[1]
        return self

    def predict_log_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Compute normalized posterior log probabilities in ``classes_`` order."""
        self._check_is_fitted()
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than training data")
        scores = (
            np.log(self.class_prior_)[None, :]
            - 0.5 * np.log(2 * np.pi * self.var_).sum(axis=1)[None, :]
            - 0.5 * (((X[:, None, :] - self.theta_) ** 2) / self.var_).sum(axis=2)
        )
        peak = scores.max(axis=1, keepdims=True)
        return scores - (
            peak + np.log(np.exp(scores - peak).sum(axis=1, keepdims=True))
        )

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return posterior probabilities in ``classes_`` order."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict the class of largest posterior probability."""
        scores = self.predict_log_proba(X)
        return self.classes_[scores.argmax(axis=1)]


def _check_alpha(alpha: float) -> None:
    """Require positive Laplace smoothing for discrete models."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and greater than zero")


def _normalize_log_scores(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize each row of log scores with log-sum-exp."""
    peak = scores.max(axis=1, keepdims=True)
    return scores - (peak + np.log(np.exp(scores - peak).sum(axis=1, keepdims=True)))


class CategoricalNB(Predictor):
    """Naive Bayes for nonnegative integer-coded categorical features."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: ArrayLike, y: ArrayLike) -> CategoricalNB:
        """Count per-class feature categories with Laplace smoothing."""
        _check_alpha(self.alpha)
        X, y = check_X_y(X, y, y_numeric=False)
        if np.any(X < 0) or np.any(X != np.floor(X)):
            raise ValueError("X must contain nonnegative integer category codes")
        X = X.astype(np.int64)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(counts / len(y))
        self.n_features_in_ = X.shape[1]
        self.n_categories_ = X.max(axis=0) + 1
        self.category_count_ = []
        self.category_log_prob_ = []
        for j, size in enumerate(self.n_categories_):
            feature_counts = np.vstack(
                [
                    np.bincount(X[y == label, j], minlength=size)
                    for label in self.classes_
                ]
            )
            self.category_count_.append(feature_counts)
            self.category_log_prob_.append(
                np.log(
                    (feature_counts + self.alpha)
                    / (counts[:, None] + self.alpha * size)
                )
            )
        return self

    def predict_log_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return normalized posterior log probabilities in ``classes_`` order."""
        self._check_is_fitted()
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than training data")
        if np.any(X < 0) or np.any(X != np.floor(X)) or np.any(X >= self.n_categories_):
            raise ValueError("X contains an invalid or unseen category code")
        X = X.astype(np.int64)
        scores = np.broadcast_to(
            self.class_log_prior_, (len(X), len(self.classes_))
        ).copy()
        for j, log_probs in enumerate(self.category_log_prob_):
            scores += log_probs[:, X[:, j]].T
        return _normalize_log_scores(scores)

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return posterior class probabilities."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict the most likely class."""
        scores = self.predict_log_proba(X)
        return self.classes_[scores.argmax(axis=1)]


class MultinomialNB(Predictor):
    """Naive Bayes for nonnegative word or event counts."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: ArrayLike, y: ArrayLike) -> MultinomialNB:
        """Sum feature counts for each class and apply Laplace smoothing."""
        _check_alpha(self.alpha)
        X, y = check_X_y(X, y, y_numeric=False)
        if np.any(X < 0):
            raise ValueError("X must contain nonnegative feature counts")
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(counts / len(y))
        self.feature_count_ = np.vstack(
            [X[y == label].sum(axis=0) for label in self.classes_]
        )
        smoothed = self.feature_count_ + self.alpha
        self.feature_log_prob_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        self.n_features_in_ = X.shape[1]
        return self

    def predict_log_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return normalized posterior log probabilities in ``classes_`` order."""
        self._check_is_fitted()
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than training data")
        if np.any(X < 0):
            raise ValueError("X must contain nonnegative feature counts")
        return _normalize_log_scores(
            X @ self.feature_log_prob_.T + self.class_log_prior_
        )

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return posterior class probabilities."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict the most likely class."""
        scores = self.predict_log_proba(X)
        return self.classes_[scores.argmax(axis=1)]
