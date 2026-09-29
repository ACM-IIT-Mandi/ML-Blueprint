"""Gaussian, Categorical, and Multinomial Naive Bayes."""

from __future__ import annotations

import math

from mlblueprint.core.base import Predictor


def _validate_X(X, n_features=None):
    """Check rectangular finite numeric samples."""
    if not X or not X[0]:
        raise ValueError("X must contain samples and features")
    width = len(X[0]) if n_features is None else n_features
    for row in X:
        if len(row) != width or any(not math.isfinite(value) for value in row):
            raise ValueError("X must be rectangular, finite and match fitted features")
    return width


class GaussianNB(Predictor):
    """Classify continuous features with independent Gaussian densities.

    ``var_smoothing`` adds a fraction of the largest overall feature variance
    to every class variance, matching scikit-learn's GaussianNB convention.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing

    def fit(self, X: list[list[float]], y: list) -> GaussianNB:
        """Estimate class priors, means and population variances."""
        if not math.isfinite(self.var_smoothing) or self.var_smoothing < 0:
            raise ValueError("var_smoothing must be finite and nonnegative")
        width = _validate_X(X)
        if len(y) != len(X):
            raise ValueError("X and y must have the same length")
        try:
            classes = sorted(set(y))
        except (TypeError, ValueError) as exc:
            raise ValueError("labels must be hashable and sortable") from exc
        overall_mean = [sum(row[j] for row in X) / len(X) for j in range(width)]
        overall_var = [
            sum((row[j] - overall_mean[j]) ** 2 for row in X) / len(X)
            for j in range(width)
        ]
        epsilon = self.var_smoothing * max(overall_var)
        class_prior = []
        theta = []
        variances = []
        for label in classes:
            rows = [row for row, target in zip(X, y, strict=True) if target == label]
            mean = [sum(row[j] for row in rows) / len(rows) for j in range(width)]
            variance = [
                sum((row[j] - mean[j]) ** 2 for row in rows) / len(rows) + epsilon
                for j in range(width)
            ]
            theta.append(mean)
            variances.append(variance)
            class_prior.append(len(rows) / len(X))
        if any(var <= 0 for class_vars in variances for var in class_vars):
            raise ValueError(
                "Zero variance: remove constant features or use "
                "training data with variation."
            )
        self.classes_ = classes
        self.class_prior_ = class_prior
        self.theta_ = theta
        self.var_ = variances
        self.n_features_in_ = width
        return self

    def predict_log_proba(self, X: list[list[float]]) -> list[list[float]]:
        """Return normalized class log probabilities for each row."""
        self._check_is_fitted()
        _validate_X(X, self.n_features_in_)
        results = []
        for row in X:
            scores = []
            for prior, mean, variance in zip(
                self.class_prior_, self.theta_, self.var_, strict=True
            ):
                score = math.log(prior)
                for value, mu, var in zip(row, mean, variance, strict=True):
                    score -= 0.5 * (
                        math.log(2 * math.pi * var) + (value - mu) ** 2 / var
                    )
                scores.append(score)
            peak = max(scores)
            normalizer = peak + math.log(sum(math.exp(s - peak) for s in scores))
            results.append([s - normalizer for s in scores])
        return results

    def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
        """Return probabilities in ``classes_`` order."""
        return [[math.exp(s) for s in row] for row in self.predict_log_proba(X)]

    def predict(self, X: list[list[float]]) -> list:
        """Return the highest posterior class for each sample."""
        predictions = []
        for scores in self.predict_log_proba(X):
            best_class = max(range(len(scores)), key=lambda index: scores[index])
            predictions.append(self.classes_[best_class])
        return predictions


def _check_alpha(alpha: float) -> None:
    """Require positive Laplace smoothing for discrete models."""
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and greater than zero")


def _log_probabilities(scores: list[float]) -> list[float]:
    """Normalize log scores without exponentiating a positive number."""
    peak = max(scores)
    normalizer = peak + math.log(sum(math.exp(score - peak) for score in scores))
    return [score - normalizer for score in scores]


class CategoricalNB(Predictor):
    """Naive Bayes for integer-coded categorical features.

    Each feature has its own category set. Categories must be nonnegative
    integers, numbered from zero. ``alpha`` adds a count to every category.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: list[list[int]], y: list) -> CategoricalNB:
        """Count each category within every class, then apply smoothing."""
        _check_alpha(self.alpha)
        width = _validate_X(X)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if any(value < 0 or int(value) != value for row in X for value in row):
            raise ValueError("X must contain nonnegative integer category codes")
        X = [[int(value) for value in row] for row in X]
        self.classes_ = sorted(set(y))
        self.n_features_in_ = width
        self.n_categories_ = [max(row[j] for row in X) + 1 for j in range(width)]
        self.class_log_prior_ = []
        self.category_count_ = []
        self.category_log_prob_ = []
        for label in self.classes_:
            rows = [row for row, target in zip(X, y, strict=True) if target == label]
            self.class_log_prior_.append(math.log(len(rows) / len(X)))
            feature_counts = [[0] * size for size in self.n_categories_]
            for row in rows:
                for j, category in enumerate(row):
                    feature_counts[j][int(category)] += 1
            self.category_count_.append(feature_counts)
            self.category_log_prob_.append(
                [
                    [
                        math.log((count + self.alpha) / (len(rows) + self.alpha * size))
                        for count in counts
                    ]
                    for counts, size in zip(
                        feature_counts, self.n_categories_, strict=True
                    )
                ]
            )
        return self

    def predict_log_proba(self, X: list[list[int]]) -> list[list[float]]:
        """Return normalized class log probabilities in ``classes_`` order."""
        self._check_is_fitted()
        _validate_X(X, self.n_features_in_)
        results = []
        for row in X:
            if any(
                value < 0 or int(value) != value or value >= size
                for value, size in zip(row, self.n_categories_, strict=True)
            ):
                raise ValueError("X contains an invalid or unseen category code")
            scores = [
                prior + sum(feature[j][int(category)] for j, category in enumerate(row))
                for prior, feature in zip(
                    self.class_log_prior_, self.category_log_prob_, strict=True
                )
            ]
            results.append(_log_probabilities(scores))
        return results

    def predict_proba(self, X: list[list[int]]) -> list[list[float]]:
        """Return posterior class probabilities."""
        return [[math.exp(score) for score in row] for row in self.predict_log_proba(X)]

    def predict(self, X: list[list[int]]) -> list:
        """Predict the class with the highest posterior score."""
        return [
            self.classes_[max(range(len(scores)), key=lambda i: scores[i])]
            for scores in self.predict_log_proba(X)
        ]


class MultinomialNB(Predictor):
    """Naive Bayes for nonnegative feature counts, such as word counts."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: list[list[float]], y: list) -> MultinomialNB:
        """Estimate smoothed feature frequencies for each class."""
        _check_alpha(self.alpha)
        width = _validate_X(X)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if any(value < 0 for row in X for value in row):
            raise ValueError("X must contain nonnegative feature counts")
        self.classes_ = sorted(set(y))
        self.n_features_in_ = width
        self.class_log_prior_ = []
        self.feature_count_ = []
        self.feature_log_prob_ = []
        for label in self.classes_:
            rows = [row for row, target in zip(X, y, strict=True) if target == label]
            counts = [sum(row[j] for row in rows) for j in range(width)]
            total = sum(counts) + self.alpha * width
            self.class_log_prior_.append(math.log(len(rows) / len(X)))
            self.feature_count_.append(counts)
            self.feature_log_prob_.append(
                [math.log((count + self.alpha) / total) for count in counts]
            )
        return self

    def predict_log_proba(self, X: list[list[float]]) -> list[list[float]]:
        """Return normalized class log probabilities in ``classes_`` order."""
        self._check_is_fitted()
        _validate_X(X, self.n_features_in_)
        if any(value < 0 for row in X for value in row):
            raise ValueError("X must contain nonnegative feature counts")
        return [
            _log_probabilities(
                [
                    prior
                    + sum(
                        value * log_p for value, log_p in zip(row, probs, strict=True)
                    )
                    for prior, probs in zip(
                        self.class_log_prior_, self.feature_log_prob_, strict=True
                    )
                ]
            )
            for row in X
        ]

    def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
        """Return posterior class probabilities."""
        return [[math.exp(score) for score in row] for row in self.predict_log_proba(X)]

    def predict(self, X: list[list[float]]) -> list:
        """Predict the class with the highest posterior score."""
        return [
            self.classes_[max(range(len(scores)), key=lambda i: scores[i])]
            for scores in self.predict_log_proba(X)
        ]
