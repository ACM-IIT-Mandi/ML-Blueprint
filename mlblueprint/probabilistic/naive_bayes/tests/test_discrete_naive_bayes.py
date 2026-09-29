"""Categorical and Multinomial Naive Bayes checks."""

import numpy as np
import pytest

from mlblueprint.core.base import NotFittedError
from mlblueprint.core.viz import discover_panels
from mlblueprint.probabilistic import (
    CategoricalNB,
    CategoricalNBScratch,
    MultinomialNB,
    MultinomialNBScratch,
)


def test_categorical_hand_count_and_prediction():
    """Each feature has separate category counts with one pseudocount."""
    X = [[0, 1], [0, 0], [1, 1], [1, 0]]
    y = ["A", "A", "B", "B"]
    model = CategoricalNB().fit(X, y)
    np.testing.assert_allclose(
        model.category_log_prob_[0], np.log([[0.75, 0.25], [0.25, 0.75]])
    )
    assert model.predict([[0, 1], [1, 0]]).tolist() == ["A", "B"]


def test_multinomial_hand_count_and_prediction():
    """A class with counts [3, 0] has smoothed feature probabilities [4/5, 1/5]."""
    X = [[2, 0], [1, 0], [0, 2], [0, 1]]
    y = ["A", "A", "B", "B"]
    model = MultinomialNB().fit(X, y)
    np.testing.assert_allclose(
        model.feature_log_prob_, np.log([[0.8, 0.2], [0.2, 0.8]])
    )
    assert model.predict([[2, 0], [0, 2]]).tolist() == ["A", "B"]


@pytest.mark.parametrize(
    ("fast", "plain", "reference", "X", "query"),
    [
        (
            CategoricalNB,
            CategoricalNBScratch,
            "CategoricalNB",
            [[0, 1], [1, 0], [0, 2], [2, 1], [1, 2], [2, 0]],
            [[0, 0], [1, 1], [2, 2]],
        ),
        (
            MultinomialNB,
            MultinomialNBScratch,
            "MultinomialNB",
            [[3, 0, 1], [2, 1, 0], [0, 3, 2], [1, 2, 3], [2, 0, 2], [0, 2, 1]],
            [[1, 0, 2], [0, 2, 0], [0, 0, 0]],
        ),
    ],
)
def test_both_versions_match_sklearn(fast, plain, reference, X, query):
    """Check parameters and posteriors against a production implementation."""
    sklearn = pytest.importorskip("sklearn.naive_bayes")
    y = ["A", "A", "B", "B", "A", "B"]
    a = fast(alpha=0.5).fit(X, y)
    b = plain(alpha=0.5).fit(X, y)
    c = getattr(sklearn, reference)(alpha=0.5).fit(X, y)
    np.testing.assert_allclose(a.predict_proba(query), c.predict_proba(query))
    np.testing.assert_allclose(b.predict_proba(query), c.predict_proba(query))
    if reference == "CategoricalNB":
        for ours, theirs in zip(a.category_log_prob_, c.feature_log_prob_, strict=True):
            np.testing.assert_allclose(ours, theirs)
    else:
        np.testing.assert_allclose(a.feature_log_prob_, c.feature_log_prob_)


def test_discrete_imbalanced_many_classes_matches_sklearn():
    """Check both discrete models on imbalanced multiclass random data."""
    sklearn = pytest.importorskip("sklearn.naive_bayes")
    rng = np.random.default_rng(31)
    y = np.concatenate([np.full(size, label) for label, size in enumerate(range(2, 8))])

    categorical_X = rng.integers(0, 4, size=(len(y), 3))
    categorical_query = rng.integers(0, 4, size=(9, 3))
    categorical_reference = sklearn.CategoricalNB(alpha=0.7).fit(categorical_X, y)
    for factory in (CategoricalNB, CategoricalNBScratch):
        model = factory(alpha=0.7).fit(categorical_X.tolist(), y.tolist())
        np.testing.assert_allclose(
            model.predict_proba(categorical_query.tolist()),
            categorical_reference.predict_proba(categorical_query),
        )

    multinomial_X = rng.uniform(0, 5, size=(len(y), 5))
    multinomial_query = rng.uniform(0, 5, size=(9, 5))
    multinomial_reference = sklearn.MultinomialNB(alpha=0.7).fit(multinomial_X, y)
    for factory in (MultinomialNB, MultinomialNBScratch):
        model = factory(alpha=0.7).fit(multinomial_X.tolist(), y.tolist())
        np.testing.assert_allclose(
            model.predict_proba(multinomial_query.tolist()),
            multinomial_reference.predict_proba(multinomial_query),
        )


@pytest.mark.parametrize("factory", [CategoricalNB, CategoricalNBScratch])
def test_categorical_rejects_invalid_or_unseen_codes(factory):
    """Integer-coded categories must be known for every feature."""
    with pytest.raises(NotFittedError):
        factory().predict([[0]])
    with pytest.raises(ValueError, match="nonnegative integer"):
        factory().fit([[0.5], [1]], [0, 1])
    model = factory().fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="unseen"):
        model.predict([[2]])


@pytest.mark.parametrize("factory", [CategoricalNB, CategoricalNBScratch])
def test_categorical_accepts_integer_valued_floats(factory):
    """Values such as 1.0 are valid integer category codes."""
    model = factory().fit([[0.0], [1.0], [1.0]], ["a", "b", "b"])
    assert list(model.predict([[1.0], [0.0]])) == ["b", "a"]


@pytest.mark.parametrize("factory", [MultinomialNB, MultinomialNBScratch])
def test_multinomial_rejects_negative_counts(factory):
    """Counts cannot be negative during fit or prediction."""
    with pytest.raises(NotFittedError):
        factory().predict([[0]])
    with pytest.raises(ValueError, match="nonnegative"):
        factory().fit([[-1], [1]], [0, 1])
    model = factory().fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="nonnegative"):
        model.predict([[-1]])


@pytest.mark.parametrize(
    "factory",
    [CategoricalNB, CategoricalNBScratch, MultinomialNB, MultinomialNBScratch],
)
def test_discrete_rejects_nonpositive_alpha_and_mismatched_shape(factory):
    """Smoothing and dimensions are part of each estimator contract."""
    with pytest.raises(ValueError, match="alpha"):
        factory(alpha=0).fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError):
        factory().fit([[0], [1]], [0])
    model = factory().fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError):
        model.predict([[0, 0]])


def test_discrete_visualizer_panels_show_changing_probabilities():
    """Each panel provides informative frames without running Streamlit."""
    pytest.importorskip("matplotlib")
    panels = discover_panels()
    categorical = panels["probabilistic.CategoricalNBPanel"]().frames()
    multinomial = panels["probabilistic.MultinomialNBPanel"]().frames()
    assert len(categorical) == 2
    assert len(multinomial) == 5
    assert categorical[0].metrics["P(A)"] != categorical[1].metrics["P(A)"]
    assert multinomial[0].metrics["P(spam)"] < multinomial[-1].metrics["P(spam)"]
