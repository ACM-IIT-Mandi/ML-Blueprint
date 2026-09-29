"""Gaussian Naive Bayes correctness and integration checks."""

import numpy as np
import pytest

from mlblueprint.core.base import NotFittedError
from mlblueprint.core.viz import discover_panels
from mlblueprint.probabilistic import GaussianNB, GaussianNBScratch


def test_hand_worked_prior_and_means():
    """Check known means and priors."""
    X = [[0.0], [2.0], [8.0], [10.0]]
    y = ["a", "a", "b", "b"]
    model = GaussianNB(var_smoothing=0.01).fit(X, y)
    np.testing.assert_allclose(model.theta_, [[1], [9]])
    np.testing.assert_allclose(model.class_prior_, [0.5, 0.5])
    assert model.predict([[0.5], [9.5]]).tolist() == ["a", "b"]
    np.testing.assert_allclose(model.predict_proba([[0.5], [9.5]]).sum(axis=1), 1)


def test_scratch_matches_numpy_and_normalizes_extreme_scores():
    """Compare both variants on extreme queries."""
    X = [[-100.0, 0.0], [-99.0, 0.2], [100.0, 1.0], [101.0, 1.2]]
    y = ["left", "left", "right", "right"]
    query = [[-10000.0, 2.0], [10000.0, -2.0]]
    plain = GaussianNBScratch().fit(X, y)
    fast = GaussianNB().fit(X, y)
    np.testing.assert_allclose(
        plain.predict_proba(query), fast.predict_proba(query), atol=1e-10
    )
    np.testing.assert_allclose(np.sum(plain.predict_proba(query), axis=1), 1)


def test_matches_sklearn_gaussian_nb():
    """Compare fitted parameters and posterior probabilities."""
    sklearn = pytest.importorskip("sklearn.naive_bayes")
    rng = np.random.default_rng(12)
    X = rng.normal(size=(90, 3))
    y = np.repeat(["a", "b", "c"], 30)
    query = rng.normal(size=(12, 3))
    ours = GaussianNB(var_smoothing=1e-5).fit(X, y)
    theirs = sklearn.GaussianNB(var_smoothing=1e-5).fit(X, y)
    np.testing.assert_allclose(ours.theta_, theirs.theta_)
    np.testing.assert_allclose(ours.var_, theirs.var_)
    np.testing.assert_allclose(
        ours.predict_proba(query), theirs.predict_proba(query), rtol=1e-10, atol=1e-12
    )


def test_scratch_gaussian_imbalanced_many_classes_matches_sklearn():
    """Exercise priors, parameters, and posteriors beyond a balanced toy case."""
    sklearn = pytest.importorskip("sklearn.naive_bayes")
    rng = np.random.default_rng(27)
    y = np.concatenate([np.full(size, label) for label, size in enumerate(range(2, 9))])
    X = rng.normal(size=(len(y), 4)) + y[:, None] * 0.25
    query = rng.normal(size=(10, 4))
    ours = GaussianNBScratch(var_smoothing=1e-6).fit(X.tolist(), y.tolist())
    theirs = sklearn.GaussianNB(var_smoothing=1e-6).fit(X, y)
    np.testing.assert_allclose(ours.class_prior_, theirs.class_prior_)
    np.testing.assert_allclose(ours.theta_, theirs.theta_)
    np.testing.assert_allclose(ours.var_, theirs.var_)
    np.testing.assert_allclose(
        ours.predict_proba(query.tolist()), theirs.predict_proba(query), rtol=1e-10
    )


@pytest.mark.parametrize("factory", [GaussianNB, GaussianNBScratch])
def test_bad_inputs_and_unfitted(factory):
    """Reject invalid input and premature prediction."""
    model = factory()
    with pytest.raises(NotFittedError):
        model.predict([[1.0]])
    with pytest.raises(ValueError):
        model.fit([], [])
    with pytest.raises(ValueError):
        model.fit([[1.0]], [])
    with pytest.raises(ValueError):
        factory(var_smoothing=-1).fit([[1.0]], [0])
    with pytest.raises(ValueError):
        model.fit([[float("nan")]], [0])
    model.fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError):
        model.predict([[1.0, 2.0]])


def test_visualizer_panel_discovered():
    """Exercise automatic panel registration."""
    pytest.importorskip("matplotlib")
    panels = discover_panels()
    panel = panels["probabilistic.GaussianNBPanel"]()
    frames = panel.frames()
    assert len(frames) == 1
    assert 0 <= frames[0].metrics["train accuracy"] <= 1


@pytest.mark.parametrize("factory", [GaussianNB, GaussianNBScratch])
def test_all_constant_features_cannot_define_a_gaussian(factory):
    """Smoothing based on overall variance is zero when all data are constant."""
    model = factory()
    with pytest.raises(ValueError, match="constant features"):
        model.fit([[2.0], [2.0]], [0, 1])
    with pytest.raises(NotFittedError):
        model.predict([[2.0]])
