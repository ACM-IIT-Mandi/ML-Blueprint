"""Visualize Gaussian, Categorical, and Multinomial Naive Bayes."""

from __future__ import annotations

import numpy as np

from mlblueprint.core.datasets import make_blobs
from mlblueprint.core.viz import Frame, Parameter, VizPanel, register_panel

from .numpy_impl import GaussianNB


@register_panel
class GaussianNBPanel(VizPanel):
    """Show how class overlap and variance smoothing alter the boundary."""

    name = "Gaussian Naive Bayes — probability landscape"
    family = "probabilistic"
    description = "Compare the fitted Gaussian centers with the posterior field."

    def parameters(self) -> list[Parameter]:
        """Expose data overlap and variance smoothing."""
        return [
            Parameter(
                "cluster_std", "Class overlap", 1.5, "float", min=0.5, max=4.0, step=0.1
            ),
            Parameter(
                "var_smoothing",
                "Variance smoothing",
                0.01,
                "float",
                min=0.0,
                max=1.0,
                step=0.01,
            ),
        ]

    def frames(
        self, cluster_std: float = 1.5, var_smoothing: float = 0.01
    ) -> list[Frame]:
        """Draw one deterministic two-class fitted probability field."""
        X, y = make_blobs(
            n_samples=160, centers=2, cluster_std=cluster_std, random_state=42
        )
        model = GaussianNB(var_smoothing=var_smoothing).fit(X, y)
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 120),
            np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 120),
        )
        probability = model.predict_proba(np.column_stack((xx.ravel(), yy.ravel())))[
            :, 1
        ].reshape(xx.shape)

        def draw(ax):
            ax.contourf(xx, yy, probability, levels=20, cmap="RdBu", alpha=0.6)
            ax.contour(xx, yy, probability, levels=[0.5], colors="black")
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="black", s=22)
            ax.scatter(
                model.theta_[:, 0], model.theta_[:, 1], marker="x", c="black", s=130
            )
            ax.set_xlabel("feature 1")
            ax.set_ylabel("feature 2")

        return [
            Frame(
                draw=draw,
                caption="Crosses mark learned class means; color shows P(class 1).",
                metrics={"train accuracy": float(np.mean(model.predict(X) == y))},
            )
        ]


@register_panel
class CategoricalNBPanel(VizPanel):
    """Show how choosing a category changes the posterior."""

    name = "Categorical Naive Bayes — category evidence"
    family = "probabilistic"
    description = "Change a category and watch the class probabilities update."

    def parameters(self) -> list[Parameter]:
        """Expose the amount of pseudocount smoothing."""
        return [
            Parameter("alpha", "Smoothing", 1.0, "float", min=0.1, max=5.0, step=0.1)
        ]

    def frames(self, alpha: float = 1.0) -> list[Frame]:
        """Return a frame for each observed category code."""
        from .numpy_impl import CategoricalNB

        X = [[0], [0], [0], [1], [1], [1]]
        y = ["A", "A", "B", "A", "B", "B"]
        model = CategoricalNB(alpha=alpha).fit(X, y)
        frames = []
        for category in (0, 1):
            probabilities = model.predict_proba([[category]])[0]

            def draw(ax, probabilities=probabilities):
                ax.bar(model.classes_, probabilities, color=["#4477AA", "#EE6677"])
                ax.set_ylim(0, 1)
                ax.set_ylabel("posterior probability")

            frames.append(
                Frame(
                    draw=draw,
                    caption=f"Observed category: {category}",
                    metrics={"P(A)": float(probabilities[0])},
                )
            )
        return frames


@register_panel
class MultinomialNBPanel(VizPanel):
    """Show how adding word counts changes the posterior."""

    name = "Multinomial Naive Bayes — word counts"
    family = "probabilistic"
    description = "Add occurrences of a word and watch the posterior change."

    def parameters(self) -> list[Parameter]:
        """Expose the amount of pseudocount smoothing."""
        return [
            Parameter("alpha", "Smoothing", 1.0, "float", min=0.1, max=5.0, step=0.1)
        ]

    def frames(self, alpha: float = 1.0) -> list[Frame]:
        """Return one frame for each number of occurrences of 'offer'."""
        from .numpy_impl import MultinomialNB

        X = [[3, 0], [2, 0], [0, 3], [0, 2]]
        y = ["spam", "spam", "work", "work"]
        model = MultinomialNB(alpha=alpha).fit(X, y)
        frames = []
        for count in range(5):
            probabilities = model.predict_proba([[count, 1]])[0]

            def draw(ax, probabilities=probabilities):
                ax.bar(model.classes_, probabilities, color=["#4477AA", "#EE6677"])
                ax.set_ylim(0, 1)
                ax.set_ylabel("posterior probability")

            frames.append(
                Frame(
                    draw=draw,
                    caption=f"'offer': {count}, 'meeting': 1",
                    metrics={"P(spam)": float(probabilities[0])},
                )
            )
        return frames
