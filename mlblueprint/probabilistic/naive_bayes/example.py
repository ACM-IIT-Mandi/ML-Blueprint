"""Run Gaussian, Categorical, and Multinomial Naive Bayes examples."""

from mlblueprint.core.datasets import make_blobs
from mlblueprint.core.metrics import accuracy_score
from mlblueprint.probabilistic import CategoricalNB, GaussianNB, MultinomialNB


def main():
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.5, random_state=42)
    gaussian = GaussianNB().fit(X[:150], y[:150])
    print("Gaussian (continuous measurements)")
    print("  test accuracy:", accuracy_score(y[150:], gaussian.predict(X[150:])))
    print("  first test probabilities:", gaussian.predict_proba(X[150:151])[0])

    # Column 1 is color (0=red, 1=blue); column 2 is shape (0=round, 1=square).
    colors_shapes = [[0, 0], [0, 1], [0, 0], [1, 1], [1, 0], [1, 1]]
    labels = ["apple", "apple", "apple", "box", "box", "box"]
    categorical = CategoricalNB().fit(colors_shapes, labels)
    print("Categorical (category codes)")
    print("  prediction for red/round:", categorical.predict([[0, 0]])[0])
    print("  class order:", categorical.classes_)

    # Columns count occurrences of the words "offer" and "meeting".
    word_counts = [[2, 0], [3, 0], [1, 0], [0, 2], [0, 3], [0, 1]]
    labels = ["spam", "spam", "spam", "work", "work", "work"]
    multinomial = MultinomialNB().fit(word_counts, labels)
    print("Multinomial (word counts)")
    print("  prediction for two offers:", multinomial.predict([[2, 0]])[0])
    print("  class order:", multinomial.classes_)


if __name__ == "__main__":
    main()
