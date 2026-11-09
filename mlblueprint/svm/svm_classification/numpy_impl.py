"""NumPy implementation of Linear Support Vector Machine."""

import numpy as np


class SvmClassification:
    """
    Linear Support Vector Machine.

    Supports primal Gradient Descent optimization for training.
    """

    def __init__(self):
        """Initialize model parameters."""
        self.weights = None
        self.bias = None

    def _validate_data(self, X, y):
        """Validate input data and labels."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y must be a 1D array with one label per sample")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain finite values")

        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("Labels must be -1 or +1")

        return X, y

    def _compute_loss(self, X, y, C):
        """
        Calculate the primal objective function.

        Loss = 1/2 * ||w||^2 + C * sum(max(0, 1 - y * (w @ x + b)))
        """
        scores = self.decision_function(X)
        hinge_loss = np.maximum(0, 1 - y * scores)

        return 0.5 * np.sum(self.weights**2) + C * np.sum(hinge_loss)

    def fit(self, X, y, lr=0.01, n_iters=1000, C=1.0):
        """
        Train using primal Gradient Descent.

        Loss function:
            1/2 * ||w||^2 + C * sum(max(0, 1 - y * (w^T x + b)))

        Hinge loss is zero at and beyond the margin boundary.
        At the boundary, a zero subgradient is used.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training input data.

        y : array-like, shape (n_samples,)
            Training labels. Must be -1 or +1.

        lr : float
            Learning rate.

        n_iters : int
            Number of gradient descent iterations.

        C : float
            Regularization parameter.

        Returns
        -------
        self : SVM
            The trained model.
        """
        X, y = self._validate_data(X, y)

        if lr <= 0:
            raise ValueError("Learning rate must be positive")

        if n_iters <= 0:
            raise ValueError("Number of iterations must be positive")

        if C <= 0:
            raise ValueError("C must be positive")

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(n_iters):
            scores = self.decision_function(X)
            margins = y * scores

            active = margins < 1

            dw = self.weights - C * (X[active].T @ y[active])
            db = -C * np.sum(y[active])

            self.weights -= lr * dw
            self.bias -= lr * db

            if epoch % 100 == 0:
                loss = self._compute_loss(X, y, C)
                print(f"epoch {epoch} loss: {loss}")

        return self

    def decision_function(self, X):
        """
        Calculate decision scores.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Signed decision scores.
        """
        X = np.asarray(X, dtype=float)

        if self.weights is None:
            raise ValueError("Model has not been trained yet")

        return X @ self.weights + self.bias

    def predict(self, X):
        """
        Predict binary class labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted labels (-1 or +1).
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)
