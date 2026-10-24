"""NumPy implementation of Linear Regression with L2 regularization."""

import numpy as np


class LinearRegression:
    """
    Ridge Regression (L2 Regularized Linear Regression) using Gradient Descent.

    Loss = (1/2N) * sum((y_hat - y)^2) + lam * sum(w^2)

    Parameters
    ----------
    lr : float
        Learning rate. Default 0.01.
    n_iters : int
        Number of gradient descent iterations. Default 1000.
    lam : float
        L2 regularization strength. Set to 0 for plain Linear Regression.
    """

    def __init__(self, lr=0.01, n_iters=1000, lam=0.1):
        """Initialize hyperparameters."""
        self.lr = lr
        self.n_iters = n_iters
        self.lam = lam
        self.weights = None
        self.bias = None
        self.theta = None

    def fit(self, X, y):
        """
        Train the model using Gradient Descent.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training input data.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : LinearRegression
            The trained model.
        """
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # training loop
        for epoch in range(self.n_iters):
            # forward pass
            y_hat = X @ self.weights + self.bias

            # loss
            mse = np.mean((y_hat - y) ** 2) / 2 + (self.lam / (2 * n_samples)) * np.sum(
                self.weights**2
            )

            # print loss
            if epoch % 100 == 0:
                print(f"epoch {epoch} loss: {mse}")

            # backpropagation
            dw = (1 / n_samples) * (np.dot(X.T, y_hat - y)) + (
                self.lam / n_samples
            ) * self.weights
            db = (1 / n_samples) * (np.sum(y_hat - y))

            # update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def equation_fit(self, X, y):
        """
        Fit the model using the closed-form solution (Normal Equation).

        Theta = (X^T * X)^-1 * X^T * y

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training input data.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : Theta coffecient
            Coefficient of the trained model.
        """
        X = np.array(X)
        y = np.array(y)

        # Add a column of ones to X matrix for the intercept (bias term)
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        I = np.eye(X.shape[1])
        I[0, 0] = 0

        A = X.T @ X + self.lam * I
        b = X.T @ y

        # Solve for theta directly without inversion ,inversion - more error prone .
        # linalg.solve used LU decomposition to solve equation making it much faster.
        self.theta = np.linalg.solve(A, b)
        self.bias = self.theta[0]
        self.weights = self.theta[1:]

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to predict on.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted values.
        """
        X = np.array(X)
        y_hat = X @ self.weights + self.bias
        return y_hat

    def equation_predict(self, X):
        """
        Make predictions using the closed-form solution.

        Y = X * Theta

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to predict on.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted values.
        """
        # Add a column of ones to X matrix for the intercept (bias term)
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        X = np.array(X)
        y_hat = X @ self.theta
        return y_hat
