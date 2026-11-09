"""Pure Python implementation of Linear SVM without external libraries."""


class SvmClassification:
    """Linear SVM using pure Python and Gradient Descent (Primal Form)."""

    def __init__(self):
        """Initialize model parameters."""
        self.weights = None
        self.bias = None

    def decision_function(self, X):
        """
        Calculate decision scores.

        Parameters
        ----------
        X : list of list of float
            Input data.

        Returns
        -------
        list of float
            Decision scores.
        """
        scores = []

        for i in range(len(X)):
            out = self.bias

            for j in range(len(self.weights)):
                out += X[i][j] * self.weights[j]

            scores.append(out)

        return scores

    def hinge_loss(self, y, scores):
        """Calculate hinge loss for each sample."""
        losses = []

        for i in range(len(y)):
            loss = max(0, 1 - y[i] * scores[i])
            losses.append(loss)

        return losses

    def compute_loss(self, X, y, C):
        """Calculate the primal objective function."""
        scores = self.decision_function(X)
        losses = self.hinge_loss(y, scores)

        regularization = 0.5 * sum(w**2 for w in self.weights)
        total_loss = regularization + C * sum(losses)

        return total_loss

    def compute_gradients(self, X, y, scores, C):
        """Calculate gradients of the primal objective."""
        n_samples = len(X)
        n_features = len(self.weights)

        dw = self.weights.copy()
        db = 0

        for i in range(n_samples):
            margin = y[i] * scores[i]

            if margin < 1:
                for j in range(n_features):
                    dw[j] -= C * y[i] * X[i][j]

                db -= C * y[i]

        return dw, db

    def fit(self, X, y, lr=0.01, n_iters=1000, C=1.0):
        """
        Train using primal Gradient Descent.

        Loss function:
            1/2 * ||w||^2 + C * sum(max(0, 1 - y * (w^T x + b)))

        Hinge loss is zero at and beyond the margin boundary.
        At the boundary, a zero hinge-loss subgradient is used.

        Parameters
        ----------
        X : list of list of float
            Training input data.

        y : list of float
            Training labels. Must be -1 or +1.

        lr : float
            Learning rate.

        n_iters : int
            Number of iterations.

        C : float
            Regularization parameter.

        Returns
        -------
        self : SVM
            The trained model.
        """
        n_samples = len(X)
        n_features = len(X[0])

        if len(y) != n_samples:
            raise ValueError("X and y must contain the same number of samples")

        if any(label not in [-1, 1] for label in y):
            raise ValueError("Labels must be -1 or +1")

        if lr <= 0 or n_iters <= 0 or C <= 0:
            raise ValueError("lr, n_iters and C must be positive")

        if any(len(row) != n_features for row in X):
            raise ValueError("All samples must have the same number of features")

        self.weights = [0.0] * n_features
        self.bias = 0.0

        for epoch in range(n_iters):
            scores = self.decision_function(X)

            dw, db = self.compute_gradients(X, y, scores, C)

            for j in range(n_features):
                self.weights[j] -= lr * dw[j]

            self.bias -= lr * db

            if epoch % 100 == 0:
                loss = self.compute_loss(X, y, C)
                print(f"epoch {epoch} loss: {loss}")

        return self

    def predict(self, X):
        """
        Predict binary class labels.

        Parameters
        ----------
        X : list of list of float
            Input data.

        Returns
        -------
        list of int
            Predicted labels (-1 or +1).
        """
        scores = self.decision_function(X)

        predictions = []

        for score in scores:
            predictions.append(1 if score >= 0 else -1)

        return predictions
