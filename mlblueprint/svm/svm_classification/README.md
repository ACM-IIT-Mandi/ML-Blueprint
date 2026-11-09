# Support Vector Machine (SVM) Classifier

Family: svm  
Completion & Scope: `draft`  
Maintainers: @VedanshTyagi

Support Vector Machine (SVM) is a supervised machine learning algorithm used for binary classification. It finds an optimal hyperplane that separates two classes while maximizing the margin between them.

This implementation uses both **primal form of SVM** with hinge loss and regularization and **dual form of SVM** with SMO. Currently, it supports primal form only.

## Files

| Version | File | Done? |
|---------|------|-------|
| Plain Python | `scratch.py` | ✅ yes |
| Fast version | `numpy_impl.py` | ✅ yes |
| PyTorch | `torch_impl.py` | ❌ no |

## Documentation

- [Intuition](docs/intuition.md) — The idea in plain words
- [Derivation](docs/derivation.md) — The maths, step by step
- [Complexity](docs/complexity.md) — Time and space analysis
- [References](docs/references.md) — Papers, books, links

## Usage

```python
from mlblueprint.svm.svm_classifier import SvmClassifier
import numpy as np

# Example data
X = np.array([[1, 2], [2, 3], [3, 3], [6, 7], [7, 8], [8, 8]])
y = np.array([-1, -1, -1, 1, 1, 1])

# Create and train the model
model = SvmClassifier(lr=0.001, n_iters=1000, lambda_param=0.01)
model.fit(X, y)

# Make predictions
predictions = model.predict(np.array([[4, 5], [7, 7]]))
print(predictions)
```