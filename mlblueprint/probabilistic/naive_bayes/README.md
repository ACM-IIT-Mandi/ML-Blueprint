# Naive Bayes

Family: probabilistic  
Completion & Scope: `complete` for Gaussian, Categorical, and Multinomial variants  
Maintainers: @dakshrathi-india

Naive Bayes predicts a class by combining how common that class is with how well each feature fits it. It assumes the features are independent once the class is known. Pick the likelihood that matches your data:

| Model | Input | Typical use |
|---|---|---|
| `GaussianNB` | Continuous numbers | Measurements such as lengths |
| `CategoricalNB` | Nonnegative integer category codes, starting at 0 per feature | Color or shape codes |
| `MultinomialNB` | Nonnegative counts | Word frequencies |

## Files

| Version | File | Done? |
|---|---|---|
| Plain Python (`*Scratch`) | `scratch.py` | yes |
| Fast NumPy version | `numpy_impl.py` | yes |
| Visualizer panels | `viz.py` | yes |

## Docs

[Intuition](docs/intuition.md), [Derivation](docs/derivation.md), [Complexity](docs/complexity.md), [References](docs/references.md)

## Usage

```python
from mlblueprint.probabilistic import GaussianNB, CategoricalNB, MultinomialNB

GaussianNB().fit([[0.0], [0.2], [4.0], [4.2]], [0, 0, 1, 1]).predict([[0.1]])
CategoricalNB(alpha=1.0).fit([[0], [0], [1], [1]], [0, 0, 1, 1]).predict([[1]])
MultinomialNB(alpha=1.0).fit([[2, 0], [1, 0], [0, 2], [0, 1]], [0, 0, 1, 1]).predict(
    [[2, 0]]
)
```

Import `GaussianNBScratch`, `CategoricalNBScratch`, or `MultinomialNBScratch` for the list-based plain Python versions. Each model exposes `fit`, `predict`, `predict_proba`, and `predict_log_proba`; probability columns follow `classes_` order. Discrete models use positive `alpha` for Laplace smoothing. `CategoricalNB` rejects categories outside the range observed for that feature during fit; encode categories consistently before training and prediction. Gaussian `var_smoothing` adds a fraction of the largest overall feature variance; `fit` raises an error if any class-feature variance remains zero after smoothing.

## Run and verify

Use Python 3.10 or newer. Open a terminal in the repository root: running `ls` there must show `pyproject.toml` and the `mlblueprint` folder.

On Windows **Git Bash**:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then install the project and run the example and checks:

```bash
python -m pip install -e ".[dev]"
python -m mlblueprint.probabilistic.naive_bayes.example
python -m pytest mlblueprint/probabilistic/naive_bayes/tests
python -m ruff format --check .
python -m ruff check .
PYTHONUTF8=1 python -m pytest
```

The example runs Gaussian, Categorical, then Multinomial. The focused tests compare predictions and probabilities with scikit-learn; the final test command checks the whole repository. The setup and pull request workflow are also in the repository's `CONTRIBUTING.md`.

## Launch the visualizer

With the virtual environment active, run this command from the repository root:

```bash
python -m streamlit run apps/visualiser/app.py
```

Streamlit opens the visualizer in your browser. In the **Algorithm** menu, select one of these panels:

- `probabilistic · Gaussian Naive Bayes — probability landscape`
- `probabilistic · Categorical Naive Bayes — category evidence`
- `probabilistic · Multinomial Naive Bayes — word counts`

Use the sidebar controls to change smoothing or class overlap. For Categorical and Multinomial, move the **Step** slider to see how the evidence changes the predicted probabilities. Stop the server with `Ctrl+C` in Git Bash.
