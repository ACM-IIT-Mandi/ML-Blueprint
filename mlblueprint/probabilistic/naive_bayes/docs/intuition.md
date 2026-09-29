# Naive Bayes — Intuition

## One idea, three kinds of data

Suppose we have labeled examples and want to classify a new one. Naive Bayes asks two questions for each possible class: **how common was this class?** And **how well do this sample's features fit the class?** It combines the answers and chooses the class with the highest score.

For example, a flower with short petals may fit one species better than another. An email containing the word "offer" twice may fit spam better than work mail. We learn those patterns from labeled training examples.

The *naive* assumption is that features provide separate clues once we know the class. Petal length and width may actually be related, so this assumption can make probabilities overconfident. It often remains a useful, fast baseline.

## Which version fits which input?

| Version | Feature value | What it learns for each class |
|---|---|---|
| Gaussian | A measurement such as 4.2 cm | The average and spread of each feature |
| Categorical | A code such as color `0=red`, `1=blue` | How often each category occurs for each feature |
| Multinomial | A count such as `offer=2` | How much of each feature occurs in that class |

**Gaussian:** A measurement close to the class average fits better than one far away. A bell curve models that fit. Each class and feature has its own average and spread.

**Categorical:** A feature is one of a fixed set of categories. For example, color and shape are separate features; color code `0` and shape code `0` mean different things. Each feature has its own category frequencies. Use consistent nonnegative integer codes, starting at zero. Codes larger than the largest training code for that feature need retraining or a defined "other" category.

**Multinomial:** Features are nonnegative counts, often word frequencies in a document. The model learns how much of each word appeared in documents of each class. It then uses the new document's counts to score each class. A feature with count zero contributes nothing to that document's score.

## Why smoothing and logarithms?

A category or word may never appear in the training examples of a class. Without smoothing, its estimated frequency would be zero, making the entire class score zero. The discrete models add `alpha` to counts before turning them into frequencies.

Gaussian features can instead have zero *variance* within a class, which makes the bell-curve calculation divide by zero. `var_smoothing` adds a small amount based on the variation in the whole training set. If that amount is also zero, the model rejects the data.

Scoring multiplies many small numbers. A computer may round their product to zero. Taking logs turns multiplication into addition without changing which class wins. The [derivation](derivation.md) shows the exact steps.
