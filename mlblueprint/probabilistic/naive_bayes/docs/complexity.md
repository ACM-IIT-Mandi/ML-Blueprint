# Naive Bayes — Complexity

## Notation

- $n$: training samples
- $d$: features
- $k$: classes
- $q$: prediction samples
- $K_j$: category slots for feature $j$
- $S=\sum_{j=1}^{d}K_j$: total category slots

## Time Complexity

### Gaussian Naive Bayes

Training reads every feature value to calculate class priors, means, and variances.

- Training: $O(nd)$
- Prediction: $O(qkd)$, because every query is evaluated using $d$ features for each of the $k$ classes.

### Categorical Naive Bayes

Training counts each category occurrence and creates probability tables containing $kS$ entries.

- Training: $O(nd+kS)$
- Prediction: $O(qkd)$, because one category probability is read for every feature and class.

### Multinomial Naive Bayes

Training reads and sums every feature count, then smooths the $kd$ class-feature values.

- Training: $O(nd)$
- Prediction: $O(qkd)$, because every query feature contributes to the score of every class.

## Space Complexity

| Model       | Model space | Stored information                                       |
| ----------- | ----------: | -------------------------------------------------------- |
| Gaussian    |     $O(kd)$ | Mean and variance for every class-feature pair           |
| Categorical |     $O(kS)$ | Probability for every class-feature-category combination |
| Multinomial |     $O(kd)$ | Count and probability for every class-feature pair       |

Returning probabilities for $q$ samples requires an additional $O(qk)$ output space.

## Implementation Note

The core complexities above exclude class ordering. The current implementation keeps `classes_` deterministic: `sorted(set(y))` costs expected $O(n+k\log k)$ in `scratch.py`, while NumPy's `np.unique` adds approximately $O(n\log n)$ in `numpy_impl.py`.

## Summary

| Model       |   Training | Prediction | Model space |
| ----------- | ---------: | ---------: | ----------: |
| Gaussian    |    $O(nd)$ |   $O(qkd)$ |     $O(kd)$ |
| Categorical | $O(nd+kS)$ |   $O(qkd)$ |     $O(kS)$ |
| Multinomial |    $O(nd)$ |   $O(qkd)$ |     $O(kd)$ |
