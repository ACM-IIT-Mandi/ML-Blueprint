# Naive Bayes — Derivation

## 0. Notation

| Symbol       | Meaning                                           | Code name                            |
| ------------ | ------------------------------------------------- | ------------------------------------ |
| $n$          | Number of training samples                        | `len(X)`                             |
| $d$          | Number of features                                | `n_features_in_`                     |
| $\mathcal C$ | Set of possible classes                           | `classes_`                           |
| $n_c$        | Number of training samples belonging to class $c$ | Class count                          |
| $x^{(i)}$    | Feature vector of training sample $i$             | `X[i]`                               |
| $y^{(i)}$    | Class label of training sample $i$                | `y[i]`                               |
| $\pi_c$      | Prior probability of class $c$                    | `class_prior_` or `class_log_prior_` |
| $a_c(x)$     | Unnormalized log score of class $c$ for input $x$ | Internal class score                 |
| $\alpha$     | Laplace-smoothing strength                        | `alpha`                              |
| $\epsilon$   | Added Gaussian variance                           | Computed from `var_smoothing`        |

---

## 1. Bayes' Rule

For an input vector

$$
x=(x_1,x_2,\ldots,x_d),
$$

the probability that it belongs to class $c$ is

$$
P(c\mid x)=\frac{p(x\mid c)P(c)}{p(x)}.
$$

Here:

- $P(c)$ is the class prior.
- $p(x\mid c)$ is the likelihood of observing $x$ in class $c$.
- $p(x)$ is the same for every candidate class.

Therefore, the predicted class is

$$
\hat c=\arg\max_{c\in\mathcal C}P(c)p(x\mid c).
$$

---

## 2. Conditional-Independence Assumption

Naive Bayes assumes that the features are independent once the class is known:

$$
p(x\mid c)=\prod_{j=1}^{d}p(x_j\mid c).
$$

Substituting this into the decision rule gives

$$
\hat c = \arg\max_{c\in\mathcal C} P(c)\prod_{j=1}^{d}p(x_j\mid c).
$$

This assumption is called _naive_ because real features may be related. It makes the model simple because each feature distribution can be learned separately.

---

## 3. Class Prior

The maximum-likelihood estimate of the prior is the observed class frequency:

$$
\pi_c=P(c)=\frac{n_c}{n},
$$

where

$$
n_c=\sum_{i=1}^{n}\mathbf 1\left(y^{(i)}=c\right).
$$

The implementations store either $\pi_c$ or $\log\pi_c$, depending on the model.

---

## 4. Why Use Log Probabilities?

The class score contains a product of probabilities:

$$
P(c)\prod_{j=1}^{d}p(x_j\mid c).
$$

Multiplying many small probabilities can underflow to zero. Since the logarithm is strictly increasing, it does not change which class has the largest score:

$$
a_c(x) = \log\pi_c + \sum_{j=1}^{d}\log p(x_j\mid c).
$$

Therefore,

$$
\hat c=\arg\max_{c\in\mathcal C}a_c(x).
$$

Products have now become sums, making the calculation numerically safer.

---

## 5. Posterior Probability Normalization

The scores $a_c(x)$ are unnormalized log probabilities. To obtain posterior probabilities, compute

$$
P(c\mid x) = \frac{e^{a_c(x)}}{\sum_{r\in\mathcal C}e^{a_r(x)}}.
$$

Direct exponentiation can overflow, so let

$$
m=\max_{r\in\mathcal C}a_r(x).
$$

Subtracting the same value from every score does not change the result:

$$
P(c\mid x) = \frac{e^{a_c(x)-m}} {\sum_{r\in\mathcal C}e^{a_r(x)-m}}.
$$

Therefore, the normalized log probability is

$$
\log P(c\mid x) = a_c(x) - \left[ m+\log\sum_{r\in\mathcal C}e^{a_r(x)-m} \right].
$$

This is the log-sum-exp calculation used by `predict_log_proba`.

---

## 6. Gaussian Naive Bayes

### 6.1 Assumption

Gaussian Naive Bayes is used for continuous features. It assumes that feature $j$ in class $c$ follows a Gaussian distribution:

$$
x_j\mid c\sim\mathcal N(\mu_{cj},v_{cj}).
$$

The corresponding probability density is

$$
p(x_j\mid c) = \frac{1}{\sqrt{2\pi v_{cj}}} \exp\left( -\frac{(x_j-\mu_{cj})^2}{2v_{cj}} \right).
$$

Each class and feature has its own mean and variance.

### 6.2 Mean Estimate

For class $c$, the mean of feature $j$ is

$$
\mu_{cj} = \frac{1}{n_c} \sum_{i:y^{(i)}=c}x_j^{(i)}.
$$

This is stored in `theta_`.

### 6.3 Variance Estimate

The maximum-likelihood population variance is

$$
v_{cj} = \frac{1}{n_c} \sum_{i:y^{(i)}=c} \left(x_j^{(i)}-\mu_{cj}\right)^2.
$$

The denominator is $n_c$, not $n_c-1$, because the model uses the maximum-likelihood estimate.

### 6.4 Variance Smoothing

A zero variance would cause division by zero in the Gaussian density. The implementation calculates

$$
\epsilon = s\max_j\mathrm{Var}(X_{:j}),
$$

where $s$ is `var_smoothing`.

The stored variance is

$$
\widetilde v_{cj}=v_{cj}+\epsilon.
$$

This is stored in `var_`.

If $\widetilde v_{cj}$ remains zero, the input does not contain enough variation to define a Gaussian density, so `fit` raises an error.

### 6.5 Gaussian Log Score

Taking the logarithm of the Gaussian density gives

$$
\log p(x_j\mid c) = -\frac{1}{2} \left[ \log(2\pi\widetilde v_{cj}) + \frac{(x_j-\mu_{cj})^2}{\widetilde v_{cj}} \right].
$$

Therefore, the complete class score is

$$
a_c(x) = \log\pi_c - \frac{1}{2} \sum_{j=1}^{d} \left[ \log(2\pi\widetilde v_{cj}) + \frac{(x_j-\mu_{cj})^2}{\widetilde v_{cj}} \right].
$$

The class with the largest value of $a_c(x)$ is predicted.

---

## 7. Categorical Naive Bayes

### 7.1 Assumption

Categorical Naive Bayes is used when every feature contains one category code.

For feature $j$, valid category codes are

$$
0,1,\ldots,K_j-1,
$$

where $K_j$ is one plus the largest category code observed during training.

Each feature has its own categorical distribution. For example, category `0` in a color feature is unrelated to category `0` in a shape feature.

### 7.2 Category Counts

Let

$$
N_{cjt} = \sum_{i=1}^{n} \mathbf 1 \left( y^{(i)}=c \text{ and } x_j^{(i)}=t \right)
$$

be the number of class-$c$ samples whose feature $j$ has category $t$.

Without smoothing, the maximum-likelihood probability is

$$
P(x_j=t\mid c)=\frac{N_{cjt}}{n_c}.
$$

If $N_{cjt}=0$, this probability becomes zero and makes the entire class score zero.

### 7.3 Laplace Smoothing

The implementation adds the pseudocount $\alpha>0$ to every category:

$$
P(x_j=t\mid c) = \frac{N_{cjt}+\alpha} {n_c+\alpha K_j}.
$$

The denominator contains $\alpha K_j$ because there are $K_j$ possible category slots for feature $j$.

The smoothed probabilities still sum to one:

$$
\sum_{t=0}^{K_j-1}P(x_j=t\mid c)=1.
$$

### 7.4 Categorical Log Score

For an input $x$, the class score is

$$
a_c(x) = \log\pi_c + \sum_{j=1}^{d} \log \left( \frac{N_{cjx_j}+\alpha} {n_c+\alpha K_j} \right).
$$

Only the probability corresponding to the observed category $x_j$ is used for each feature.

A category code outside

$$
0,\ldots,K_j-1
$$

cannot index the learned table, so prediction rejects it.

---

## 8. Multinomial Naive Bayes

### 8.1 Assumption

Multinomial Naive Bayes is used for nonnegative counts, such as word frequencies.

For class $c$, let

$$
N_{cj} = \sum_{i:y^{(i)}=c}x_j^{(i)}
$$

be the total count of feature $j$ across all class-$c$ training samples.

This is a sum of feature counts, not the number of samples containing the feature.

### 8.2 Smoothed Feature Probability

Without smoothing, the estimated feature probability is

$$
\phi_{cj} = \frac{N_{cj}} {\sum_{\ell=1}^{d}N_{c\ell}}.
$$

With Laplace smoothing,

$$
\phi_{cj} = \frac{N_{cj}+\alpha} {\sum_{\ell=1}^{d}N_{c\ell}+\alpha d}.
$$

The denominator contains $\alpha d$ because there are $d$ possible features.

The probabilities for each class sum to one:

$$
\sum_{j=1}^{d}\phi_{cj}=1.
$$

### 8.3 Multinomial Likelihood

For a count vector $x$ with

$$
m=\sum_{j=1}^{d}x_j
$$

total occurrences, the multinomial likelihood is

$$
P(x\mid c) = \frac{m!}{\prod_{j=1}^{d}x_j!} \prod_{j=1}^{d}\phi_{cj}^{x_j}.
$$

The factorial term counts the number of possible arrangements of the observed occurrences.

Taking the logarithm gives

$$
\log P(x\mid c) = \log \left( \frac{m!}{\prod_{j=1}^{d}x_j!} \right) + \sum_{j=1}^{d}x_j\log\phi_{cj}.
$$

The factorial term depends on $x$ but not on the class. It therefore cancels when classes are compared or posterior probabilities are normalized.

### 8.4 Multinomial Log Score

The class-dependent score used by the implementation is

$$
a_c(x) = \log\pi_c + \sum_{j=1}^{d}x_j\log\phi_{cj}.
$$

A feature with count zero contributes nothing because

$$
0\log\phi_{cj}=0.
$$

A feature occurring multiple times contributes its log probability multiple times.

The multinomial probability formula assumes integer counts. The scoring rule can also accept nonnegative weighted counts, although they no longer have the literal interpretation of event frequencies.

---

## 9. Categorical vs Multinomial

The models are related but represent different inputs.

| Property               | Categorical                     | Multinomial                                  |
| ---------------------- | ------------------------------- | -------------------------------------------- |
| Meaning of a feature   | One category choice             | A nonnegative count                          |
| Example                | `color = red`                   | `offer = 3`                                  |
| Feature value          | One integer code                | Count or weight                              |
| Probability tables     | Separate table for each feature | One feature-frequency distribution per class |
| Smoothing denominator  | $n_c+\alpha K_j$                | $\sum_jN_{cj}+\alpha d$                      |
| Log-score contribution | $\log P(x_j\mid c)$             | $x_j\log\phi_{cj}$                           |

Categorical Naive Bayes asks which category was selected for each feature. Multinomial Naive Bayes asks how many times each feature occurred.

---

## 10. Formula-to-Code Mapping

| Mathematical value                             | Stored attribute     |
| ---------------------------------------------- | -------------------- |
| Class labels $\mathcal C$                      | `classes_`           |
| Gaussian prior $\pi_c$                         | `class_prior_`       |
| Gaussian mean $\mu_{cj}$                       | `theta_`             |
| Smoothed Gaussian variance $\widetilde v_{cj}$ | `var_`               |
| Discrete log prior $\log\pi_c$                 | `class_log_prior_`   |
| Categorical counts $N_{cjt}$                   | `category_count_`    |
| Categorical log probabilities                  | `category_log_prob_` |
| Multinomial counts $N_{cj}$                    | `feature_count_`     |
| Multinomial log probabilities $\log\phi_{cj}$  | `feature_log_prob_`  |

All three variants eventually produce one log score per class, normalize those scores using log-sum-exp, and select the class with the largest posterior probability.
