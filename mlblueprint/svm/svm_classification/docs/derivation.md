# SVM Classification — Derivation (Primal Form)

## 1. The Model

For a dataset with $n$ features and binary targets $y \in \{-1, +1\}$, the linear SVM classifier predicts the decision boundary as:

$$f(x) = w^T x + b$$

The decision rule for predicting the class label $\hat{y}$ is:

$$\hat{y} = \text{sign}(w^T x + b)$$

Where:
- **$X$** is the feature matrix of shape $(N, n)$
- **$w$** is the weight vector (normal vector to the hyperplane) of shape $(n, 1)$
- **$b$** is the bias term (offset)
- **$y^{(i)} \in \{-1, +1\}$** is the true target label

---

## 2. Margin & Objective (Primal Form)

### 2.1 Geometric Margin & Hard Margin

The geometric distance from any point $x^{(i)}$ to the hyperplane $w^T x + b = 0$ is:

$$\gamma^{(i)} = \frac{y^{(i)}(w^T x^{(i)} + b)}{\|w\|}$$

To maximize the margin $M = \frac{2}{\|w\|}$ subject to enforcing correct classification $y^{(i)}(w^T x^{(i)} + b) \ge 1$, we formulate the equivalent minimization problem:

$$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{subject to} \quad y^{(i)}(w^T x^{(i)} + b) \ge 1, \quad \forall i$$

### 2.2 Soft Margin & Hinge Loss

To handle non-separable data, we introduce slack variables $\xi_i \ge 0$. Re-framing this as an unconstrained optimization problem leads to the **L2-Regularized Hinge Loss**:

$$L(w, b) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{N} \max\left(0, 1 - y^{(i)}(w^T x^{(i)} + b)\right)$$

Where:
- **$\frac{1}{2} \|w\|^2$** is the L2 regularization term enforcing a large margin.
- **$\max(0, 1 - y^{(i)}(w^T x^{(i)} + b))$** is the **Hinge Loss**.
- **$C$** is the regularization hyperparameter controlling the trade-off between maximizing margin and minimizing margin violations.

---

## 3. Gradient Calculation (Sub-gradient Descent)

Because the hinge loss has a kink at $1 - y^{(i)}(w^T x^{(i)} + b) = 0$, we calculate its **sub-gradients**:

For a single example $i$, let $t_i = y^{(i)}(w^T x^{(i)} + b)$:

### 3.1 Sub-gradient w.r.t. Weights ($w$)

$$\frac{\partial L_i}{\partial w} = \begin{cases} 
w & \text{if } y^{(i)}(w^T x^{(i)} + b) \ge 1 \quad \text{(Correct & outside margin)} \\ 
w - C \cdot y^{(i)} x^{(i)} & \text{if } y^{(i)}(w^T x^{(i)} + b) < 1 \quad \text{(Margin violation / Wrong class)} 
\end{cases}$$

### 3.2 Sub-gradient w.r.t. Bias ($b$)

$$\frac{\partial L_i}{\partial b} = \begin{cases} 
0 & \text{if } y^{(i)}(w^T x^{(i)} + b) \ge 1 \\ 
- C \cdot y^{(i)} & \text{if } y^{(i)}(w^T x^{(i)} + b) < 1 
\end{cases}$$

Summing/averaging across all $N$ training examples:

$$\frac{\partial L}{\partial w} = w + \frac{C}{N} \sum_{i=1}^{N} \begin{cases} 0 & \text{if } y^{(i)}(w^T x^{(i)} + b) \ge 1 \\ -y^{(i)} x^{(i)} & \text{if } y^{(i)}(w^T x^{(i)} + b) < 1 \end{cases}$$

$$\frac{\partial L}{\partial b} = \frac{C}{N} \sum_{i=1}^{N} \begin{cases} 0 & \text{if } y^{(i)}(w^T x^{(i)} + b) \ge 1 \\ -y^{(i)} & \text{if } y^{(i)}(w^T x^{(i)} + b) < 1 \end{cases}$$

---

## 4. Parameter Update Rule

Using standard Sub-gradient Descent with learning rate $\alpha$:

$$w := w - \alpha \frac{\partial L}{\partial w}$$

$$b := b - \alpha \frac{\partial L}{\partial b}$$

---

## 5. Roadmap: Dual Formulation & Kernels (To Be Added)