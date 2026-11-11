# SVM Classification — Complexity Analysis (Primal Form)

## Overview

This document analyzes the time and space complexity of Linear SVM classification trained via Sub-Gradient Descent in the **Primal Form**.

## Notation

- **N** = number of training samples
- **n** = number of features
- **T** = number of training iterations (epochs)
- **$N_{sv}$** = number of support vectors (relevant for Dual Form / prediction)

---

## Time Complexity

### Training (fit via Primal Sub-gradient Descent)

| Operation | Complexity | Explanation |
|-----------|-----------|-------------|
| Functional margin check | O(N × n) | Compute $y^{(i)}(X w + b)$ for all $N$ samples |
| Gradient computation | O(N × n) | Vectorized check of margin condition and accumulation |
| Update weights & bias | O(n) | Vectorized parameter step |
| **Per iteration** | **O(N × n)** | Dominated by dot product matrix operations |
| **Total training** | **O(T × N × n)** | T iterations of sub-gradient descent |

### Prediction (predict)

| Operation | Complexity | Explanation |
|-----------|-----------|-------------|
| Single prediction (Primal) | O(n) | Compute $\text{sign}(w^T x + b)$ |
| Batch prediction (Primal) | O(M × n) | M new samples, each requires O(n) dot product |

---

## Space Complexity

| Component | Space | Explanation |
|-----------|-------|-------------|
| Weights ($w$) | O(n) | One weight parameter per feature |
| Bias ($b$) | O(1) | Single scalar parameter |
| Input data ($X$) | O(N × n) | Training dataset matrix |
| Margins / Gradient buffers | O(N) | Working arrays for active violations |
| **Total Space** | **O(N × n)** | Dominated by input data storage |

---

## Comparative Summary & Expansion Blueprint

| Approach | Training Time | Prediction Time | Memory | Notes |
|----------|--------------|─────────────────|--------|-------|
| **Primal (GD)** *(Current)* | **O(T × N × n)** | **O(n)** | **O(n)** | Fast & scalable for large $N$ when $n$ is modest. Linear kernel only. |
| **Dual (QP Solver)** *(Future)* | $O(N^2 \sim N^3)$ | $O(N_{sv} \times n)$ | $O(N^2)$ | Required for non-linear kernels; expensive for large $N$. |
| **Kernel Dual** *(Future)* | $O(N^2 \sim N^3)$ | $O(N_{sv} \times n_{\text{kernel}})$ | $O(N^2)$ | Computes kernel matrix $K(x_i, x_j)$. |