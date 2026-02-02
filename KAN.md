# Introduction to Kolmogorov–Arnold Networks (KAN)

**Kolmogorov–Arnold Networks (KANs)** are a neural network architecture inspired by the Kolmogorov–Arnold representation theorem.

Unlike classical neural networks (MLPs) that stack linear layers with fixed nonlinearities, KANs replace **scalar weights** with **learnable functions**, fundamentally changing how relationships between variables are represented.

---

## 1. Mathematical Origin

Any continuous multivariate function  

$f(x_1, x_2, ..., x_n)$  

can be represented as a sum of compositions of one-dimensional functions.

Formally, there exist univariate functions $\phi_q$ and $\psi_{q,p}$ such that:

$$
f(x_1,\dots,x_n) = \sum_{q=1}^{2n+1} \phi_q \left( \sum_{p=1}^{n} \psi_{q,p}(x_p) \right)
$$

👉 **Major consequence:**  
The complexity of a multivariate function can be decomposed into 1D functions.

KANs are a modern attempt to embody this theorem in a neural architecture.

---

## 2. Key Difference from an MLP

| Classical MLP | KAN |
|--------------|-----|
| Weights = real numbers | Weights = 1D functions |
| Fixed nonlinearity (ReLU, GELU…) | Learnable nonlinearity |
| Distributed, opaque learning | More interpretable structure |
| Implicit feature engineering | Emergent and visible feature engineering |

In an MLP:

$$
y = \sigma(Wx + b)
$$

In a KAN:

$$
y_i = \sum_j \Phi_{ij}(x_j)
$$

where each $\Phi_{ij}$ is a learned function.

---

## 3. Geometric Intuition

A KAN learns **how each variable should be individually transformed** before combination.

It can be seen as:

> A network that learns the **right coordinate system** of the problem.

---

## 4. Typical KAN Structure

1. Univariate transformation per edge  
2. Summation (no classical matrix product)  
3. Curvature regularization  

---

## 5. Why It’s Powerful

- Parameter efficiency  
- Interpretability  
- Automatic feature engineering  

---

## Key Sentence to Remember

**An MLP learns weights. A KAN learns laws.**
