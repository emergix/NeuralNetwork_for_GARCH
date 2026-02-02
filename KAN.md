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


> [!IMPORTANT]
> ### 🔎 KAN as Learned Feature Engineering
> A **Kolmogorov–Arnold Network (KAN)** can be interpreted as a neural architecture where  
> **optimal feature engineering is learned jointly with the model itself**.
>
> In a classical pipeline:
>
> **Raw data → Feature engineering → Model**
>
> feature transformations are manually designed (logs, ratios, nonlinearities, domain heuristics).
>
> In a KAN:
>
> $$
> y_i = \sum_j \Phi_{ij}(x_j)
> $$
>
> each $\Phi_{ij}$ is a learnable univariate function acting as an **adaptive feature transform**.
>
> This means the network:
>
> - does not just learn weights  
> - learns **how each variable should be transformed**  
> - makes the feature space itself trainable  
>
> **Conceptually:**  
> **Raw data → Learned nonlinear coordinate system → Linear combination**
>
> 👉 **KAN = Neural network + automatic discovery of the right feature representation**


# Kolmogorov–Arnold Networks (KAN) — Mathematical Formulation

KANs are neural networks where scalar weights are replaced by learnable functions.

---

## 1. Kolmogorov–Arnold Theorem

Let  

$$
f : [0,1]^n \rightarrow \mathbb{R}
$$

be continuous.

There exist univariate functions  

$$
\phi_q : \mathbb{R} \rightarrow \mathbb{R}, \quad q = 1,\dots,2n+1
$$

and  

$$
\psi_{q,p} : [0,1] \rightarrow \mathbb{R}, \quad p = 1,\dots,n
$$

such that:

$$
f(x_1,\dots,x_n) = \sum_{q=1}^{2n+1} \phi_q \left( \sum_{p=1}^{n} \psi_{q,p}(x_p) \right)
$$

---

## 2. Classical MLP Approximation

$$
f(x) \approx \sum_{i=1}^{m} a_i \, \sigma(w_i^\top x + b_i)
$$

---

## 3. KAN Layer Formulation

Let

$$
x \in \mathbb{R}^{n_{\text{in}}}, \quad y \in \mathbb{R}^{n_{\text{out}}}
$$

A neuron:

$$
y_i = \sum_{j=1}^{n_{\text{in}}} \Phi_{ij}(x_j)
$$

A layer:

$$
\mathcal{K}(x)_i = \sum_j \Phi_{ij}(x_j)
$$

---

## 4. Parameterization of Functions

### B-splines
$$
\Phi_{ij}(x) = \sum_{k=1}^{K} c_{ijk} B_k(x)
$$

### Polynomials
$$
\Phi_{ij}(x) = \sum_{k=0}^{d} c_{ijk} x^k
$$

### Radial bases
$$
\Phi_{ij}(x) = \sum_k c_{ijk} \exp(-\gamma_k (x-\mu_k)^2)
$$

---

## 5. Deep KAN Network

$$
f(x) = \mathcal{K}^{(L)} \circ \dots \circ \mathcal{K}^{(1)}(x)
$$

Each layer:

$$
x^{(\ell+1)}_i = \sum_j \Phi^{(\ell)}_{ij}(x^{(\ell)}_j)
$$

---

## 6. Functional Regularization

Curvature penalty:

$$
\mathcal{L}_{\text{smooth}} = \lambda \sum_{i,j} \int \left( \frac{d^2}{dx^2} \Phi_{ij}(x) \right)^2 dx
$$

---

## 7. Formal Summary

$$
x^{(\ell+1)}_i = \sum_j \Phi^{(\ell)}_{ij}(x^{(\ell)}_j)
$$

A KAN operates in **function space rather than weight space**.



## 8) Module Fourni

 → notebook python : [KAN_learning.ipynb](./notebooks/KAN_learning.ipynb)

