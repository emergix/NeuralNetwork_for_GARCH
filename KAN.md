# Introduction to Kolmogorov–Arnold Networks (KAN)

**Kolmogorov–Arnold Networks (KANs)** are a neural network architecture directly inspired by a fundamental 20th-century mathematical result:  
the **Kolmogorov–Arnold representation theorem**.

Unlike classical neural networks (MLPs) that stack linear layers with fixed nonlinearities, KANs replace **scalar weights** with **learnable functions**.  
This fundamentally changes how relationships between variables are represented.

---

## 1. Mathematical Origin

In 1957,  
- **Andrey Kolmogorov**  
- and later **Vladimir Arnold**

proved a surprising result:

> Any continuous multivariate function  
> \( f(x_1, x_2, ..., x_n) \)  
> can be represented as a sum of compositions of one-dimensional functions.

Formally, there exist univariate functions \( \phi_q \) and \( \psi_{q,p} \) such that:

\[
f(x_1,\dots,x_n) = \sum_{q=1}^{2n+1} \phi_q \left( \sum_{p=1}^{n} \psi_{q,p}(x_p) \right)
\]

👉 **Major consequence:**  
The complexity of a multivariate function can be decomposed into 1D functions.

KANs are a modern attempt to **embody this theorem in a neural architecture**.

---

## 2. Key Difference from an MLP

| Classical MLP | KAN |
|--------------|-----|
| Weights = real numbers | Weights = 1D functions |
| Fixed nonlinearity (ReLU, GELU…) | Learnable nonlinearity |
| Distributed, opaque learning | More interpretable structure |
| Implicit feature engineering | Emergent and visible feature engineering |

In an MLP:

\[
y = \sigma(Wx + b)
\]

In a KAN:

\[
y_i = \sum_j \Phi_{ij}(x_j)
\]

where each \( \Phi_{ij} \) is a **learned function**, often parameterized using splines.

---

## 3. Geometric Intuition

An MLP learns a global linear combination and then applies a nonlinear distortion.

A KAN instead learns **how each variable should be individually transformed** before combination.

It can be seen as:

> 🔍 A network that learns the **right coordinate system** of the problem.

KANs tend to excel when:
- functional structure matters  
- underlying physical laws exist  
- feature engineering is crucial  

---

## 4. Typical KAN Structure

A KAN block includes:

1. **Univariate transformation per edge**
   - Each connection is a function \( f_{ij}(x) \)
   - Implemented via splines (B-splines, polynomials, etc.)

2. **Summation**
   - Outputs are added (no classical matrix product)

3. **Curvature regularization**
   - Prevents overly oscillatory functions

---

## 5. Why It’s Powerful

### ✅ Parameter efficiency  
A well-chosen function can replace many linear layers.

### ✅ Interpretability  
We can plot learned functions:
- identify important variables  
- detect physical nonlinearities  
- find thresholds, regimes, saturation

### ✅ Link with feature engineering  
KANs can reveal useful transformations that can then be injected into an MLP or econometric model.

---

## 6. Limitations

❌ Slower than MLPs (function evaluation)  
❌ Harder to train  
❌ Sensitive to function parameterization  
❌ Less hardware-optimized (GPU)

---

## 7. Typical Use Cases

KANs shine when the problem is **structural rather than purely statistical**:

- Physical modeling  
- Nonlinear dynamics  
- Differentiable systems  
- Quantitative finance with hidden structures  
- PDE / ODE learning  

---

## 8. Summary

A **KAN** is not just another neural network:

> It is a model that learns **the functions themselves**, not just their combinations.

It sits between:
- neural networks,
- functional approximation,
- and automatic feature engineering.

---

## Key Sentence to Remember

> **An MLP learns weights.  
A KAN learns laws.**
