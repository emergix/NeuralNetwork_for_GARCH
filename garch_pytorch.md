# GARCH(1,1) Calibration Using PyTorch

## 🧠 Overview

The GARCH(1,1) model is commonly used for modeling volatility in financial time series:

![GARCH Equation](./assets/images/garch_equation_1.jpg)



This document explains how to implement and calibrate this model using **PyTorch**, leveraging its automatic differentiation and optimization tools.

---

## ⚙️ Log-Likelihood Function

Under Gaussian errors, the negative log-likelihood for returns \( r_t \) is:

![Log-Likelihood Equation](./assets/images/garch_equation_2.jpg)


This function is minimized using gradient descent.

---

## ✅ Key Steps

1. **Define a PyTorch model with parameters**: omega, alpha, beta.
2. **Compute conditional variances \( \sigma_t^2 \)** recursively.
3. **Minimize the negative log-likelihood** with an optimizer like Adam or LBFGS.
4. **Optionally**: enforce constraints using `softplus()` or clamping.

---

## 📎 Features You Can Add

- Student-t innovations (heavy tails),
- Parameter constraints: \( \omega > 0 \), \( \alpha \geq 0 \), \( \beta \geq 0 \), \( \alpha + \beta < 1 \),
- LSTM-GARCH hybrids or Bayesian estimation with Pyro.

---

## 📂 Files Included

- `garch_pytorch.md`: This documentation.
- `garch_model.py`: PyTorch implementation of GARCH(1,1) with optimization loop.
