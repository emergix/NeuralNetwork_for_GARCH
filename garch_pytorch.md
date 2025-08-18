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

## 🐍 Example Python Code

```python
__all__ = ["GARCH11", "negative_log_likelihood"]

import torch
import torch.nn as nn
import torch.optim as optim

class GARCH11(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.8))

    def forward(self, returns):
        T = len(returns)
        sigma2 = torch.zeros_like(returns)
        sigma2[0] = returns.var()
        for t in range(1, T):
            sigma2[t] = (
                self.omega +
                self.alpha * returns[t-1]**2 +
                self.beta * sigma2[t-1]
            )
        return sigma2

def negative_log_likelihood(model, returns):
    sigma2 = model(returns)
    loglik = 0.5 * (torch.log(sigma2) + (returns**2) / sigma2)
    return loglik.sum()
```

## 🐍 Example of use of the  Code

You need first to put in a file named garch_model.py the preceding code.
Then in a jupyter notebook you can use the model, by copying the following:

```python

import torch
from garch_model import GARCH11, negative_log_likelihood
import matplotlib.pyplot as plt

# Generate synthetic returns (e.g., normal noise with volatility clustering)
torch.manual_seed(0)
T = 500
returns = torch.randn(T) * 0.02

# Initialize and train the model
model = GARCH11()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

n_epochs = 1000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = negative_log_likelihood(model, returns)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Get fitted conditional variances
sigma2 = model(returns).detach().numpy()

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(returns.numpy(), label="Returns", alpha=0.6)
plt.plot(sigma2**0.5, label="Estimated Volatility (σ)", color="orange")
plt.title("GARCH(1,1) Model Fit on Simulated Returns")
plt.legend()
plt.tight_layout()
plt.show()


```


## 📎 Features You Can Add

- Student-t innovations (heavy tails),
- Parameter constraints: \( \omega > 0 \), \( \alpha \geq 0 \), \( \beta \geq 0 \), \( \alpha + \beta < 1 \),
- LSTM-GARCH hybrids or Bayesian estimation with Pyro.

---

## 📂 Files Included

- `garch_pytorch.md`: This documentation.
- `garch_model.py`: PyTorch implementation of GARCH(1,1) with optimization loop.
