
# 📈 GARCH(1,1) Calibration: Air Liquide Time Series

This section describes how we calibrate a **GARCH(1,1)** model to the historical opening prices of the **Air Liquide** stock (from 1990 to 2025), and how to interpret the estimated volatility dynamics.

---

## ⚙️ Step-by-Step Calibration Process

### 1. **Data Preparation**
We load a CSV file containing daily **opening prices** and compute the daily log-returns:
```python
df["log_return"] = 100 * (df["Open"] / df["Open"].shift(1)).apply(np.log)
```
> Rescaling the returns ×100 ensures better numerical convergence of the optimizer.

---

### 2. **Model Definition**
We specify a **GARCH(1,1)** model with normal residuals using the `arch` Python library:
```python
from arch import arch_model

model = arch_model(returns, vol="GARCH", p=1, q=1, dist="normal")
result = model.fit(disp="off")
```

---

## 🧠 Interpreting the Results

The model estimates the following parameters:

### Mean Equation:
| Parameter | Estimate | Meaning |
|-----------|----------|---------|
| `mu`      | ~0.0246  | Average daily return (rescaled) |

### Volatility Equation:
The conditional variance follows the GARCH(1,1) process:

\[
\sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2
\]

| Parameter | Estimate | Meaning |
|-----------|----------|---------|
| `omega`   | 7.14e-3  | Long-term volatility floor |
| `alpha[1]`| 2.87e-2  | Impact of past squared residuals (short-term shocks) |
| `beta[1]` | 0.9650   | Persistence of past volatility (volatility clustering) |

---

## 🔍 Key Insights

- **Persistence**: The sum `alpha + beta ≈ 0.993` shows high persistence — volatility shocks decay slowly, which is common in financial time series.
- **Low omega**: Suggests that volatility reverts to a low unconditional level.
- **Non-zero mu**: Daily drift is positive but very small, consistent with long-term market growth.

---

## 📊 Next Steps

- Plot the conditional volatility over time:
```python
plt.plot(result.conditional_volatility)
plt.title("Estimated Volatility (GARCH)")
```

- Compare the GARCH forecast with realized volatility or implied volatility from options.

- Extend to `t-distributed` GARCH or `EGARCH` if asymmetric shocks are observed.

---

## 📁 Files Used

- `Data/air_liquide.csv` — input time series (Open prices)
- `notebooks/calibration_garch.ipynb` — calibration notebook

---

## ✍️ Author

Olivier Croissant — [@croissant-olivier](https://github.com/croissant-olivier)
