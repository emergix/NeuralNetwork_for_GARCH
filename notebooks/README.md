# 📓 Notebooks Directory

This directory contains practical implementations of Neural Network-based GARCH calibration methods. Each notebook demonstrates a complete workflow from data processing to real-world financial applications.

---

## 🧠 Available Notebooks

### 1. GARCH Calibration with Neural Networks

## 📊 The Heston Stochastic Volatility Model

The **Heston model** is a widely used stochastic volatility model in quantitative finance, particularly for pricing derivatives. Unlike Black-Scholes, which assumes constant volatility, Heston introduces a **random process for volatility itself**, capturing market phenomena such as the volatility smile and clustering.

### 🧮 Model Dynamics

The asset price \( S_t \) and its variance \( v_t \) follow the system of stochastic differential equations:

\[
\begin{aligned}
dS_t &= \mu S_t\,dt + \sqrt{v_t} S_t\,dW_t^S \\\\
dv_t &= \kappa(\theta - v_t)\,dt + \sigma \sqrt{v_t}\,dW_t^v
\end{aligned}
\]

- \( \mu \) : drift of the asset  
- \( v_t \) : instantaneous variance  
- \( \kappa \) : rate of mean reversion  
- \( \theta \) : long-term variance  
- \( \sigma \) : volatility of volatility  
- \( W_t^S \), \( W_t^v \) : Brownian motions with correlation \( \rho \)

### 🔍 Key Features

- **Mean-reverting variance** captures realistic volatility behavior  
- **Closed-form solution** for European options via Fourier inversion  
- **Flexible calibration** to volatility surfaces (smile/skew)

### 📄 Calibration Example

See the full calibration process using historical Air Liquide data:

👉 [`README_GARCH_Calibration.md`](./README_GARCH_Calibration.md)


### 📚 Applications

The Heston model is commonly used for:
- Pricing European and exotic options  
- Risk management and scenario analysis  
- Calibrating implied volatility surfaces



[`garch_calibration_pytorch.ipynb`](./garch_calibration_pytorch.ipynb)  
*End-to-end workflow for calibrating GARCH parameters using neural networks*

Key features:
- Data preprocessing for financial time series
- Neural network architecture design (LSTM/GRU)
- Model training and validation
- Real-time calibration on streaming data
- Performance benchmarking vs traditional methods

```python
# Core calibration workflow
import tensorflow as tf
from arch import arch_model

# Neural network calibration
nn_model = tf.keras.Sequential([...])
garch_params = nn_model.predict(streaming_data)

# Feed to stochastic model
heston_model.calibrate(initial_params=garch_params[['VL','persistence']])

# Portfolio optimization
optimizer.run(volatility_forecast=garch_params['conditional_volatility'])
