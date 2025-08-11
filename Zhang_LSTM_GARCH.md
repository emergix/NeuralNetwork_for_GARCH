### **LSTM-GARCH Hybrid Model – Zhang et al. (2020)**

Zhang et al. (2020) proposed a hybrid framework combining **GARCH** and **Long Short-Term Memory (LSTM)** networks to exploit their complementary strengths:

- **GARCH** component models *short-term volatility clustering* and captures autoregressive conditional heteroskedasticity.
- **LSTM** component captures *long-term temporal dependencies* and nonlinear patterns in volatility not explained by GARCH.

---

#### **1. GARCH Component**

The conditional variance from the GARCH(1,1) process is computed as:

$$\sigma_t^2 = \omega + \alpha  \varepsilon_{t-1}^2 + \beta  \sigma_{t-1}^2$$

where:
- $\sigma_t^2$ = conditional variance at time $t$
- $\varepsilon_{t-1}$ = lagged residual
- $\omega > 0$, $\alpha \geq 0$, $\beta \geq 0$ = parameters satisfying $\alpha + \beta < 1$

---



#### **2. LSTM Component**

The LSTM network receives a sequence of past returns $\{ r_{t-k}, \dots, r_{t-1} \}$ or past GARCH volatilities $\{ \sigma_{t-k}^2, \dots, \sigma_{t-1}^2 \}$ and learns to produce an adjusted volatility forecast.

LSTM cell update equations:

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) & \text{(forget gate)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) & \text{(input gate)} \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) & \text{(candidate cell state)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t & \text{(cell state update)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) & \text{(output gate)} \\
h_t &= o_t \odot \tanh(C_t) & \text{(hidden state)}
\end{aligned}
$$

---

#### **3. Hybrid Forecast**

The final volatility prediction is a weighted combination of GARCH and LSTM outputs:

$$\hat{\sigma}_t^2 = \lambda  \sigma_{t,\text{GARCH}}^2 + (1 - \lambda)  \sigma_{t,\text{LSTM}}^2$$

where $\lambda \in [0,1]$ is tuned via validation.

---

#### **Key Insights**
- The GARCH model captures immediate volatility clustering effects.
- The LSTM network accounts for nonlinear, long-memory patterns.
- The hybrid improves **out-of-sample volatility forecasting accuracy**, especially in high-volatility regimes.

---

## **Practical Use of LSTM-GARCH Hybrid Models on a Derivatives Trading Desk**

A trading desk can leverage the LSTM-GARCH hybrid model for **volatility forecasting**, **risk management**, and **pricing adjustments**.  
The main idea is to combine short-term market dynamics (from GARCH) with long-term nonlinear patterns (from LSTM) to improve forecasts of implied and realized volatility.

---

### **1. Volatility Surface Forecasting**
- **Objective:** Predict the *future* dynamics of the implied volatility surface (IVS) for options.
- **How:**  
  1. Use GARCH to model near-term volatility clustering.  
  2. Feed historical IVS slices (or realized volatility) into the LSTM to capture regime shifts and slow-moving patterns.
- **Application:**  
  - Anticipate skew/smile changes.  
  - Adjust delta, vega, and gamma hedges preemptively.  
  - Identify opportunities where market-implied vol deviates from model-forecasted vol.

---

### **2. Trading Variance and Volatility Swaps**
- **Objective:** Identify mispricings between implied variance (from options) and forecasted realized variance.
- **How:**  
  - Use LSTM-GARCH to forecast the **realized variance** over the swap horizon.  
  - Compare with market-implied variance from variance swap quotes.
- **Application:**  
  - Go long/short variance swaps when the spread between implied and model-forecasted variance exceeds a threshold.  
  - Hedge exposure using options or volatility futures.

---

### **3. Exotic Option Pricing Adjustments**
- **Objective:** Improve pricing and hedging of path-dependent derivatives (barriers, autocallables, cliquets).
- **How:**  
  - Replace constant-vol or flat smile assumptions in pricing models with volatility forecasts from LSTM-GARCH.
- **Application:**  
  - More accurate Monte Carlo simulations with time-varying vol forecasts.  
  - Better estimation of hitting probabilities for barriers.

---

### **4. Intraday Risk Management**
- **Objective:** Anticipate intraday volatility spikes that could affect Greeks and margin requirements.
- **How:**  
  - Feed intraday data into a short-horizon LSTM-GARCH variant.
- **Application:**  
  - Adjust hedge ratios in anticipation of volatility jumps.  
  - Set tighter risk limits before macro announcements.

---

### **5. Statistical Arbitrage on Volatility Products**
- **Objective:** Trade VIX futures, variance futures, or listed volatility ETFs.
- **How:**  
  - Forecast short-term realized vol and compare with VIX term structure.
- **Application:**  
  - Long/short calendar spreads when mispricing is detected.  
  - Capture mean-reversion in volatility.

---

### **Implementation Notes**
- **Data Sources:**  
  - Historical returns, realized volatilities, and option-implied volatilities.
- **Integration:**  
  - Model runs daily (overnight) for strategic positioning.  
  - Intraday refresh for high-frequency volatility-sensitive products.
- **Validation:**  
  - Backtest on historical derivative P&L.  
  - Monitor live forecast error to recalibrate.
- **Risk:**  
  - Model drift in high-volatility regimes.  
  - Overfitting risk if too many LSTM parameters.

---

**Bottom line:**  
By combining GARCH’s short-term clustering with LSTM’s long-term memory, a derivatives desk can improve **volatility forecasting**, leading to **better pricing, hedging, and risk management**—especially for products where volatility is the primary risk driver.
