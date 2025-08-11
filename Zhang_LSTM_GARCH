### **LSTM-GARCH Hybrid Model – Zhang et al. (2020)**

Zhang et al. (2020) proposed a hybrid framework combining **GARCH** and **Long Short-Term Memory (LSTM)** networks to exploit their complementary strengths:

- **GARCH** component models *short-term volatility clustering* and captures autoregressive conditional heteroskedasticity.
- **LSTM** component captures *long-term temporal dependencies* and nonlinear patterns in volatility not explained by GARCH.

---

#### **1. GARCH Component**

The conditional variance from the GARCH(1,1) process is computed as:

$$
\sigma_t^2 = \omega + \alpha \, \epsilon_{t-1}^2 + \beta \, \sigma_{t-1}^2
$$

where:

- $\sigma_t^2$ = conditional variance at time $t$
- $\epsilon_{t-1}^2$ = lagged squared residual
- $\omega > 0$, $\alpha \geq 0$, $\beta \geq 0$ = parameters satisfying $\alpha + \beta < 1$

---

#### **2. LSTM Component**

The LSTM network receives a sequence of past returns $\{ r_{t-k}, \dots, r_{t-1} \}$ or past GARCH volatilities $\{ \sigma_{t-k}^2, \dots, \sigma_{t-1}^2 \}$ and learns to produce an adjusted volatility forecast.

LSTM cell update equations:

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) &\text{(forget gate)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) &\text{(input gate)} \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) &\text{(candidate cell state)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t &\text{(cell state update)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) &\text{(output gate)} \\
h_t &= o_t \odot \tanh(C_t) &\text{(hidden state)}
\end{aligned}
$$

---

#### **3. Hybrid Forecast**

The final volatility prediction is a weighted combination of GARCH and LSTM outputs:

$$
\hat{\sigma}_t^2 = \lambda \, \sigma_{t,\text{GARCH}}^2 + (1 - \lambda) \, \sigma_{t,\text{LSTM}}^2
$$

where $\lambda \in [0,1]$ is tuned via validation.

---

#### **Key Insights**

- The GARCH model captures immediate volatility clustering effects.
- The LSTM network accounts for nonlinear, long-memory patterns.
- The hybrid improves **out-of-sample volatility forecasting accuracy**, especially in high-volatility regimes.
