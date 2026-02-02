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

# Calibration of GARCH(1,1) using a simple library 
👉 A very simple way to do a GARCH calibration is the following:

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
```



# Calibration of GARCH(1,1) using **Hybrid Neural Approach**: ANN (acov_multi) + $$\mu$$ via WLS + MLE (Gaussian & Student‑t)  

👉 The following notebook implements a calibration of GARCH idea , with a lot of refinements

[`garch_ann_full_pipeline_v2.ipynb`](./garch_ann_full_pipeline_v2.ipynb)  
*End-to-end workflow for calibrating GARCH parameters using neural networks*


This notebook explores a **hybrid approach** to calibrating a **GARCH(1,1)** model by combining:

- a **statistical** estimate of persistence $$\mu \approx \alpha_1 + \beta_1$$ from autocorrelation decay (WLS),
- an **ANN/MLP** trained to predict $$\alpha_1$$ from a **multi‑lag autocovariance/autocorrelation feature vector** (“acov_multi”),
- a final calibration and **benchmarks** using **MLE** (Gaussian and Student‑t), with **NLL/AIC** comparisons plus profiling over $$\nu$$ (degrees of freedom) and a heatmap over $$(\nu, \mu)$$.

---

## 1) Data: CSV input or simulated GARCH

The notebook first looks for a `return.csv` or `returns.csv` file.  
- If found: the **first numeric column** is loaded as the return series $$x_t$$.  
- Otherwise: a **simulated GARCH(1,1)** path (default “true” parameters) is generated as a sandbox.

In all cases, the series is **demeaned**.

---

## 2) Utilities: $$\hat\gamma_n$$, WLS($$\mu$$), and log‑likelihood (NLL)

### 2.1 Autocovariance / “gamma_hat”
The notebook defines:
- empirical autocovariance at a given lag,
- a function returning:
  - $$\hat\gamma_n$$ (variance‑normalized) for multiple lags,
  - empirical variance $$\hat\sigma^2$$.

> **Core idea**: build a multi‑lag “signature” of the series, used as **feature engineering** for parameter identification.

### 2.2 Estimating $$\mu$$ via WLS on $$\hat\gamma_n$$ decay
Two variants are tested:
- standard WLS regression of $$\log(\hat\gamma_n)$$ vs $$(n-1)$$,
- an alternative where weights depend on $$(T-n)$$ (effective sample size).

The model assumes:
$$
\hat\gamma_n \approx C\,\mu^{(n-1)} \Rightarrow \log \hat\gamma_n \approx \log C - (n-1)\log\mu
$$

> **Key point**: $$\mu$$ estimates persistence and is used to reconstruct $$\beta_1 = \mu - \alpha_1$$.

### 2.3 Gaussian & Student‑t NLL
The notebook implements Gaussian and Student‑t negative log‑likelihoods under constraints:
$$
\alpha_0>0, \; \alpha_1\ge0, \; \beta_1\ge0, \; \alpha_1+\beta_1 < 1
$$

---

## 3) Features: “acov_multi” + log‑variance

The feature vector uses lags $$3..16$$:
- $$g = (\hat\gamma_3, \ldots, \hat\gamma_{16})$$
- $$\log(\hat\sigma^2)$$

> **Idea tested**: multi‑lag structure + variance level contains enough information to infer $$\alpha_1$$.

---

## 4) Robust $$\alpha_1$$ proxy (Patch C): local multi‑start MLE

The training target is a **proxy** for $$\alpha_1$$ obtained on rolling sub‑windows:

1. estimate $$\mu$$ via WLS using multiple lag grids,
2. try several initial guesses for $$\alpha_1$$,
3. set $$\beta_1 = \mu - \alpha_1$$,
4. set $$\alpha_0 \approx \hat\sigma^2(1-\mu)$$,
5. minimize local Gaussian NLL.

> **Goal**: produce a stable proxy via multi‑lags + multi‑starts.

---

## 5) Dataset construction (Patches A/B)

- long windows: $$win = 768$$,
- dense step: $$step = 16$$.

Targets are **winsorized** between the 1% and 99% quantiles.

> **Idea**: more samples while keeping reliable local estimates and limiting extreme outliers.

---

## 6) Train/val split, scaling, ANN training

- 80/20 split,
- standardized features.

**Model:**
- 2 hidden layers (~192 units),
- ReLU,
- light Dropout (p=0.02),
- Sigmoid output to keep $$\alpha_1\in(0,1)$$.

Optimization via AdamW + LR scheduler.

---

## 7) Evaluation and calibration

Metrics:
- MSE, RMSE, $$R^2$$,
- scatter and residual histograms.

Final calibration of $$\alpha_1$$ predictions:
- isotonic regression if available,
- otherwise linear fallback.

---

## 8) Reconstruction of GARCH parameters

From ANN output:
$$
\hat\beta_1 = \hat\mu - \hat\alpha_1, \quad
\hat\alpha_0 = \hat\sigma^2(1-\hat\mu)
$$

---

## 9) Classical MLE baseline

Gaussian MLE and Student‑t MLE (including $$\nu$$).

---

## 10) Benchmark ANN vs MLE

### NLL & AIC comparison
Evaluation under Gaussian and Student‑t likelihoods.

### $$\nu$$ profile
Curve $$\nu \mapsto \text{NLL}_t$$.

### Heatmap $$(\nu,\mu)$$
Visualizes NLL landscape.

---

## Conceptual summary

Main ideas tested:

1. multi‑lag feature engineering,
2. decomposition: WLS for $$\mu$$ + ANN for $$\alpha_1$$,
3. robust local proxy via multi‑start MLE,
4. temporal data augmentation,
5. isotonic calibration,
6. statistical validation via NLL/AIC and sensitivity maps.

---

### Notebook outputs
- `garch_ann_student_t_eval_summary.csv`
Great for numerical comparison.



