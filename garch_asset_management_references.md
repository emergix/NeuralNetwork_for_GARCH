# 📚 Unified Literature Review: GARCH Models in Asset Management

This document presents a curated and standardized review of key academic contributions related to the use of GARCH and hybrid GARCH models in asset management. These works span traditional econometric methods, Bayesian approaches, and cutting-edge neural network-enhanced models.

---

## 🟥 1. Comparative Study of Multivariate GARCH Models for Portfolio Optimization

**Reference**: [ScienceDirect, 2018](https://www.sciencedirect.com/science/article/pii/S106294081830038X)

This study evaluates several multivariate GARCH models—including DCC-GARCH—for portfolio optimization using empirical financial data. The models are compared based on:
- Sharpe ratio,
- Value-at-Risk (VaR),
- Realized volatility.

**Conclusion**: Dynamic models like DCC-GARCH consistently outperform static covariance methods in risk-adjusted performance.

📄 Details in: [`engle_2002_dcc_garch.md`](./engle_2002_dcc_garch.md)

---

## 🟥 2. Flavin & Wickens (2000) – Tactical Asset Allocation Using Multivariate GARCH

**Citation**: Flavin, T. J., & Wickens, M. R. (2000). _A multivariate GARCH model for predicting portfolio risk_. *Journal of Empirical Finance*.

Introduces a tactical allocation strategy based on GARCH-forecasted time-varying covariance matrices. Results show:
- ~5% reduction in realized portfolio risk,
- Improved rebalancing efficiency during regime shifts.

---

## 🟥 3.Ryo Kinoshita (2014) Asset allocation under higher moments with the GARCH filter
**Reference**: [SpringerLink](https://link.springer.com/article/10.1007/s00181-014-0871-1)

This paper extends GARCH filtering to include higher-order moments (skewness, kurtosis). Key findings:
- Enhanced capture of asymmetric return behavior,
- Better performance than classical mean-variance frameworks.

---

## 🟥 4. Long Kang (2009). Asset Allocation in a Bayesian Copula-Garch Framework: An Application to the 'Passive Funds versus Active Funds' Problem

**Reference**: [ssrn](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1355727)]

**Summary**: Proposes a Bayesian Copula-GARCH model with Metropolis-within-Gibbs sampling to model dependence and parameter uncertainty. The framework allows:
- Joint estimation of tail risk and marginal volatilities,
- Robust performance under risk-averse scenarios.

**Outcome**: Reduced portfolio volatility and improved resilience to tail events.

---

## 🟥 5. Orthogonal Non-Elliptical GARCH for Robust Portfolio Selection

**Reference**: [ScienceDirect, 2021](https://www.sciencedirect.com/science/article/pii/S0378426621000042)

Introduces an orthogonal GARCH model relaxing elliptical distribution assumptions. It is designed to:
- Capture non-Gaussian features of returns,
- Minimize rebalancing while preserving diversification.

---

## 🟥 6. Xu et al. (2024) – GINN: GARCH-Informed Neural Network

**Reference**: [arXiv](https://arxiv.org/html/2402.06642v1)

Presents GINN, a model that combines GARCH volatility dynamics with LSTM-based deep learning. Demonstrates:
- Superior R², MSE, and MAE in volatility prediction,
- Strong performance in turbulent regimes.

---

## 🟥 7. GARCHNet – LSTM-GARCH for VaR Forecasting

**Reference**: [Academia.edu](https://www.academia.edu/119434615/GARCHNet_Value_at_Risk_Forecasting_with_GARCH_Models_Based_on_Neural_Networks)

Builds a hybrid model that merges LSTM with classic GARCH for improved Value-at-Risk forecasting. Results:
- Better VaR backtesting accuracy,
- Tested across multiple equity indices.

---

## 🟥 8. Zhao et al. (2024) – From GARCH to Neural Networks

**Reference**: [arXiv](https://arxiv.org/html/2402.06642v1)

Explores theoretical and empirical links between GARCH dynamics and deep learning. Proposes:
- A GARCH-LSTM framework,
- Demonstrated improvements in regime detection and forecast stability.

---

## 🟥 9. NeuralGARCH – Yin & Barucca (2022)

**Reference**: [arXiv](https://arxiv.org/abs/2202.11285)

Builds RNN-based volatility models with variational Bayesian inference. Benefits:
- Adaptive modeling of time-varying volatility parameters,
- Robustness to heavy-tailed innovations via Student-t distribution.

---

## 🟥 10. Michańków et al. (2023) – GRU-GARCH Hybrid

**Reference**: [arXiv](https://arxiv.org/abs/2310.01063)

Combines Gated Recurrent Units (GRUs) with multiple GARCH structures (APARCH, EGARCH). Findings:
- Significant enhancement in VaR and Expected Shortfall metrics,
- Outperformance in equity, commodity, and crypto markets.

---

## 📌 Summary Table

| # | Reference | Model | Key Contribution |
|--:|-----------|--------|------------------|
| 1 | ScienceDirect (2018) | DCC-GARCH & others | Sharpe/Volatility-based portfolio outperformance |
| 2 | Flavin & Wickens (2000) | Multivariate GARCH | Tactical allocation and lower realized risk |
| 3 | Kinoshita (2015) | GARCH + higher moments | Allocation with skewness and kurtosis |
| 4 | Bayesian Copula-GARCH | Copula + GARCH | Tail-dependence modeling, Bayesian robustness |
| 5 | Orthogonal GARCH (2021) | Non-elliptical GARCH | Non-Gaussian volatility modeling |
| 6 | Xu et al. (2024) | GINN (GARCH + LSTM) | Superior volatility forecasting with LSTM |
| 7 | GARCHNet | LSTM + GARCH | VaR forecasting and risk metric improvements |
| 8 | Zhao et al. (2024) | GARCH-LSTM | Neural approximation of GARCH processes |
| 9 | Yin & Barucca (2022) | NeuralGARCH | Bayesian RNNs with time-varying dynamics |
|10 | Michańków et al. (2023) | GRU-GARCH | Enhanced VaR/ES in diverse markets |

