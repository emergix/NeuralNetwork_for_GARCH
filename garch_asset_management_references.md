# Literature Review: GARCH Models in Asset Management

## 🧠 Introduction

GARCH models have long been instrumental in volatility modeling and have found a range of applications in asset management. Recent studies show their effectiveness when extended to multivariate forms, hybridized with other models, or estimated via Bayesian methods. This review summarizes five key academic contributions that demonstrate the value of GARCH models in tactical and strategic portfolio allocation.

---

## 📘 1. Comparative Study of Multivariate GARCH Models for Portfolio Optimization

**Reference**: [ScienceDirect, 2018](https://www.sciencedirect.com/science/article/pii/S106294081830038X)

This paper evaluates several multivariate GARCH models—including DCC-GARCH—for portfolio optimization using empirical data. It compares these models based on risk-adjusted performance metrics such as:
- Sharpe ratio,
- VaR,
- realized volatility.

**Conclusion**: Dynamic models such as DCC-GARCH significantly improve portfolio performance relative to static approaches.

---

## 📘 1.Engle (2002)** – introduced the **DCC-GARCH** model, widely used for asset allocation.

  📄 Details in  : [engle_2002_dcc_garch.md](./engle_2002_dcc_garch.md)
  
---


## 📗 2. Flavin & Wickens (2000) – Tactical Asset Allocation Using Multivariate GARCH

**Citation**: Flavin, T. J., & Wickens, M. R. (2000). _A multivariate GARCH model for predicting portfolio risk_. Journal of Empirical Finance.

This study introduces a tactical asset allocation strategy using multivariate GARCH forecasts of time-varying covariance matrices. Their method shows:
- Reduced realized risk (5% lower) compared to constant covariance strategies,
- More efficient portfolio rebalancing under changing market regimes.

---

## 📙 3. Kinoshita (2015) – GARCH Filtering with Higher Moments for Allocation

**Reference**: [SpringerLink](https://link.springer.com/article/10.1007/s00181-014-0871-1)

This paper incorporates higher-order moments (skewness and kurtosis) into the volatility estimation process using GARCH filters. Results show:
- Better capture of asymmetry in asset returns,
- Outperformance versus classic mean-variance strategies.

---

## 📕 4. Bayesian Copula-GARCH for Asset Allocation

**Summary**: This study proposes a **Bayesian Copula-GARCH** framework for portfolio selection, using Metropolis-within-Gibbs sampling. The approach handles:
- Parameter uncertainty explicitly,
- Tail dependence between assets.

**Outcome**: Portfolios exhibit lower realized volatility and reduced exposure to high-risk assets under high risk aversion levels.

---

## 📒 5. Orthogonal Non-Elliptical GARCH for Robust Portfolio Selection

**Reference**: [ScienceDirect, 2021](https://www.sciencedirect.com/science/article/pii/S0378426621000042)

This paper introduces an orthogonal GARCH model that does not rely on elliptical distribution assumptions. It is used to:
- Forecast multivariate volatility robustly,
- Optimize portfolios while avoiding excessive rebalancing.

**Strength**: Captures non-Gaussian behavior in returns for better risk-adjusted allocation.


---

## 📌 Summary Table

| Reference | Model | Key Contribution |
|----------|-------|------------------|
| Comparative Study (2018) | DCC-GARCH & others | Outperformance in Sharpe & VaR |
| Flavin & Wickens (2000) | Multivariate GARCH | Tactical allocation with risk reduction |
| Kinoshita (2015) | GARCH + higher moments | Allocation under asymmetry & excess kurtosis |
| Bayesian Copula-GARCH | Copula + GARCH (Bayesian) | Uncertainty-aware robust allocation |
| Orthogonal GARCH (2021) | Non-elliptical GARCH | Robust covariance modeling |

