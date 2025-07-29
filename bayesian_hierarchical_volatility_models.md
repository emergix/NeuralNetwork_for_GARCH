# Literature Review: Bayesian Hierarchical Frameworks in Volatility Modeling

## 🧠 Introduction

Bayesian hierarchical frameworks have become increasingly popular in financial econometrics, particularly in the context of volatility modeling. These approaches offer a flexible and probabilistically coherent way to model complex financial data structures, incorporate prior knowledge, and quantify uncertainty in a principled manner.

In contrast to classical estimation methods (e.g., MLE), Bayesian hierarchical models provide **full posterior distributions** of parameters and latent variables, making them highly suitable for forecasting and risk management.

---

## 🧱 What Is a Bayesian Hierarchical Model?

A hierarchical Bayesian model decomposes the modeling process into **multiple levels**:

- **Level 1**: Observation model (e.g., returns conditioned on volatility),
- **Level 2**: Latent structure (e.g., GARCH parameters, volatility states),
- **Level 3**: Hyperpriors (prior distributions on model parameters).

This layered structure allows:
- **Parameter sharing** across assets or time,
- **Pooling of information** for sparse data settings,
- Modeling **regime switches**, **structural breaks**, or **random effects**.

---

## 🔬 Key Contributions and Approaches

### 1. **Bayesian Estimation of GARCH Models**
- **Ardia (2008)** – Introduced a full Bayesian approach for univariate and multivariate GARCH models using MCMC methods.
- **Vrontos et al. (2003)** – Developed Bayesian inference for MGARCH models with stochastic volatility.

### 2. **Hierarchical GARCH Models**
- **Jacquier et al. (2004)** – Modeled volatility using latent stochastic processes with hierarchical priors.
- **Chan et al. (2006)** – Used a hierarchical structure for modeling time-varying parameters across multiple markets or asset classes.

### 3. **Dynamic Shrinkage and Sparsity**
- **Bitto & Frühwirth-Schnatter (2019)** – Proposed Bayesian shrinkage priors (e.g., horseshoe) in time-varying volatility models.
- **Griffin & Steel (2010)** – Developed models with sparsity-inducing priors to identify structural changes.

---

## 🛠 Methodologies and Inference Techniques

- **Markov Chain Monte Carlo (MCMC)**: Gibbs sampling, Metropolis-Hastings, Hamiltonian Monte Carlo.
- **Variational Bayes (VB)**: Faster but approximate posterior inference.
- **Bayesian Model Averaging (BMA)**: Combines forecasts from different volatility models.
- **State-Space and Particle Filters**: For sequential learning in time-series.

---

## 📊 Evaluation Metrics

- **Predictive performance** using log score or posterior predictive checks,
- **Value-at-Risk (VaR) coverage tests**,
- **Bayes factors** and posterior model probabilities for comparison.

---

## 📈 Applications in Finance

- **Multivariate volatility modeling** with cross-sectional structure,
- **Portfolio allocation** under parameter uncertainty,
- **Stress testing** with probabilistic scenarios,
- **Macro-financial forecasting**.

---

## 📎 Selected References (BibTeX format)

```bibtex
@book{ardia2008financial,
  title={Financial Risk Management with Bayesian Estimation of GARCH Models},
  author={Ardia, David},
  year={2008},
  publisher={Springer}
}

@article{vrontos2003bayesian,
  title={Bayesian inference for GARCH models using Gibbs sampling},
  author={Vrontos, Ioannis D. and Dellaportas, Petros and Politis, Dimitris N.},
  journal={Computational Statistics & Data Analysis},
  year={2003}
}

@article{jacquier2004bayesian,
  title={Bayesian analysis of stochastic volatility models with fat tails and correlated errors},
  author={Jacquier, Eric and Polson, Nicholas G. and Rossi, Peter E.},
  journal={Journal of Econometrics},
  year={2004}
}

@article{bitto2019bayesian,
  title={Achieving shrinkage in a time-varying parameter model framework},
  author={Bitto, Andreas and Frühwirth-Schnatter, Sylvia},
  journal={Journal of Econometrics},
  year={2019}
}
```
