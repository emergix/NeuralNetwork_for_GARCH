# Hybrid GARCH–ANN Models: What the Papers Do (with Formulas)

> **Note.** There are multiple papers that could match “Xu et al. (2019)” and “Kim & Shin (2007).” The equations below capture the canonical formulations used in those works (GARCH for conditional variance + MLP/ANN to capture nonlinearities, asymmetry, or regime effects). If you share the exact titles/links/DOIs, I’ll align the notation precisely to each paper and update this section accordingly.

---

## 1) Baseline GARCH Setup
Let \((r_t)\) be (demeaned or mean‑modeled) returns.

- **Observation equation**
  \[ r_t = \mu_t + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t, \; z_t \sim i.i.d.\; \mathcal{D}(0,1) \]  
  where \(\mathcal{D}\) is typically \(\mathcal{N}(0,1)\) or Student‑t(\(\nu\)).

- **GARCH(\(p,q\)) variance recursion**
  \[ \sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i\, \varepsilon_{t-i}^2 + \sum_{j=1}^q \beta_j\, \sigma_{t-j}^2, \qquad \omega>0,\; \alpha_i,\beta_j\ge 0. \]
  
- **Asymmetric variants** (often used by Kim & Shin‑style hybrids):
  - **GJR‑GARCH(\(p,q\))**: \[ \sigma_t^2 = \omega + \sum_{i=1}^p (\alpha_i + \gamma_i \mathbb{1}_{\{\varepsilon_{t-i}<0\}})\,\varepsilon_{t-i}^2 + \sum_{j=1}^q \beta_j\,\sigma_{t-j}^2. \]  
  - **EGARCH(\(p,q\))** (log‑variance): \[ \log \sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i\Big(\frac{|\varepsilon_{t-i}|}{\sigma_{t-i}} - \mathrm{E}\Big[\frac{|z|}{1}\Big]\Big) + \sum_{i=1}^p \gamma_i \frac{\varepsilon_{t-i}}{\sigma_{t-i}} + \sum_{j=1}^q \beta_j \log \sigma_{t-j}^2. \]

- **Log‑likelihood** (Gaussian):
  \[ \ell(\Theta) = -\tfrac{1}{2}\sum_{t=1}^T \Big( \log(2\pi) + \log \sigma_t^2 + \tfrac{\varepsilon_t^2}{\sigma_t^2} \Big), \]
  with \(\Theta\) collecting all parameters; replace the kernel for Student‑t.

## 2) ANN Component

Let $x_t \in \mathbb{R}^d$ be a feature vector (e.g., lags of returns $r_{t-k}$, absolute/signed lags $|r_{t-k}|$, $\text{sgn}(r_{t-k})$, realized measures, volume, macro factors). A standard feed-forward MLP with one hidden layer is:

$$
h_t = \phi(W_1 x_t + b_1), \quad y_t = W_2 h_t + b_2
$$

where $\phi \in \{\tanh, \text{ReLU}, \text{ELU}\}$. Deeper MLPs stack this mapping.

Where the ANN enters the volatility model (common hybrid patterns):

**(A) ANN as a nonlinear mean model**  
$r_t = m(x_t; \vartheta) + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim D(0,1)$  

with $m(\cdot; \vartheta)$ an MLP and $\sigma_t^2$ following a GARCH-type recursion. Removing nonlinear mean structure can sharpen variance dynamics.

**(B) ANN as an additive or multiplicative correction to GARCH variance**  

Baseline GARCH($p,q$) variance:  
$h_t^{\text{GARCH}} \equiv \omega + \sum_{i=1}^p \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^q \beta_j \sigma_{t-j}^2, \quad \sigma_t^2 > 0$  

Add an ANN term $g(x_t; \vartheta)$:  

- **Additive (level):**  
  $\sigma_t^2 = h_t^{\text{GARCH}} + g(x_t; \vartheta), \quad g(x) \geq 0$  
  with positivity enforced via a link, e.g. $g(x) = \text{softplus}(y_t) = \log(1 + e^{y_t})$.  

- **Multiplicative (log-variance):**  
  $\log \sigma_t^2 = \log h_t^{\text{GARCH}} + y_t, \quad y_t = \text{MLP}(x_t)$  
  A numerically stable alternative is:  
  $\log \sigma_t^2 = h_t^{\text{lin}} + y_t$  
  with $h_t^{\text{lin}}$ a linear GARCH-style ARMA in $\log \sigma^2$.

**(C) ANN to capture asymmetry/leverage**  
Encode sign effects explicitly in inputs:  
$x_t = (\epsilon_{t-1}^2, \epsilon_{t-1} \cdot 1_{\{\epsilon_{t-1} < 0\}}, \sigma_{t-1}^2, |\epsilon_{t-1}|, \dots)$  

then map via $y_t = \text{MLP}(x_t)$ and set either:  
$\sigma_t^2 = h_t^{\text{GARCH}} + \text{softplus}(y_t) \quad \text{or} \quad \log \sigma_t^2 = \omega + y_t$  

This subsumes GJR/EGARCH-type asymmetries when the MLP is linear.

**(D) ANN-only volatility with GARCH-inspired inputs**  
$\sigma_t^2 = f(\epsilon_{t-1}^2, \dots, \sigma_{t-1}^2, \dots; \vartheta)$  
where $f$ is an MLP; classical GARCH is recovered by restricting $f$ to be linear and nonnegative.

---

### Design notes

- **Positivity**: Prefer log-variance parameterizations or positive links (softplus/exponential) on ANN outputs.  
- **Stationarity of the backbone**: Maintain $\sum_i \alpha_i + \sum_j \beta_j < 1$ for the GARCH component.  
- **Features**: Include realized volatility (RV), bipower variation, or option-implied measures when available; standardize inputs.  
- **Alternatives**: Replacing the MLP with an RNN/LSTM is straightforward, but most cited hybrids use MLPs for simplicity.