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

... (content truncated for brevity in code, full text would be here) ...
