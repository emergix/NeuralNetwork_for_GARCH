### 1. **Hybrid GARCH-ANN Models**

#### Xu et al. (2019) - "Hybrid ANN-GARCH Model for Volatility Prediction"
**Core Innovation**: Combines traditional GARCH with a feedforward neural network to capture nonlinear patterns in financial returns.

**Model Structure**:
1. **GARCH Component**:
   $$\epsilon_t = \sigma_t z_t, \quad z_t \sim N(0,1)$$
   $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

2. **ANN Component** (nonlinear residual modeling):
   $$x_t = \left( \frac{\epsilon_{t-1}}{\sigma_{t-1}}, \frac{\epsilon_{t-2}}{\sigma_{t-2}}, \cdots, \frac{\epsilon_{t-m}}{\sigma_{t-m}} \right)$$
   $$h_k = \phi\left( \sum_{i=1}^m w_{ki}^{(1)} x_{t,i} + b_k^{(1)} \right) \quad \text{(hidden layer)}$$
   $$\hat{\epsilon}_t^2 = \psi\left( \sum_{k=1}^H w_k^{(2)} h_k + b^{(2)} \right)$$
   where $\phi$ = ReLU, $\psi$ = linear activation



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


   

3. **Hybrid Integration**:
   $$\sigma_{t,\text{hybrid}}^2 = \underbrace{\omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2}_{\text{GARCH}} + \underbrace{\gamma \hat{\epsilon}_t^2}_{\text{ANN}}$$
   with $\gamma$ controlling ANN contribution

**Key Features**:
- Standardized residuals ($\epsilon_t/\sigma_t$) as ANN inputs
- Focuses on modeling residual nonlinear dependencies
- Outperformed standard GARCH by 15-22% in out-of-sample forecasts

---

#### Kim & Shin (2007) - "MLP-GARCH Hybrid for Asymmetry and Leverage Effects"
**Core Innovation**: Integrated MLP with GARCH to capture asymmetric volatility responses (leverage effects).

**Model Architecture**:
1. **Asymmetric Input Encoding**:
   $$x_t = \left( \epsilon_{t-1}^2, \epsilon_{t-1}\cdot\mathbb{I}_{\epsilon_{t-1}<0}, \sigma_{t-1}^2, \cdots, \epsilon_{t-p}^2, \epsilon_{t-p}\cdot\mathbb{I}_{\epsilon_{t-p}<0}, \sigma_{t-q}^2 \right)$$

2. **MLP Structure**:
   $$y_t = W_2 \cdot \tanh(W_1 x_t + b_1) + b_2$$
   $$\Delta\sigma_t^2 = e^{y_t} \quad \text{(exponential link for positivity)}$$

3. **GARCH-MLP Integration**:
   $$\sigma_t^2 = \underbrace{\omega + \sum_{i=1}^p \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^q \beta_j \sigma_{t-j}^2}_{\text{GARCH core}} + \underbrace{\delta \Delta\sigma_t^2}_{\text{MLP asymmetry correction}}$$

**Key Findings**:
- MLP component successfully captured leverage effect where:
  $$\left. \frac{\partial \sigma_t^2}{\partial \epsilon_{t-1}} \right|_{\epsilon_{t-1}<0} > \left. \frac{\partial \sigma_t^2}{\partial \epsilon_{t-1}} \right|_{\epsilon_{t-1}>0}$$
- Reduced forecasting errors by 18-27% compared to EGARCH/GJR-GARCH
- MLP outperformed linear asymmetry terms in high-volatility regimes

---

### Comparative Analysis
| **Feature**               | Xu et al. (2019)           | Kim & Shin (2007)         |
|---------------------------|----------------------------|---------------------------|
| **ANN Role**              | Residual pattern modeling  | Asymmetry capture         |
| **Key Inputs**            | Standardized residuals     | Signed return lags        |
| **Activation**            | ReLU (hidden), Linear (out)| tanh (hidden)             |
| **Positivity Enforcement**| Additive correction        | Exponential output link   |
| **GARCH Integration**     | Additive combination       | Augmented GARCH equation  |
| **Primary Improvement**   | Forecast accuracy          | Leverage effect capture   |





**(A) ANN as a nonlinear mean model**  
