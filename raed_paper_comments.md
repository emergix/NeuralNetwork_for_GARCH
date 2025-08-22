# 📄 Commentaries on "Neural Network Method for GARCH Parameters Calibration"

## 1. Reminder of the Paper’s Approach

The paper by **Mohamed Raed Blel (2024)** proposes a novel method for calibrating the parameters of **GARCH(1,1)** models using **neural networks**.  
This approach differs from traditional optimization methods in several important ways:

- **Neural Network Calibration**  
  Instead of solving moment equations via constrained optimization, a deep neural network is trained to learn the mapping between statistical moments (E[x²], Γ₄, Γ₆) and the GARCH parameters (α₀, α₁, β₁).  

- **Mathematical Refinements**  
  The paper introduces a corrected formula for the autocovariance of squared returns (γₙ), which improves theoretical consistency and stabilizes the training process.  

- **Architecture & Training**  
  The author develops a multi-layer neural network with sinusoidal activations (instead of ReLU) and sigmoid outputs, ensuring parameter estimates remain in valid ranges. Training data are synthetically generated within admissible parameter domains.  

- **Comparison with Classical Methods**  
  The neural network is benchmarked against traditional minimization algorithms (SLSQP, Differential Evolution, Couenne, Random Search). While classical solvers struggle in irregular or noisy parameter spaces, the neural network achieves faster, more stable convergence.  

- **Robustness Tests**  
  Perturbation studies demonstrate that the neural network retains predictive accuracy under noisy data, highlighting its resilience compared to direct optimization methods.  

- **Applications**  
  The approach is extended to real time-series data, maximum likelihood estimation tests, and even SABR model calibration, showing both accuracy and superior computational efficiency.  

---

👉 Next step: we can add **your personal comments** in section 2 (strengths, limitations, extensions).  

First Correction to the formla (4) :

-  → Comments : [PDF.md](./Correction_GARCH6.pdf)
