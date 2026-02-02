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

### 🔍 Key Features

- **Mean-reverting variance** captures realistic volatility behavior  
- **Closed-form solution** for European options via Fourier inversion  
- **Flexible calibration** to volatility surfaces (smile/skew)

### 📄 Calibration Example

See the full calibration process using historical Air Liquide data:

👉 [`README_GARCH_Calibration.md`](./README_GARCH_Calibration.md)


# GARCH(1,1) — calibration «hybride» : **ANN (acov_multi)** + $$\mu$$ via WLS + MLE (Gauss & Student‑t)  
*(Notebook : `garch_ann_full_pipeline_v2.ipynb`, v2 “patches A–C”)*

Ce notebook explore une approche **hybride** pour calibrer un **GARCH(1,1)** en combinant :

- une estimation **statistique** de la persistance $$\mu \approx \alpha_1 + \beta_1$$ à partir de la décroissance des autocorrélations (WLS),
- une **ANN/MLP** entraînée à prédire $$\alpha_1$$ à partir d’un vecteur de **features de type autocovariance/autocorrélation multi‑lags** (“acov_multi”),
- une calibration finale et des **benchmarks** via **MLE** (Gauss et Student‑t), avec comparaisons **NLL/AIC** + profilage de $$\nu$$ (ddl) et heatmap $$(\nu,\mu)$$.

---

## 1) Données : lecture CSV ou simulation GARCH

Le notebook commence par chercher un fichier `return.csv` ou `returns.csv`.  
- Si trouvé : il charge la **première colonne numérique** comme série $$x_t$$ (rendements).  
- Sinon : il **simule** une trajectoire GARCH(1,1) (paramètres “vrais” par défaut), ce qui sert de sandbox pour valider le pipeline.

Dans tous les cas, la série est **centrée** (soustraction de la moyenne).

---

## 2) Utilitaires : $$\hat\gamma_n$$, WLS($$\mu$$) et log‑vraisemblance (NLL)

### 2.1 Autocovariance / “gamma_hat”
Le notebook définit :
- une **autocovariance empirique** à un lag donné,
- une fonction `gamma_hat_series(x, lags)` qui retourne :
  - $$\hat\gamma_n$$ (normalisé par la variance) pour plusieurs lags,
  - $$\hat\sigma^2$$ (variance empirique).

> **Idée** : construire un “empreinte” multi‑lags de la série, utilisée comme **feature engineering** pour identifier les paramètres GARCH.

### 2.2 Estimation de $$\mu$$ par WLS sur la décroissance des $$\hat\gamma_n$$
Deux variantes sont testées :
- `estimate_mu_wls(g, lags)` : régression pondérée sur $$\log(\hat\gamma_n)$$ vs $$(n-1)$$,
- `estimate_mu_wls_alt(g, lags, T_eff)` : variante où les poids dépendent de $$T - n$$ (effet “taille d’échantillon effective”).

Les formules codées reviennent à approximer une loi du type :
\[
\hat\gamma_n \approx C\,\mu^{(n-1)} \quad \Rightarrow\quad \log \hat\gamma_n \approx \log C - (n-1)\,\log\mu
\]
d’où $$\mu = e^{-\text{slope}}$$ après régression.

> **Point clé** : $$\mu$$ est ensuite utilisé comme estimation de la **persistance** $$\alpha_1+\beta_1$$, et sert à reconstruire $$\beta_1 = \mu-\alpha_1$$.

### 2.3 NLL Gaussienne & Student‑t
Le notebook implémente :
- `nll_gaussian(alpha0, alpha1, beta1, x)`  
- `nll_student(alpha0, alpha1, beta1, nu, x)`

en imposant des contraintes de validité :
\[
\alpha_0>0,\; \alpha_1\ge 0,\; \beta_1\ge 0,\; \alpha_1+\beta_1 < 1
\]

---

## 3) Lags / features : “acov_multi” (+ log‑variance)

Le vecteur de features est construit à partir d’un ensemble de lags (par défaut `lags = 3..16`) :

- $$g = (\hat\gamma_{3}, \ldots, \hat\gamma_{16})$$
- et un terme supplémentaire : $$\log(\hat\sigma^2)$$

> **Idée testée** : la structure multi‑lags + niveau de variance contient assez d’information pour “deviner” $$\alpha_1$$ (et indirectement $$\beta_1$$ via $$\mu$$).

---

## 4) Proxy $$\alpha_1$$ robuste (Patch C) : “local MLE” multi‑fenêtres / multi‑guesses

Pour entraîner l’ANN, il faut une cible $$y$$. Ici, $$y$$ est un **proxy** de $$\alpha_1$$ obtenu **localement** sur des sous‑fenêtres.

La fonction `estimate_alpha1_local_mle_robust(sub)` :
1. calcule $$\mu$$ via WLS sur **plusieurs sets de lags** (“multi‑fenêtres” côté lags),
2. lance plusieurs initialisations `guesses` pour $$\alpha_1$$,
3. pour chaque essai, impose $$\beta_1 = \mu - \alpha_1$$,
4. fixe $$\alpha_0 \approx \hat\sigma^2(1-\mu)$$ (reconstruction depuis la variance),
5. minimise une NLL gaussienne locale.

> **Idée testée** : produire un $$\alpha_1$$ proxy **stable** (moins sensible aux minima locaux et au bruit) grâce à :
- **multi‑lags** (deux grilles de lags),
- **multi‑starts** (plusieurs points initiaux),
- et sélection du meilleur optimum (NLL minimale).

---

## 5) Dataset : fenêtres longues + pas dense + winsorisation (Patches A/B)

Le dataset d’apprentissage est construit par glissement de fenêtre via :
- `win = 768` (fenêtres **longues**),
- `step = 16` (pas **dense**).

Pour chaque fenêtre, on construit :
- les features (gamma multi‑lags + log variance),
- la cible $$\alpha_1$$ via le proxy “local MLE robuste”.

Ensuite, le notebook applique une **winsorisation** sur la cible :
- filtration entre les quantiles 1% et 99% sur $$y$$.

> **Idées testées** :
- plus de données (pas dense) tout en gardant une estimation locale plus “fiable” (fenêtres longues),
- limiter l’impact des valeurs extrêmes de proxy (winsorisation), qui peuvent déstabiliser l’entraînement.

---

## 6) Split / standardisation / entraînement ANN (MSE + léger dropout)

### 6.1 Split + standardisation
- train/val split (80/20),
- standardisation des features : $$X \leftarrow (X-\mu_X)/\sigma_X$$.

### 6.2 Modèle
MLP simple (PyTorch) :
- 2 couches cachées (taille ~192),
- activations ReLU,
- **Dropout léger** (p=0.02),
- sortie `Sigmoid()` pour borner $$\hat\alpha_1\in(0,1)$$.

### 6.3 Optimisation
- AdamW,
- scheduler `ReduceLROnPlateau`,
- early stopping (dans l’esprit : arrêter quand la val n’améliore plus).

> **Idée testée** : une ANN “petite mais robuste” (régularisée) suffit à apprendre la correspondance  
features $$\rightarrow$$ $$\alpha_1$$ proxy.

---

## 7) Évaluation : MSE / R² + calibration finale de $$\alpha_1$$

Le notebook évalue sur la validation :
- MSE, RMSE, R²,
- scatter “vrai proxy vs prédit” (calibration plot),
- histogramme des résidus.

Puis il applique une **calibration** sur $$\alpha_1$$ prédit :
- **Isotonic Regression** (si dispo),
- sinon fallback **linéaire**.

> **Idée testée** : même si le MLP approxime bien la cible, une calibration monotone (isotone) peut corriger des biais systématiques (compression/étirement).

---

## 8) Reconstruction des paramètres GARCH finaux à partir de l’ANN

Une fois entraîné, le notebook calcule les paramètres “globaux” :

1. construire la feature globale sur toute la série (gamma multi‑lags + log variance),
2. prédire $$\alpha_1$$ puis calibrer : $$\hat\alpha_1$$,
3. estimer $$\hat\mu$$ via WLS alt,
4. reconstruire :
\[
\hat\beta_1 = \hat\mu - \hat\alpha_1,
\qquad
\hat\alpha_0 = \hat\sigma^2\,(1-\hat\mu)
\]
avec clipping pour respecter les contraintes (positivité et stationnarité).

---

## 9) MLE (Gauss & Student‑t) : baseline “classique”

Le notebook calcule ensuite :
- MLE gaussien ($$\alpha_0,\alpha_1,\beta_1$$),
- MLE Student‑t ($$\alpha_0,\alpha_1,\beta_1,\nu$$) initialisé par la solution gaussienne.

Objectif : obtenir une baseline **optimale en vraisemblance** sous ces hypothèses.

---

## 10) Benchmark ANN vs MLE : NLL/AIC + profil $$\nu$$ + heatmap $$(\nu,\mu)$$

### 10.1 Comparaison NLL & AIC
Le notebook évalue :
- NLL gaussienne pour les paramètres ANN,
- NLL Student‑t pour les paramètres ANN :
  - avec $$\nu$$ **fixé** au $$\nu$$ du MLE‑t,
  - avec $$\nu$$ **profilé** (minimisation sur une grille $$\nu\in[2.2,100]$$).

Puis il construit un tableau récapitulatif (export CSV + HTML) avec :
- paramètres,
- NLL,
- AIC (via `aic_from_nll(NLL, k)` où $$k$$ est le nombre de paramètres).

### 10.2 Profil de $$\nu$$
Courbe : $$\nu \mapsto \text{NLL}_t(\text{params ANN},\nu)$$  
→ extraction de $$\nu^*$$ minimisant la NLL.

### 10.3 Heatmap $$(\nu,\mu)$$
Le notebook calcule une carte :
- axe x : $$\mu = \alpha_1+\beta_1$$,
- axe y : $$\nu$$,
- couleur : NLL Student‑t

autour de la solution ANN, et compare visuellement avec le point MLE.

> **Idée testée** : diagnostiquer la sensibilité de la vraisemblance Student‑t aux degrés de liberté $$\nu$$ et à la persistance $$\mu$$, et vérifier si la solution ANN “tombe” dans une vallée de NLL comparable à la MLE.

---

## Lecture “conceptuelle” du notebook

En résumé, les idées principales testées sont :

1. **Feature engineering** via signature multi‑lags (acov/ACF) + niveau de variance.
2. **Décomposition du problème** :  
   - estimer $$\mu$$ par une méthode simple (WLS),  
   - apprendre $$\alpha_1$$ par ANN,  
   - déduire $$\beta_1$$ puis $$\alpha_0$$.
3. **Proxy robuste** de $$\alpha_1$$ par local‑MLE multi‑starts (Patch C) pour générer un dataset d’entraînement stable.
4. **Data augmentation temporelle** : fenêtres longues + pas dense (Patches A/B), puis winsorisation.
5. **Calibration finale** (isotone) pour corriger les biais du modèle.
6. **Validation statistique** : comparaison NLL/AIC vs MLE, exploration de $$\nu$$ et des interactions $$(\nu,\mu)$$.

---

### Fichiers produits par le notebook (exports)
- `garch_ann_student_t_eval_summary.csv`
- `garch_ann_student_t_eval_summary.html`

*(si les cellules d’export sont exécutées)*  



[`garch_ann_full_pipeline_v2.ipynb`](./garch_ann_full_pipeline_v2.ipynb)  
*End-to-end workflow for calibrating GARCH parameters using neural networks*

Key features:
- Data preprocessing for financial time series
- Neural network architecture design (LSTM/GRU)
- Model training and validation
- Real-time calibration on streaming data
- Performance benchmarking vs traditional methods

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
