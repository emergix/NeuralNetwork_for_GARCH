# Calibration GARCH(1,1) via Réseau de Neurones — Fiche technique

> Fiche synthétique pour calibrer un **GARCH-normal(1,1)** avec un **réseau de neurones (ANN)** en remplaçant/accélérant le MLE classique : le réseau prédit **α₁** à partir de caractéristiques empiriques robustes, puis on **reconstruit β₁ et α₀** par formules fermées.

---

## 1) Objectif

- **But** : calibration rapide et précise d'un GARCH(1,1) sur données financières, adaptée aux contextes temps-réel / haute fréquence.
- **Idée** : apprendre la relation *caractéristiques de série* → *paramètre* (ici **α₁**) sur données **synthétiques** (générées par les formules analytiques du modèle), puis **déduire** les autres paramètres analytiquement.

---

## 2) Modèle

$$\sigma_t^2=\alpha_0+\alpha_1 x_{t-1}^2+\beta_1 \sigma_{t-1}^2,\qquad x_t=\sigma_t Z_t,\; Z_t\sim\mathcal N(0,1)$$

Conditions usuelles : $\alpha_1 \ge 0,\ \beta_1 \ge 0,\ \alpha_1+\beta_1 < 1$.

**Moments & autocovariances (GARCH(1,1), innovations gaussiennes)**  
- Variance inconditionnelle :
$$\sigma^2=\frac{\alpha_0}{1-\alpha_1-\beta_1}$$
- Kurtosis :
$$\Gamma_4 = 3 + \frac{6\alpha_1^2}{1-3\alpha_1^2-2\alpha_1\beta_1-\beta_1^2}$$
- Autocovariance normalisée des carrés (lag $n$) :
$$\hat\gamma_n = \frac{2\alpha_1(1-\alpha_1\beta_1-\beta_1^2)}{1-3\alpha_1^2-2\alpha_1\beta_1-\beta_1^2}(\alpha_1+\beta_1)^{n-1}$$

> Remarque : des contraintes supplémentaires (positivité des dénominateurs ci-dessus) délimitent l'espace $(\alpha_1,\beta_1)$ valide.

---

## 3) Caractéristiques (entrées du réseau)

Deux variantes pratiques :

### Variante A — **Moments**
Entrées : $[\sigma^2,\ \Gamma_4,\ \Gamma_6]$

- $\Gamma_6$ : moment standardisé d'ordre 6 (expression fermée plus longue ; utile pour la génération synthétique et l'entraînement).
- Avantage : facile à estimer proprement sur séries longues.
- Attention : des moments d'ordre trop élevé ($\Gamma_8,\Gamma_{10}$) **resserrent** l'espace admissible et **dégradent** souvent la précision.

### Variante B — **Autocovariance normalisée** (souvent la plus performante)
Entrées : $[\sigma^2,\ \Gamma_4,\ \hat\gamma_n]$  
Choix recommandé : **$n=6$** (bon compromis biais/variance observé).

---

## 4) Sortie du réseau & reconstruction des paramètres

Le réseau **prédit uniquement $\alpha_1$**. On reconstruit ensuite :

$$\beta_1 = \sqrt{1 - 2\alpha_1^2 - \frac{6\alpha_1^2}{\Gamma_{4,\text{emp}} - 3}} - \alpha_1$$
*(clip interne de l'argument de la racine à $[0,1]$ pour robustesse)*

$$\alpha_0 = \sigma^2_{\text{emp}}(1-\alpha_1-\beta_1)$$

- $\Gamma_{4,\text{emp}}$ et $\sigma^2_{\text{emp}}$ sont estimés **sur la série**.
- Si $\Gamma_{4,\text{emp}}\le 3$ (cas extrême / estimation bruitée), utiliser un **fallback** prudent (clip ou borne inférieure sur $\beta_1$).

---

## 5) Réseau & entraînement

- **Architecture** : MLP à 4 couches cachées, tailles **(128, 2048, 2048, 128)**, activations **ReLU**, sortie scalaire (α₁).
- **Perte** : **MSE**, **optimiseur Adam**, **early stopping** (sur val_loss).
- **Données d'entraînement** : **synthétiques** en tirant $(\alpha_0,\alpha_1,\beta_1)$ dans l'espace valide, puis en calculant les caractéristiques analytiquement.
- **Normalisation** : les features sont naturellement à échelles comparables (kurtosis, autocovariance normalisée). Optionnel : standardiser/MinMax scaler pour stabiliser l'optimisation.

---

## 6) Flux de calibration (recette)

1. **Construire/charger** un modèle ANN **entraîné** sur données synthétiques, pour la variante choisie (Moments ou $\hat\gamma_n$).
2. **Extraire** sur la série empirique :
   - $\sigma^2_{\text{emp}} = \mathbb{E}[x^2]$
   - $\Gamma_{4,\text{emp}} = \frac{\mathbb{E}[x^4]}{\mathbb{E}[x^2]^2}$
   - Soit $\Gamma_{6,\text{emp}}$ (Variante A), soit $\hat\gamma_{n,\text{emp}}$ (Variante B, ex. $n=6$).
3. **Prédire** $\alpha_1$ avec le réseau : $\alpha_1 = f_\theta(\text{features})$.
4. **Reconstruire** $\beta_1$ via la formule fermée avec $\Gamma_{4,\text{emp}}$.
5. **Reconstruire** $\alpha_0$ via $\sigma^2_{\text{emp}}$.
6. **(Optionnel)** Vérifier les contraintes ($\alpha_1\ge0,\ \beta_1\ge0,\ \alpha_1+\beta_1<1$). Clip ajusté si besoin.

---

## 7) Conseils pratiques & limitations

- **Early stopping** indispensable : les courbes train/val peuvent être **bruitées**.
- **Moments élevés** : $\Gamma_6$ ok ; $\Gamma_8,\Gamma_{10}$ souvent **moins stables**.
- **Autocovariance** : $n=6$ fonctionne très bien en pratique (meilleur out-of-sample observé dans l'article).
- **Bruit d'estimation** : sur séries courtes, l'estimation de $\Gamma_4$ et $\hat\gamma_n$ peut fluctuer ⇒ lisser (fenêtres roulantes) ou allonger l'horizon si possible.
- **Interprétabilité** : on ne remplace pas l'analyse statistique ; on **complète**/accélère la calibration.
- **Généralisation** : l'entraînement synthétique doit **couvrir** la zone plausible des paramètres pour la classe d'actifs considérée.

---

## 8) Exemple d'utilisation (avec le module fourni)

```python
# pip install torch numpy
from garch_ann_calibration import Calibrator, TrainConfig
import numpy as np

# 1) Entraînement (variante autocovariance, lag=6)
cal = Calibrator(variant="acov", lag=6)
result = cal.fit(
    n_samples=150_000,
    cfg=TrainConfig(epochs=5000, lr=1e-2, batch_size=1024, patience=50)
)
print("Best val loss:", result.best_val_loss)

# 2) Calibration sur une série empirique
returns = np.random.standard_t(df=6, size=20000) * 0.01  # exemple jouet
params = cal.calibrate_from_empirical(returns)
print(params)  # CalibratedParams(alpha0=..., alpha1=..., beta1=...)
```

---

## 8) Module Fourni

 → Fichier python : [garch_ann_calibration.py](./garch_ann_calibration.py)

## 8) Etude Complete de la calibration de GARCH(1,1) avec  reseau neuronal

 → notebook python : [garch_ann_full_study.ipynb](./garch_ann_full_study.ipynb)

