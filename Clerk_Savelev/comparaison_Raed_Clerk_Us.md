# Comparatif — De Clerk & Savel’ev vs Papier étudiant vs Notre pipeline

| Axe | De Clerk & Savel’ev (2022) | Papier Raed | Notre travail ici |
|---|---|---|---|
| **But principal** | Prédire **α₁** via ANN, puis **déduire β₁ et α₀** par formules fermées (à partir de Γ₄, σ²). | Tente d’**apprendre (α₀, α₁, β₁)** simultanément avec un NN; abandon partiel des formules fermées jugées instables. | Apprend **α₁** avec ANN (**acov_multi**), **estime μ=α₁+β₁** par régression (WLS) sur log γ̂ₙ, puis **β₁=μ−α₁**, **α₀=σ²(1−μ)**. |
| **Entrées (features)** | Moments {E[x²], Γ₄, Γ₆} et/ou **autocovariances normalisées γ̂ₙ** (n clés: 2, 6, 10). | Moments + **γ̂ₙ multiples** + essais **L-moments**; résultats instables/bruités. | **acov_multi** (lags consécutifs, p.ex. 3..12/16), pas d’L-moments. |
| **Formules clés** | Équations analytiques: E[x²], Γ₄, Γ₆, γ̂ₙ (Bollerslev). | Reprend les équations, annonce une “correction” de γ̂ₙ sans preuve solide. | Même set; **μ** extrait directement de **log γ̂ₙ** (régression linéaire/pondérée). |
| **Architecture & entraînement** | MLP ReLU compact, données scalées, Adam + **early-stopping**. | MLP **très large** (et sinusoidal), **sigmoïdes** en sortie; LR adaptatif “maison”. | MLP **frugal** (acov_multi) + early-stopping; **scheduler** standard possible; normalisation des features. |
| **Contraintes / faisabilité** | Conditions d’existence des moments discutées; prudence sur la zone (α₁,β₁). | Pénalisation **α₁+β₁<1** trop faible; sigmoïde borne tous les params **[0,1]** (peu naturel pour α₀). | **μ** estimé ⇒ **α₁+β₁<1** garanti; **α₀>0** via formule/clip; pas de bornage arbitraire d’α₀. |
| **Évaluation / résultats** | OOS bruité, **n=6** souvent meilleur pour γ̂ₙ; intérêt d’early-stopping. | Nombreuses figures “Y≈X” mais peu de **métriques** (RMSE/NLL/AIC) claires. | **Benchmarks NLL/AIC** sur **série réelle** (Gauss vs **Student-t**), profils **ν** et **carte (ν, μ)**; exports CSV/HTML. |
| **Choix distribution** | Gauss conditionnelle (côté ANN). | Majoritairement Gauss; MLE normal dérivé. | **Gauss vs Student-t** (ν fixé & **profilé**), choix appuyé par AIC. |
| **Points faibles** | Perte bruitée; prédire tous les params crée des “edges” ⇒ préfère **α₁ seul**. | Formule γ̂ₙ incertaine; réseaux surdimensionnés; sorties sigmoïdes; LR exotique. | Biais récurrent: **μ sous-estimé** si régression non pondérée ⇒ corrigé par **WLS**/lags étendus. |
| **Plus-value / originalité** | Simplicité justifiée: **α₁ seul**, lags γ̂ₙ ciblés (**n=6**). | Ambition end-to-end; exploration L-moments et “decision-maker” de LR. | Pipeline **pragmatique**: ANN rapide → **refine local** (optionnel) + diagnostics (profils **ν**, heatmap **ν-μ**). |

---

## Commentaires clés

- **Philosophie** : De Clerk & Savel’ev privilégient un **design minimal** (ANN pour α₁ + formules), le papier étudiant tente le **tout-en-un**, notre approche **hybride** garde l’ANN pour α₁ mais **verrouille μ** via γ̂ₙ (stable) — c’est ce qui sécurise la **stationnarité** et évite de “forcer” α₀ dans [0,1].
- **Pourquoi Raed plafonne** : (i) bornage sigmoïde des **trois** paramètres, (ii) pénalisation **α₁+β₁<1** trop molle, (iii) architecture trop **large** → optimisation instable, (iv) γ̂ₙ mal géré aux **grands lags** ⇒ **μ sous-estimé**.
- **Ce qui marche ici** : (i) **acov_multi** + **WLS** sur log γ̂ₙ, (ii) **benchmarks** MLE Gauss/**Student-t** avec **constantes incluses** (AIC comparables), (iii) **profils ν** + **heatmap (ν, μ)** pour diagnostiquer où gagner (hausse de μ ~0.97–0.98 a amélioré la NLL).

### Recommandations rapides pour le papier Raed
1. Remplacer la pénalisation simple par une **barrière douce** couvrant **toutes** les contraintes de moments (Γ₄/Γ₆).
2. Retirer la sigmoïde sur **α₀** (garder **softplus**), normaliser les features, **réduire** l’architecture.
3. Estimer **μ** par **régression pondérée** sur **log γ̂ₙ** (lags 3..12/16), puis **β₁=μ−α₁**.
4. Évaluer sur série réelle avec **NLL/AIC** cohérents (Gauss **et** t), et, si besoin, un **affinage local** (Nelder–Mead) depuis l’ANN.

**En bref** : De Clerk & Savel’ev posent une base **simple et robuste**; le papier Raed s’éparpille et perd en stabilité; notre pipeline **hybride** garde la vitesse de l’ANN tout en **contrôlant μ** et en **mesurant** correctement la qualité (NLL/AIC, ν-profil).
