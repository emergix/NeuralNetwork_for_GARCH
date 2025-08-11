# A hybrid GARCH, convolutional neural network and long short-term memory methods for volatility prediction in stock market  
**Dessie, E.; Birhane, T.; Seid, A. M.; Walelgn, A.** (2025). *Journal of Combinatorial Mathematics and Combinatorial Computing*, 124, 461–476. DOI: 10.61091/jcmcc124-30 :contentReference[oaicite:1]{index=1}

---

## 1. Objectif du papier

- Proposer un **modèle hybride** pour améliorer la précision des prévisions de **volatilité boursière**.
- Combine un modèle **GARCH(1,1)** classique avec un pipeline **CNN + LSTM** pour exploiter les avantages de chacun :
  - le **GARCH** pour capturer les clusters de volatilité et la kurtosis ;
  - le **CNN** pour extraire les motifs spatiaux dans les séries temporelles ;
  - le **LSTM** pour capturer les dépendances temporelles de longue durée. :contentReference[oaicite:2]{index=2}

## 2. Méthodologie

1. Estimation d’un **GARCH(1,1)** retenu selon AIC, log-vraisemblance maximale et significativité des paramètres.
2. Prévision de la volatilité (ou résidus standardisés ?), utilisée comme input pour les réseaux.
3. Conception d’un réseau **CNN + LSTM** alimenté par les données issues du GARCH.
4. Évaluation sur données hors échantillon, via **MSE**, **RMSE**, **MAE**.
5. Comparaison aux modèles GARCH seul, CNN seul, LSTM seul, voire CNN-LSTM sans GARCH.
6. Test statistique de Diebold-Mariano pour valider la supériorité des performances. :contentReference[oaicite:3]{index=3}

## 3. Résultats clés

- Le modèle **hybride (GARCH + CNN + LSTM)** surpasse les autres modèles, avec une **amélioration de précision** variant entre **8 % et 13 %** selon les métriques considérées.
- Le test **Diebold-Mariano** confirme que la différence de performance est **statistiquement significative**. :contentReference[oaicite:4]{index=4}

## 4. Apports et implications

- Le **pré-traitement GARCH** stabilise les séries financières, réduisant l'hétéroscédasticité.
- Le **CNN** extrait automatiquement des motifs structurés (corrélations, patterns sectoriels).
- Le **LSTM** assure une bonne modélisation de la mémoire temporelle (autocorrélations, cycles).
- La **combinaison synergique** de ces composants améliore notablement la qualité des prévisions de volatilité.

## 5. Limites et pistes de recherche

- Le papier ne précise pas l’architecture exacte (nombre de couches CNN ou LSTM, tailles de filtres, hyperparamètres).
- Les données exploitées (actifs, fréquences temporelles, période historique) ne sont pas détaillées dans l’aperçu.
- Il serait utile d’ajouter des évaluations économiques (test MCS, ablation, robustesse dans des crises, etc.).

---

# Workflow modèle GARCH + CNN + LSTM pour un desk dérivés

## Diagramme principal

```mermaid
flowchart LR
    subgraph Data[Sources de données]
      A1[Marchés temps réel\n(Ticks/1m/5m/Daily)]
      A2[Vol implicite & surfaces\n(Options listées)]
      A3[Facteurs/Macro/News]
    end

    subgraph Prep[Pré-traitement & GARCH]
      B1[Nettoyage/Alignement\n+ z-score par actif]
      B2[GARCH/EGARCH/GJR par actif\nσ_t, résidus standardisés ε_t]
      B3[Fenêtrage L×N\n(construction tenseurs)]
    end

    subgraph NN[Feature learning & Forecast]
      C1[Bloc CNN (spatial)\nco-mouvements cross-section]
      C2[Bloc LSTM (temporel)\nmémoire & régimes]
      C3[Tête de sortie\nμ_vol(t+h), quantiles, incertitude]
    end

    subgraph Orchestration[Orchestration & Monitoring]
      D1[Backtest walk-forward\nPnL, Sharpe, turnover]
      D2[Évaluation en continu\nDrift & recalibrage]
      D3[Explainability\nFiltres CNN, importance temporelle]
    end

    subgraph Consumers[Consommateurs côté desk]
      E1[Pricing surfaces\n(SABR/SVI/Bergomi)]
      E2[Trading vol\n(variance/vol swaps, VIX, options)]
      E3[Hedging Vega/Gamma]
      E4[Risk (VaR/ES, Stress tests)]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2 --> B3 --> C1 --> C2 --> C3
    C3 --> D1
    C3 --> D2
    C3 --> D3
    C3 --> E1
    C3 --> E2
    C3 --> E3
    C3 --> E4
```
