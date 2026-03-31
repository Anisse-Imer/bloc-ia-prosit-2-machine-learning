# Prosit 2 — Machine Learning : Prédiction immobilière en Californie

Projet CESI Bloc IA — comparaison de modèles de Machine Learning pour prédire le prix médian des logements en Californie à partir du dataset [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).

## Modèles comparés

| Modèle | Notebook | Résultat |
|---|---|---|
| Régression linéaire | `predictive_linear_regression.ipynb` | Baseline (OLS via statsmodels) |
| Random Forest | `predictive_random_forest.ipynb` | Visualisation dendrogramme & feature importance |
| MLP (Perceptron multicouche) | `predictive_perceptron.ipynb` | **Meilleur modèle** — MAPE ~28% |

## Structure du projet

```
datasets/               # Données brutes et transformées
models/                 # Modèle MLP exporté (.joblib) + scaler
data_transform.ipynb    # Pipeline de transformation (KNN imputation, outliers, Z-Score)
WS_Regression.ipynb     # Workshop de référence sur la régression
fusion_notebook.ipynb   # Notebook de synthèse avec résultats
predict.py              # Script interactif de prédiction
```

## Installation

```bash
pip install -r requirements.txt
```

## Prédiction

Le modèle MLP entraîné peut être utilisé directement via le script interactif :

```bash
python predict.py
```

Il demande les caractéristiques du logement (longitude, latitude, âge, nombre de pièces, revenu médian, proximité océan…) et renvoie un prix estimé avec marge d'erreur.

## Rapport

Le rapport complet est disponible dans `anisse_imer_prosit_2.pdf`.
