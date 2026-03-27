"""
Prédiction du prix d'un appartement avec le MLP entraîné.
Saisie manuelle des paramètres → prix estimé.

Usage : python predict.py
"""

import numpy as np
import joblib

# Charger le modèle et le scaler
model = joblib.load('./models/mlp_model.joblib')
scaler = joblib.load('./models/scaler.joblib')
feature_cols = joblib.load('./models/feature_cols.joblib')
model_features = joblib.load('./models/model_features.joblib')

OCEAN_CATEGORIES = {
    '0': ('<1H OCEAN', 0),
    '1': ('INLAND', 1),
    '2': ('ISLAND', 2),
    '3': ('NEAR BAY', 3),
    '4': ('NEAR OCEAN', 4),
}


def get_input():
    print("=" * 55)
    print("  PREDICTION DU PRIX IMMOBILIER (MLP Perceptron)")
    print("=" * 55)
    print()

    raw = {}

    raw['longitude'] = float(input("Longitude (ex: -122.23) : "))
    raw['latitude'] = float(input("Latitude (ex: 37.88) : "))
    raw['housing_median_age'] = float(input("Age médian du logement (ex: 41) : "))
    raw['total_rooms'] = float(input("Nombre total de pièces du bloc (ex: 880) : "))
    raw['total_bedrooms'] = float(input("Nombre total de chambres du bloc (ex: 129) : "))
    raw['population'] = float(input("Population du bloc (ex: 322) : "))
    raw['households'] = float(input("Nombre de ménages du bloc (ex: 126) : "))
    raw['median_income'] = float(input("Revenu médian (en dizaines de k$, ex: 8.3) : "))

    print()
    print("Proximité océan :")
    for k, (name, _) in OCEAN_CATEGORIES.items():
        print(f"  {k} = {name}")
    choice = input("Choix (0-4) : ")
    raw['ocean_proximity'] = OCEAN_CATEGORIES[choice][1]

    return raw


def predict(raw):
    # Construire le vecteur dans l'ordre du scaler (feature_cols)
    values = np.array([[raw[col] for col in feature_cols]])

    # Normaliser avec le scaler (même que l'entraînement)
    values_scaled = scaler.transform(values)

    # Sélectionner les features utilisées par le modèle
    feature_indices = [feature_cols.index(f) for f in model_features]
    values_model = values_scaled[:, feature_indices]

    # Prédire
    price = model.predict(values_model)[0]
    return price


def main():
    try:
        raw = get_input()
        price = predict(raw)

        print()
        print("-" * 55)
        print(f"  Prix estimé : {price:,.0f} $")
        print("-" * 55)
        print()
        print("  (MAPE ~28% → marge d'erreur estimée :")
        lower = price * 0.72
        upper = price * 1.28
        print(f"   entre {lower:,.0f} $ et {upper:,.0f} $)")
        print()

    except KeyboardInterrupt:
        print("\nAnnulé.")
    except Exception as e:
        print(f"\nErreur : {e}")


if __name__ == '__main__':
    main()
