"""
predict.py — Use the trained model to predict insurance cost for a new patient
Usage: python predict.py
"""

import joblib
import numpy as np
import pandas as pd

model = joblib.load("models/best_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")


def predict_cost(age, sex, bmi, children, smoker, region):
    """
    Predict insurance charges for a patient.

    Parameters
    ----------
    age      : int   (18–64)
    sex      : str   ('male' | 'female')
    bmi      : float (e.g. 27.5)
    children : int   (0–5)
    smoker   : str   ('yes' | 'no')
    region   : str   ('northeast' | 'northwest' | 'southeast' | 'southwest')

    Returns
    -------
    float : predicted insurance cost in USD
    """
    row = {
        "age":      age,
        "sex":      1 if sex == "male" else 0,
        "bmi":      bmi,
        "children": children,
        "smoker":   1 if smoker == "yes" else 0,
        # One-hot encoded region (drop_first=True → northeast is baseline)
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }

    df = pd.DataFrame([row])[feature_names]
    prediction = model.predict(df)[0]
    return round(prediction, 2)


if __name__ == "__main__":
    print("\n── Single Patient Prediction ──────────────────────────")

    examples = [
        dict(age=28, sex="female", bmi=26.2, children=0, smoker="no",  region="northwest"),
        dict(age=45, sex="male",   bmi=34.1, children=2, smoker="yes", region="southeast"),
        dict(age=60, sex="female", bmi=29.5, children=3, smoker="no",  region="northeast"),
    ]

    for i, ex in enumerate(examples, 1):
        cost = predict_cost(**ex)
        print(f"\n  Patient {i}:")
        for k, v in ex.items():
            print(f"    {k:<12}: {v}")
        print(f"  → Predicted Cost: ${cost:,.2f}")

    print("\n── Sensitivity Analysis: Smoking Impact ───────────────")
    base = dict(age=35, sex="male", bmi=28.0, children=1, region="northeast")
    for smoker in ["no", "yes"]:
        cost = predict_cost(**base, smoker=smoker)
        print(f"  Smoker={smoker:<4} → ${cost:,.2f}")

    print("\n── Sensitivity Analysis: BMI Gradient ─────────────────")
    for bmi in [20, 25, 30, 35, 40]:
        cost = predict_cost(age=40, sex="male", bmi=bmi, children=0,
                            smoker="no", region="northeast")
        print(f"  BMI={bmi} → ${cost:,.2f}")
