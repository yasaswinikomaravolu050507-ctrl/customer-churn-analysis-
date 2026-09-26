"""
Score new customers with the trained churn model.

Usage:
    # Predict on a CSV of customers (same columns as training, minus 'Churn')
    python src/predict.py --input data/new_customers.csv --output data/predictions.csv

    # If no --input is given, a single built-in example customer is scored.
"""

import argparse
import os

import joblib
import pandas as pd

MODEL_PATH = "models/churn_model.joblib"

EXAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5,
    "TotalCharges": 179.0,
}


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. Train it first with:  python src/train.py"
        )
    return joblib.load(path)


def predict(df: pd.DataFrame, model=None) -> pd.DataFrame:
    model = model or load_model()
    if "customerID" in df.columns:
        ids = df["customerID"]
        df = df.drop(columns=["customerID"])
    else:
        ids = pd.Series([f"row_{i}" for i in range(len(df))], name="customerID")
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    proba = model.predict_proba(df)[:, 1]
    result = df.copy()
    result.insert(0, "customerID", ids.values)
    result["churn_probability"] = proba.round(4)
    result["churn_prediction"] = (proba >= 0.5).astype(int)
    result["risk_band"] = pd.cut(
        proba, bins=[-0.01, 0.3, 0.6, 1.01], labels=["Low", "Medium", "High"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict customer churn.")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model = load_model()

    if args.input:
        df = pd.read_csv(args.input)
    else:
        df = pd.DataFrame([EXAMPLE_CUSTOMER])
        print("No --input provided; scoring the built-in example customer.\n")

    result = predict(df, model)
    cols = ["customerID", "churn_probability", "churn_prediction", "risk_band"]
    print(result[cols].to_string(index=False))

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nSaved predictions -> {args.output}")


if __name__ == "__main__":
    main()
