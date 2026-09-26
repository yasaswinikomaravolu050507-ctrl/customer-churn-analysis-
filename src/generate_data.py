"""
Generate a realistic, synthetic Telco-style customer churn dataset.

This lets the whole project run end-to-end without downloading anything.
The columns and churn behaviour mimic the well-known "Telco Customer Churn"
dataset so the pipeline works the same on the real data if you drop it in.

Usage:
    python src/generate_data.py --rows 7000 --out data/churn.csv
"""

import argparse
import os

import numpy as np
import pandas as pd


def generate(rows: int = 7000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Categorical feature pools -----------------------------------------
    genders = rng.choice(["Male", "Female"], rows)
    senior = rng.choice([0, 1], rows, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], rows, p=[0.30, 0.70])

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], rows, p=[0.55, 0.21, 0.24]
    )
    paperless = rng.choice(["Yes", "No"], rows, p=[0.59, 0.41])
    payment = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        rows,
        p=[0.34, 0.23, 0.22, 0.21],
    )
    internet = rng.choice(
        ["DSL", "Fiber optic", "No"], rows, p=[0.34, 0.44, 0.22]
    )
    phone = rng.choice(["Yes", "No"], rows, p=[0.90, 0.10])

    no_phone = phone == "No"
    multiple_lines = rng.choice(["Yes", "No"], rows, p=[0.42, 0.58]).astype(object)
    multiple_lines[no_phone] = "No phone service"

    def dependent_service(base_choices, no_internet_mask):
        """Services that only exist when the customer has internet."""
        out = rng.choice(base_choices, rows, p=[0.5, 0.5])
        out = out.astype(object)
        out[no_internet_mask] = "No internet service"
        return out

    no_net = internet == "No"
    online_security = dependent_service(["Yes", "No"], no_net)
    online_backup = dependent_service(["Yes", "No"], no_net)
    device_protection = dependent_service(["Yes", "No"], no_net)
    tech_support = dependent_service(["Yes", "No"], no_net)
    streaming_tv = dependent_service(["Yes", "No"], no_net)
    streaming_movies = dependent_service(["Yes", "No"], no_net)

    # --- Numeric features ---------------------------------------------------
    tenure = rng.integers(0, 73, rows)

    base_charge = np.where(internet == "Fiber optic", 70, np.where(internet == "DSL", 45, 20))
    monthly = base_charge + rng.normal(0, 8, rows)
    monthly += (streaming_tv == "Yes") * 6
    monthly += (streaming_movies == "Yes") * 6
    monthly = np.clip(monthly, 18, 120).round(2)

    total = (monthly * tenure + rng.normal(0, 25, rows)).clip(0).round(2)

    # --- Churn probability model (the "ground truth" signal) ---------------
    logit = (
        -1.6
        + (contract == "Month-to-month") * 1.5
        + (contract == "Two year") * -1.3
        + (internet == "Fiber optic") * 0.9
        + (payment == "Electronic check") * 0.7
        + (senior == 1) * 0.4
        + (partner == "No") * 0.3
        + (tech_support == "No") * 0.4
        + (online_security == "No") * 0.3
        + (paperless == "Yes") * 0.25
        - (tenure / 72.0) * 2.2
        + rng.normal(0, 0.5, rows)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(rows) < churn_prob).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": [f"C{100000 + i}" for i in range(rows)],
            "gender": genders,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": churn_label,
        }
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic churn data.")
    parser.add_argument("--rows", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/churn.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)

    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"Wrote {len(df):,} rows to {args.out}")
    print(f"Churn rate: {churn_rate:.1%}")


if __name__ == "__main__":
    main()
