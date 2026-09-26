"""
Train and evaluate customer churn models.

Pipeline:
    1. Load data (auto-generates synthetic data if the CSV is missing).
    2. Clean + split into train/test.
    3. Preprocess (impute, scale numerics, one-hot encode categoricals).
    4. Train 3 models: Logistic Regression, Random Forest, XGBoost.
    5. Compare with cross-validated ROC-AUC, evaluate the best on the test set.
    6. Save the best pipeline, a metrics report, and diagnostic plots.

Usage:
    python src/train.py
    python src/train.py --data data/churn.csv --out models
"""

import argparse
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

TARGET = "Churn"
DROP_COLS = ["customerID"]


def load_data(path: str) -> pd.DataFrame:
    """Load the dataset, generating synthetic data if it doesn't exist yet."""
    if not os.path.exists(path):
        print(f"'{path}' not found -> generating synthetic data...")
        from generate_data import generate

        os.makedirs(os.path.dirname(path), exist_ok=True)
        generate().to_csv(path, index=False)

    df = pd.read_csv(path)

    # The real Telco dataset stores TotalCharges as text with blanks -> fix it.
    if df["TotalCharges"].dtype == object:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )


def get_models(scale_pos_weight: float) -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Train churn models.")
    parser.add_argument("--data", type=str, default="data/churn.csv")
    parser.add_argument("--out", type=str, default="models")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    reports_dir = os.path.join(args.out, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # --- Load + split -------------------------------------------------------
    df = load_data(args.data)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Dataset: {X.shape[0]:,} rows, {X.shape[1]} features | churn rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(X)
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = get_models(scale_pos_weight)

    # --- Cross-validate each model -----------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    pipelines = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = (scores.mean(), scores.std())
        pipelines[name] = pipe
        print(f"  {name:20s} CV ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_results, key=lambda k: cv_results[k][0])
    print(f"\nBest model: {best_name}")

    # --- Fit best on full training set, evaluate on held-out test ----------
    best_pipe = pipelines[best_name]
    best_pipe.fit(X_train, y_train)

    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "best_model": best_name,
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "cv_roc_auc": {k: round(v[0], 4) for k, v in cv_results.items()},
    }
    print("\nTest-set performance:")
    print(json.dumps(metrics, indent=2))
    print("\n" + classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

    # --- Save model + metrics ----------------------------------------------
    model_path = os.path.join(args.out, "churn_model.joblib")
    joblib.dump(best_pipe, model_path)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model -> {model_path}")

    # --- Diagnostic plots ---------------------------------------------------
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Stay", "Churn"], cmap="Blues"
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"), dpi=120)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"ROC Curve - {best_name} (AUC={metrics['roc_auc']})")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "roc_curve.png"), dpi=120)
    plt.close()

    plot_feature_importance(best_pipe, reports_dir, best_name)
    print(f"Saved plots -> {reports_dir}")


def plot_feature_importance(pipe: Pipeline, reports_dir: str, name: str, top_n: int = 15):
    """Plot top feature importances / coefficients for the trained model."""
    try:
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
    except Exception:
        return

    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importance = np.abs(clf.coef_[0])
    else:
        return

    order = np.argsort(importance)[::-1][:top_n]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(order)), importance[order][::-1], color="#2563eb")
    plt.yticks(range(len(order)), [feature_names[i] for i in order][::-1], fontsize=8)
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Features - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "feature_importance.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
