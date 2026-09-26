"""
Interactive churn prediction demo (Streamlit).

Run:
    streamlit run app.py

Adjust a customer's attributes in the sidebar and get a live churn-risk score.
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/churn_model.joblib"
METRICS_PATH = "models/metrics.json"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


st.title("📉 Customer Churn Predictor")
st.caption("Classic ML project — predicts whether a telecom customer will leave.")

model = load_model()
if model is None:
    st.error("Model not found. Train it first:  `python src/train.py`")
    st.stop()

metrics = load_metrics()
if metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", metrics["best_model"])
    c2.metric("ROC-AUC", metrics["roc_auc"])
    c3.metric("Recall", metrics["recall"])
    c4.metric("F1", metrics["f1"])

st.sidebar.header("Customer details")


def sb_select(label, options, index=0):
    return st.sidebar.selectbox(label, options, index=index)


tenure = st.sidebar.slider("Tenure (months)", 0, 72, 5)
monthly = st.sidebar.slider("Monthly charges ($)", 18.0, 120.0, 85.0)
total = st.sidebar.slider("Total charges ($)", 0.0, 9000.0, float(monthly * tenure))

customer = {
    "gender": sb_select("Gender", ["Male", "Female"]),
    "SeniorCitizen": 1 if sb_select("Senior citizen", ["No", "Yes"]) == "Yes" else 0,
    "Partner": sb_select("Has partner", ["Yes", "No"], 1),
    "Dependents": sb_select("Has dependents", ["Yes", "No"], 1),
    "tenure": tenure,
    "PhoneService": sb_select("Phone service", ["Yes", "No"]),
    "MultipleLines": sb_select("Multiple lines", ["Yes", "No", "No phone service"], 1),
    "InternetService": sb_select("Internet service", ["DSL", "Fiber optic", "No"], 1),
    "OnlineSecurity": sb_select("Online security", ["Yes", "No", "No internet service"], 1),
    "OnlineBackup": sb_select("Online backup", ["Yes", "No", "No internet service"], 1),
    "DeviceProtection": sb_select("Device protection", ["Yes", "No", "No internet service"], 1),
    "TechSupport": sb_select("Tech support", ["Yes", "No", "No internet service"], 1),
    "StreamingTV": sb_select("Streaming TV", ["Yes", "No", "No internet service"]),
    "StreamingMovies": sb_select("Streaming movies", ["Yes", "No", "No internet service"]),
    "Contract": sb_select("Contract", ["Month-to-month", "One year", "Two year"]),
    "PaperlessBilling": sb_select("Paperless billing", ["Yes", "No"]),
    "PaymentMethod": sb_select(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    ),
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}

if st.sidebar.button("Predict churn", type="primary"):
    df = pd.DataFrame([customer])
    proba = float(model.predict_proba(df)[:, 1][0])

    st.subheader("Result")
    st.progress(min(proba, 1.0))
    if proba >= 0.6:
        st.error(f"HIGH churn risk — {proba:.1%} probability of leaving.")
    elif proba >= 0.3:
        st.warning(f"MEDIUM churn risk — {proba:.1%} probability of leaving.")
    else:
        st.success(f"LOW churn risk — {proba:.1%} probability of leaving.")

    with st.expander("See customer input"):
        st.json(customer)
else:
    st.info("Set the customer details in the sidebar, then click **Predict churn**.")
