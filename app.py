"""
Interactive Customer Churn Prediction Dashboard (Streamlit)

Run:
    streamlit run app.py
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/churn_model.joblib"
METRICS_PATH = "models/metrics.json"


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📉",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# LOAD MODEL METRICS
# --------------------------------------------------

def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)

    return None


model = load_model()

if model is None:
    st.error(
        "Model not found. Train the model first using: "
        "`python src/train.py`"
    )
    st.stop()


metrics = load_metrics()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📉 Customer Churn Analytics & Prediction Dashboard")

st.write(
    "Analyze customer information, predict churn risk, "
    "and generate business recommendations."
)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

if metrics:

    st.subheader("🤖 Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Best Model",
        metrics.get("best_model", "N/A")
    )

    c2.metric(
        "ROC-AUC",
        f"{metrics.get('roc_auc', 0):.3f}"
    )

    c3.metric(
        "Recall",
        f"{metrics.get('recall', 0):.3f}"
    )

    c4.metric(
        "F1 Score",
        f"{metrics.get('f1', 0):.3f}"
    )


st.divider()


# --------------------------------------------------
# SIDEBAR CUSTOMER INPUT
# --------------------------------------------------

st.sidebar.header("👤 Customer Details")


def sb_select(label, options, index=0):
    return st.sidebar.selectbox(
        label,
        options,
        index=index
    )


tenure = st.sidebar.slider(
    "Tenure (months)",
    0,
    72,
    5
)

monthly = st.sidebar.slider(
    "Monthly Charges ($)",
    18.0,
    120.0,
    85.0
)

total = st.sidebar.slider(
    "Total Charges ($)",
    0.0,
    9000.0,
    float(monthly * tenure)
)


customer = {

    "gender": sb_select(
        "Gender",
        ["Male", "Female"]
    ),

    "SeniorCitizen": (
        1
        if sb_select(
            "Senior citizen",
            ["No", "Yes"]
        ) == "Yes"
        else 0
    ),

    "Partner": sb_select(
        "Has partner",
        ["Yes", "No"],
        1
    ),

    "Dependents": sb_select(
        "Has dependents",
        ["Yes", "No"],
        1
    ),

    "tenure": tenure,

    "PhoneService": sb_select(
        "Phone service",
        ["Yes", "No"]
    ),

    "MultipleLines": sb_select(
        "Multiple lines",
        [
            "Yes",
            "No",
            "No phone service"
        ],
        1
    ),

    "InternetService": sb_select(
        "Internet service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ],
        1
    ),

    "OnlineSecurity": sb_select(
        "Online security",
        [
            "Yes",
            "No",
            "No internet service"
        ],
        1
    ),

    "OnlineBackup": sb_select(
        "Online backup",
        [
            "Yes",
            "No",
            "No internet service"
        ],
        1
    ),

    "DeviceProtection": sb_select(
        "Device protection",
        [
            "Yes",
            "No",
            "No internet service"
        ],
        1
    ),

    "TechSupport": sb_select(
        "Tech support",
        [
            "Yes",
            "No",
            "No internet service"
        ],
        1
    ),

    "StreamingTV": sb_select(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    ),

    "StreamingMovies": sb_select(
        "Streaming movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    ),

    "Contract": sb_select(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    ),

    "PaperlessBilling": sb_select(
        "Paperless billing",
        [
            "Yes",
            "No"
        ]
    ),

    "PaymentMethod": sb_select(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    ),

    "MonthlyCharges": monthly,

    "TotalCharges": total
}


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

predict_button = st.sidebar.button(
    "🔮 Predict Churn",
    type="primary"
)


if predict_button:

    df = pd.DataFrame([customer])

    probability = float(
        model.predict_proba(df)[:, 1][0]
    )


    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    st.subheader("🔮 Churn Prediction")

    st.progress(
        min(probability, 1.0)
    )

    st.metric(
        "Churn Probability",
        f"{probability:.1%}"
    )


    # --------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------

    if probability >= 0.70:

        risk_level = "🔴 High Risk"

        st.error(
            f"HIGH churn risk — "
            f"{probability:.1%} probability of leaving."
        )

    elif probability >= 0.40:

        risk_level = "🟠 Medium Risk"

        st.warning(
            f"MEDIUM churn risk — "
            f"{probability:.1%} probability of leaving."
        )

    else:

        risk_level = "🟢 Low Risk"

        st.success(
            f"LOW churn risk — "
            f"{probability:.1%} probability of leaving."
        )


    # --------------------------------------------------
    # RISK LEVEL DISPLAY
    # --------------------------------------------------

    st.subheader("📊 Customer Risk Level")

    st.write(
        f"### {risk_level}"
    )


    # --------------------------------------------------
    # BUSINESS RECOMMENDATION
    # --------------------------------------------------

    st.subheader("💡 Business Recommendation")


    if probability >= 0.70:

        recommendation = (
            "Prioritize this customer for retention. "
            "Consider a personalized discount, "
            "loyalty offer, or proactive customer support."
        )

    elif probability >= 0.40:

        recommendation = (
            "Monitor this customer and consider "
            "targeted engagement, loyalty benefits, "
            "or personalized communication."
        )

    else:

        recommendation = (
            "Continue regular engagement and "
            "loyalty activities."
        )


    st.info(recommendation)


    # --------------------------------------------------
    # CUSTOMER DETAILS
    # --------------------------------------------------

    with st.expander("👤 View Customer Information"):

        st.json(customer)


    # --------------------------------------------------
    # CUSTOMER SUMMARY
    # --------------------------------------------------

    st.subheader("📋 Customer Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Tenure",
        f"{tenure} months"
    )

    c2.metric(
        "Monthly Charges",
        f"${monthly:.2f}"
    )

    c3.metric(
        "Total Charges",
        f"${total:.2f}"
    )


else:

    st.info(
        "Set the customer details in the sidebar "
        "and click **Predict Churn**."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction | "
    "Machine Learning + Streamlit"
)
