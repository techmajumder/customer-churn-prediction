"""
Customer Churn Prediction — Streamlit App
Free to deploy forever on Streamlit Community Cloud.
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("outputs/churn_model.joblib")
    encoders = joblib.load("outputs/encoders.joblib")
    with open("outputs/feature_names.json") as f:
        feature_names = json.load(f)
    return model, encoders, feature_names

model, encoders, FEATURE_NAMES = load_model()

st.title("📉 Customer Churn Prediction")
st.caption("Random Forest trained on the IBM Telco Customer Churn dataset (7,043 customers, 84.3% ROC-AUC)")

tab1, tab2 = st.tabs(["📋 Batch: Score My Whole Customer List", "🧍 Single Customer: Manual Entry"])

# ── TAB 1: Batch prediction ─────────────────────────────────────
with tab1:
    st.write("Upload a CSV of customers (same columns as the Telco dataset). Get every customer's churn risk, ranked highest to lowest.")
    uploaded_file = st.file_uploader("Upload customer CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        id_col = None
        for candidate in ["customerID", "CustomerID", "customer_id", "ID", "Name"]:
            if candidate in df.columns:
                id_col = candidate
                break

        work = df.copy()
        drop_cols = [c for c in [id_col, "Churn"] if c and c in work.columns]
        work = work.drop(columns=drop_cols)

        if "TotalCharges" in work.columns:
            work["TotalCharges"] = pd.to_numeric(work["TotalCharges"], errors="coerce")
            work["TotalCharges"] = work["TotalCharges"].fillna(work["MonthlyCharges"])

        for col in FEATURE_NAMES:
            if col in encoders:
                le = encoders[col]
                work[col] = work[col].apply(lambda v: le.transform([v])[0] if v in le.classes_ else 0)

        X = work[FEATURE_NAMES]
        proba = model.predict_proba(X)[:, 1]

        result = pd.DataFrame()
        result["Customer"] = df[id_col] if id_col else [f"Row {i+1}" for i in range(len(df))]
        result["Churn Risk %"] = (proba * 100).round(1)
        result["Risk Level"] = pd.cut(proba, bins=[-0.01, 0.3, 0.5, 1.0], labels=["Low", "Medium", "High"])
        result = result.sort_values("Churn Risk %", ascending=False).reset_index(drop=True)

        at_risk = int((proba > 0.5).sum())
        st.success(f"{at_risk} out of {len(df)} customers are High Risk (>50% churn probability). Sorted highest risk first.")
        st.dataframe(result, use_container_width=True)

        csv_out = result.to_csv(index=False).encode()
        st.download_button("Download ranked results as CSV", csv_out, "churn_risk_ranked.csv", "text/csv")

# ── TAB 2: Single customer ──────────────────────────────────────
with tab2:
    st.write("Enter one customer's details to check their individual risk.")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.radio("Gender", ["Male", "Female"], index=1)
        senior_citizen = st.radio("Senior Citizen", ["Yes", "No"], index=1)
        partner = st.radio("Has Partner", ["Yes", "No"], index=1)
        dependents = st.radio("Has Dependents", ["Yes", "No"], index=1)
        tenure = st.slider("Tenure (months as customer)", 0, 72, 12)
        contract = st.radio("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        paperless_billing = st.radio("Paperless Billing", ["Yes", "No"], index=0)

    with col2:
        phone_service = st.radio("Phone Service", ["Yes", "No"], index=0)
        multiple_lines = st.radio("Multiple Lines", ["Yes", "No", "No phone service"], index=1)
        internet_service = st.radio("Internet Service", ["DSL", "Fiber optic", "No"], index=1)
        online_security = st.radio("Online Security", ["Yes", "No", "No internet service"], index=1)
        online_backup = st.radio("Online Backup", ["Yes", "No", "No internet service"], index=1)
        device_protection = st.radio("Device Protection", ["Yes", "No", "No internet service"], index=1)
        tech_support = st.radio("Tech Support", ["Yes", "No", "No internet service"], index=1)
        streaming_tv = st.radio("Streaming TV", ["Yes", "No", "No internet service"], index=1)
        streaming_movies = st.radio("Streaming Movies", ["Yes", "No", "No internet service"], index=1)

    c1, c2 = st.columns(2)
    monthly_charges = c1.number_input("Monthly Charges ($)", value=70.0)
    total_charges = c2.number_input("Total Charges to Date ($)", value=840.0)

    if st.button("Predict Churn Risk", type="primary"):
        raw = {
            'gender': gender, 'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone_service, 'MultipleLines': multiple_lines,
            'InternetService': internet_service, 'OnlineSecurity': online_security,
            'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
            'TechSupport': tech_support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies, 'Contract': contract,
            'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }
        row = []
        for col in FEATURE_NAMES:
            val = raw[col]
            if col in encoders:
                le = encoders[col]
                val = le.transform([val])[0] if val in le.classes_ else 0
            row.append(val)

        X = pd.DataFrame([row], columns=FEATURE_NAMES)
        proba = model.predict_proba(X)[0][1]

        if proba > 0.5:
            st.error(f"🔴 High Risk — {proba*100:.1f}% churn probability")
        elif proba > 0.3:
            st.warning(f"🟡 Medium Risk — {proba*100:.1f}% churn probability")
        else:
            st.success(f"🟢 Low Risk — {proba*100:.1f}% churn probability")

        st.progress(float(proba))

st.divider()
st.caption("Top churn drivers: contract type, tenure, and lack of tech support / online security add-ons.")
