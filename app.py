"""
Customer Churn Prediction — Gradio App
Enter a customer's details, get a churn risk score + explanation.
"""
import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr

model = joblib.load("outputs/churn_model.joblib")
encoders = joblib.load("outputs/encoders.joblib")
with open("outputs/feature_names.json") as f:
    FEATURE_NAMES = json.load(f)


def predict_churn(gender, senior_citizen, partner, dependents, tenure,
                   phone_service, multiple_lines, internet_service,
                   online_security, online_backup, device_protection,
                   tech_support, streaming_tv, streaming_movies,
                   contract, paperless_billing, payment_method,
                   monthly_charges, total_charges):

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

    risk = "🔴 High Risk" if proba > 0.5 else ("🟡 Medium Risk" if proba > 0.3 else "🟢 Low Risk")
    return {"Will Churn": float(proba), "Will Stay": float(1 - proba)}, f"{risk} — {proba*100:.1f}% churn probability"


def predict_batch(file):
    if file is None:
        return None, "Upload a CSV file first."

    df = pd.read_csv(file.name)

    # keep an ID column for display if present, otherwise make one
    id_col = None
    for candidate in ["customerID", "CustomerID", "customer_id", "ID", "Name"]:
        if candidate in df.columns:
            id_col = candidate
            break

    work = df.copy()
    if id_col:
        work = work.drop(columns=[id_col])

    if 'TotalCharges' in work.columns:
        work['TotalCharges'] = pd.to_numeric(work['TotalCharges'], errors='coerce')
        work['TotalCharges'] = work['TotalCharges'].fillna(work['MonthlyCharges'])

    for col in FEATURE_NAMES:
        if col in encoders:
            le = encoders[col]
            work[col] = work[col].apply(lambda v: le.transform([v])[0] if v in le.classes_ else 0)

    X = work[FEATURE_NAMES]
    proba = model.predict_proba(X)[:, 1]

    result = pd.DataFrame()
    result['Customer'] = df[id_col] if id_col else [f"Row {i+1}" for i in range(len(df))]
    result['Churn Risk %'] = (proba * 100).round(1)
    result['Risk Level'] = pd.cut(proba, bins=[-0.01, 0.3, 0.5, 1.0], labels=['Low', 'Medium', 'High'])
    result = result.sort_values('Churn Risk %', ascending=False).reset_index(drop=True)

    at_risk_count = (proba > 0.5).sum()
    summary = f"{at_risk_count} out of {len(df)} customers are High Risk (>50% churn probability). Sorted highest risk first."

    return result, summary


with gr.Blocks(title="Customer Churn Predictor") as demo:
    gr.Markdown("# 📉 Customer Churn Prediction")

    with gr.Tab("Batch: Score My Whole Customer List"):
        gr.Markdown(
            "Upload a CSV of customers (same columns as the Telco dataset). "
            "Get every customer's churn risk, ranked from most to least likely to leave — "
            "this is the version a business would actually use to decide who to call first."
        )
        file_input = gr.File(label="Upload customer CSV", file_types=[".csv"])
        batch_btn = gr.Button("Score All Customers", variant="primary")
        summary_output = gr.Textbox(label="Summary")
        table_output = gr.Dataframe(label="Customers Ranked by Churn Risk")

        batch_btn.click(predict_batch, inputs=[file_input], outputs=[table_output, summary_output])

    with gr.Tab("Single Customer: Manual Entry"):
        gr.Markdown("Enter one customer's details to check their individual risk.")
        with gr.Row():
            with gr.Column():
                gender = gr.Radio(["Male", "Female"], value="Female", label="Gender")
                senior_citizen = gr.Radio(["Yes", "No"], value="No", label="Senior Citizen")
                partner = gr.Radio(["Yes", "No"], value="No", label="Has Partner")
                dependents = gr.Radio(["Yes", "No"], value="No", label="Has Dependents")
                tenure = gr.Slider(0, 72, value=12, step=1, label="Tenure (months as customer)")
                contract = gr.Radio(["Month-to-month", "One year", "Two year"], value="Month-to-month", label="Contract Type")
                payment_method = gr.Dropdown(
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    value="Electronic check", label="Payment Method")
                paperless_billing = gr.Radio(["Yes", "No"], value="Yes", label="Paperless Billing")

            with gr.Column():
                phone_service = gr.Radio(["Yes", "No"], value="Yes", label="Phone Service")
                multiple_lines = gr.Radio(["Yes", "No", "No phone service"], value="No", label="Multiple Lines")
                internet_service = gr.Radio(["DSL", "Fiber optic", "No"], value="Fiber optic", label="Internet Service")
                online_security = gr.Radio(["Yes", "No", "No internet service"], value="No", label="Online Security")
                online_backup = gr.Radio(["Yes", "No", "No internet service"], value="No", label="Online Backup")
                device_protection = gr.Radio(["Yes", "No", "No internet service"], value="No", label="Device Protection")
                tech_support = gr.Radio(["Yes", "No", "No internet service"], value="No", label="Tech Support")
                streaming_tv = gr.Radio(["Yes", "No", "No internet service"], value="No", label="Streaming TV")
                streaming_movies = gr.Radio(["Yes", "No", "No internet service"], value="No", label="Streaming Movies")

        with gr.Row():
            monthly_charges = gr.Number(value=70.0, label="Monthly Charges ($)")
            total_charges = gr.Number(value=840.0, label="Total Charges to Date ($)")

        predict_btn = gr.Button("Predict Churn Risk", variant="primary")

        with gr.Row():
            label_output = gr.Label(label="Prediction")
            risk_output = gr.Textbox(label="Risk Summary")

        predict_btn.click(
            predict_churn,
            inputs=[gender, senior_citizen, partner, dependents, tenure, phone_service,
                    multiple_lines, internet_service, online_security, online_backup,
                    device_protection, tech_support, streaming_tv, streaming_movies,
                    contract, paperless_billing, payment_method, monthly_charges, total_charges],
            outputs=[label_output, risk_output]
        )

    gr.Markdown(
        "*Model: Random Forest trained on the IBM Telco Customer Churn dataset "
        "(7,043 customers, 84.3% ROC-AUC). Top churn drivers: contract type, "
        "tenure, and lack of tech support / online security add-ons.*"
    )

if __name__ == "__main__":
    demo.launch()
