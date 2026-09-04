# 📉 Customer Churn Prediction

Predicts whether a telecom customer is likely to cancel their subscription, and explains *why* — so the business can intervene before they leave.

**🔗 Live app:** https://customer-churn-prediction-sc6uf4jrevrvrmqdhvpbyw.streamlit.app/

## Business Problem

Acquiring a new customer costs far more than retaining an existing one. If a company can identify **which customers are at risk of leaving before they actually cancel**, it can proactively offer discounts, better support, or improved plans to retain them. This project builds that early-warning system — including a batch tool that scores an entire customer list and ranks them by risk, not just one customer at a time.

## Dataset

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) — 7,043 real-style telecom customers with billing, service usage, and account details, labeled with actual churn outcomes. 26.5% churn rate (realistic, imbalanced — handled with class-weighted training).

## Approach

- **Model:** Random Forest Classifier (`class_weight='balanced'` to handle the imbalanced churn rate)
- **Features:** Contract type, tenure, monthly/total charges, internet service, security/support add-ons, payment method, demographics
- **Evaluation:** Train/test split (80/20, stratified), evaluated on precision/recall/F1 for the churned class specifically (accuracy alone is misleading on imbalanced data) plus ROC-AUC
- **App:** Two-tab Streamlit interface — batch scoring (upload a CSV, get every customer ranked by risk) and single-customer manual entry

## Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.843 |
| Recall (Churned customers caught) | 79% |
| Precision (Churned) | 54% |
| Accuracy | 77% |

**Why recall matters more than precision here:** missing an at-risk customer (false negative) costs the business a lost customer; flagging a loyal customer as at-risk (false positive) just costs a wasted discount offer. So the model is tuned to catch as many true churners as possible.

### Top Churn Drivers (from feature importance)
1. **Contract type** — month-to-month customers churn far more than annual-contract customers
2. **Tenure** — newer customers are highest risk
3. **Total/Monthly charges**
4. **Lack of Online Security / Tech Support add-ons**

**Business takeaway:** incentivizing longer contracts and bundling tech support / security add-ons are concrete, data-backed retention levers.

See `outputs/confusion_matrix.png`, `outputs/roc_curve.png`, `outputs/feature_importance.png` for full evaluation.

## Project Structure

```
├── data.csv                    # Telco churn dataset
├── train.py                    # Full training + evaluation pipeline
├── streamlit_app.py            # Live Streamlit app (batch + single prediction)
├── app.py                      # Gradio version (kept for reference)
├── requirements.txt
├── sample_customers.csv        # Sample file to test batch scoring
└── outputs/
    ├── churn_model.joblib       # Trained model
    ├── encoders.joblib          # Label encoders for categorical features
    ├── feature_names.json
    ├── metrics.json
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```

## How to Run Locally

```bash
pip install -r requirements.txt
python train.py            # retrain the model (optional, already trained)
streamlit run streamlit_app.py
```

## Author

Sayan Majumder
