"""
Customer Churn Prediction — Training Pipeline
Dataset: Telco Customer Churn (IBM)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score
)

# ── 1. Load & clean ──────────────────────────────────────────────
df = pd.read_csv("data.csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])  # new customers, 0 tenure
df = df.drop(columns=['customerID'])

target = (df['Churn'] == 'Yes').astype(int)
df = df.drop(columns=['Churn'])

# ── 2. Encode categorical features ───────────────────────────────
cat_cols = df.select_dtypes(include='object').columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

feature_names = df.columns.tolist()

# ── 3. Train / test split ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df, target, test_size=0.2, random_state=42, stratify=target
)

# ── 4. Train model ───────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    class_weight='balanced',   # important: churn is imbalanced (26.5%)
    random_state=42
)
model.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, target_names=['Stayed', 'Churned'], output_dict=True)
auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)

print(classification_report(y_test, y_pred, target_names=['Stayed', 'Churned']))
print(f"ROC-AUC: {auc:.3f}")

# ── 6. Confusion matrix ──────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stayed', 'Churned'], yticklabels=['Stayed', 'Churned'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150)
plt.close()

# ── 7. ROC curve ──────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f'ROC-AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/roc_curve.png', dpi=150)
plt.close()

# ── 8. Feature importance (the "why" — the useful business insight) ──
importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
plt.figure(figsize=(7, 6))
importances.head(12).plot(kind='barh')
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.title('Top Factors Driving Customer Churn')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150)
plt.close()

print("\nTop 8 churn drivers:")
print(importances.head(8))

# ── 9. Save model + encoders + metadata ──────────────────────────
joblib.dump(model, 'outputs/churn_model.joblib')
joblib.dump(encoders, 'outputs/encoders.joblib')

with open('outputs/feature_names.json', 'w') as f:
    json.dump(feature_names, f)

metrics = {
    'accuracy': report['accuracy'],
    'precision_churn': report['Churned']['precision'],
    'recall_churn': report['Churned']['recall'],
    'f1_churn': f1,
    'roc_auc': auc,
    'top_features': importances.head(8).to_dict()
}
with open('outputs/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nSaved model, encoders, and plots to outputs/")
