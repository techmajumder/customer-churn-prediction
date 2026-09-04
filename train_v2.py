"""
Customer Churn Prediction — Advanced Training Pipeline (v2)
Adds: 5-fold cross-validation, model comparison (RF vs XGBoost vs Logistic Regression),
and SHAP explainability for per-customer predictions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
)
from xgboost import XGBClassifier
import shap

# ── 1. Load & clean ──────────────────────────────────────────────
df = pd.read_csv("data.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
df = df.drop(columns=['customerID'])

target = (df['Churn'] == 'Yes').astype(int)
df = df.drop(columns=['Churn'])

cat_cols = df.select_dtypes(include='object').columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

feature_names = df.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    df, target, test_size=0.2, random_state=42, stratify=target
)

# ── 2. CROSS-VALIDATION + MODEL COMPARISON ───────────────────────
print("=" * 60)
print("STEP 1: Comparing models with 5-fold cross-validation")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric='logloss', random_state=42
    ),
}

cv_results = {}
for name, clf in candidates.items():
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc')
    cv_results[name] = {"mean_auc": scores.mean(), "std_auc": scores.std(), "fold_scores": scores.tolist()}
    print(f"{name:22s} | ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})  folds: {np.round(scores, 3)}")

best_model_name = max(cv_results, key=lambda k: cv_results[k]["mean_auc"])
print(f"\nBest model by cross-validated ROC-AUC: {best_model_name}")

# ── 3. Train the winning model on the full training set ─────────
print("\n" + "=" * 60)
print(f"STEP 2: Training final model ({best_model_name}) on full train set")
print("=" * 60)

model = candidates[best_model_name]
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, target_names=['Stayed', 'Churned'], output_dict=True)
auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)

print(classification_report(y_test, y_pred, target_names=['Stayed', 'Churned']))
print(f"Test set ROC-AUC: {auc:.3f}")

# ── 4. Standard evaluation plots ─────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stayed', 'Churned'], yticklabels=['Stayed', 'Churned'])
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(f'Confusion Matrix ({best_model_name})')
plt.tight_layout(); plt.savefig('outputs/confusion_matrix.png', dpi=150); plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f'ROC-AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve')
plt.legend(); plt.tight_layout(); plt.savefig('outputs/roc_curve.png', dpi=150); plt.close()

# ── 5. Model comparison chart ────────────────────────────────────
plt.figure(figsize=(7, 4))
names = list(cv_results.keys())
means = [cv_results[n]["mean_auc"] for n in names]
stds = [cv_results[n]["std_auc"] for n in names]
colors = ['#4C72B0' if n != best_model_name else '#55A868' for n in names]
plt.barh(names, means, xerr=stds, color=colors)
plt.xlabel('Cross-Validated ROC-AUC')
plt.title('Model Comparison (5-fold CV)')
plt.xlim(0.5, 1.0)
plt.tight_layout(); plt.savefig('outputs/model_comparison.png', dpi=150); plt.close()

# ── 6. Feature importance (skip cleanly for Logistic Regression) ─
if hasattr(model, 'feature_importances_'):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(7, 6))
    importances.head(12).plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.xlabel('Importance'); plt.title('Top Factors Driving Customer Churn')
    plt.tight_layout(); plt.savefig('outputs/feature_importance.png', dpi=150); plt.close()
    print("\nTop 8 churn drivers:\n", importances.head(8))
else:
    importances = pd.Series(np.abs(model.coef_[0]), index=feature_names).sort_values(ascending=False)

# ── 7. SHAP explainability ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Computing SHAP values for explainability")
print("=" * 60)

# Use a sample for speed
X_sample = X_test.sample(min(200, len(X_test)), random_state=42)

if best_model_name in ("Random Forest", "XGBoost"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # Handle all known SHAP output shapes for binary classifiers:
    # - list of 2 arrays [class0, class1] (older shap)
    # - 3D array (samples, features, classes) (newer shap)
    # - plain 2D array (some binary XGBoost cases)
    if isinstance(shap_values, list):
        shap_vals_churn = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_vals_churn = shap_values[:, :, 1]
    else:
        shap_vals_churn = shap_values
else:
    explainer = shap.LinearExplainer(model, X_train)
    shap_vals_churn = explainer.shap_values(X_sample)

plt.figure()
shap.summary_plot(shap_vals_churn, X_sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('outputs/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

joblib.dump(explainer, 'outputs/shap_explainer.joblib')
print("Saved SHAP summary plot and explainer.")

# ── 8. Save everything ────────────────────────────────────────────
joblib.dump(model, 'outputs/churn_model.joblib')
joblib.dump(encoders, 'outputs/encoders.joblib')

with open('outputs/feature_names.json', 'w') as f:
    json.dump(feature_names, f)

metrics = {
    'best_model': best_model_name,
    'cv_comparison': {k: {'mean_auc': v['mean_auc'], 'std_auc': v['std_auc']} for k, v in cv_results.items()},
    'test_accuracy': report['accuracy'],
    'test_precision_churn': report['Churned']['precision'],
    'test_recall_churn': report['Churned']['recall'],
    'test_f1_churn': f1,
    'test_roc_auc': auc,
}
with open('outputs/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nAll done. Saved model, encoders, plots, and SHAP explainer to outputs/")
print(f"\nFINAL SUMMARY:")
print(f"  Winning model: {best_model_name}")
print(f"  Test ROC-AUC: {auc:.3f}")
print(f"  Test Recall (Churned): {report['Churned']['recall']:.3f}")
