"""
Diabetes Risk Prediction - EDA + Model Training
Run: python train_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

RANDOM_STATE = 42

# 1. LOAD DATA
df = pd.read_csv("data/diabetes.csv")
print("Shape:", df.shape)
print(df.head())
print("\nClass balance:\n", df["Outcome"].value_counts(normalize=True))

# 2. CLEAN: zeros in these columns are missing values, not real zeros
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
print("\nMissing values after conversion:\n", df[zero_as_missing].isna().sum())

for col in zero_as_missing:
    df[col] = df[col].fillna(df[col].median())

# 3. EDA PLOTS
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0, 0])
axes[0, 0].set_title("Feature correlation heatmap")

sns.histplot(data=df, x="Glucose", hue="Outcome", kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Glucose distribution by outcome")

sns.boxplot(data=df, x="Outcome", y="BMI", ax=axes[1, 0])
axes[1, 0].set_title("BMI by outcome")

df["Outcome"].value_counts().plot(kind="bar", ax=axes[1, 1], color=["#5DCAA5", "#D85A30"])
axes[1, 1].set_title("Class balance (0 = No diabetes, 1 = Diabetes)")
axes[1, 1].set_xticklabels(["No diabetes", "Diabetes"], rotation=0)

plt.tight_layout()
plt.savefig("docs/eda_plots.png", dpi=150)
print("\nSaved docs/eda_plots.png")

# 4. TRAIN/TEST SPLIT + SCALING
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. TRAIN MULTIPLE MODELS AND COMPARE
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", random_state=RANDOM_STATE
    ),
}

results = []
fitted_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, preds), 3),
        "Precision": round(precision_score(y_test, preds), 3),
        "Recall": round(recall_score(y_test, preds), 3),
        "F1": round(f1_score(y_test, preds), 3),
        "ROC-AUC": round(roc_auc_score(y_test, probs), 3),
    })
    fitted_models[name] = model

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\nModel comparison:\n", results_df.to_string(index=False))
results_df.to_csv("docs/model_metrics.csv", index=False)

# 6. SELECT BEST MODEL AND SAVE IT
import sys, os

best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]
print(f"\nBest model: {best_model_name}", flush=True)

try:
    print("\nClassification report for best model:", flush=True)
    print(classification_report(y_test, best_model.predict(X_test_scaled)), flush=True)

    print("Current working directory:", os.getcwd(), flush=True)
    print("models/ exists:", os.path.isdir("models"), flush=True)

    joblib.dump(best_model, "models/diabetes_model.pkl")
    print("Saved diabetes_model.pkl", flush=True)

    joblib.dump(scaler, "models/scaler.pkl")
    print("Saved scaler.pkl", flush=True)

    joblib.dump(list(X.columns), "models/feature_order.pkl")
    print("Saved feature_order.pkl", flush=True)

    print("\nAll files saved successfully.", flush=True)

except Exception as e:
    print("ERROR while saving model files:", repr(e), flush=True)
    import traceback
    traceback.print_exc()