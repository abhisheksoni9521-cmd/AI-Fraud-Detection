import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load data
df = pd.read_csv("../data/fraud_features.csv")

# Remove categorical feature
df = df.drop("Amount_Category", axis=1)

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Random Forest...")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Fraud probability
y_probability = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n========== RANDOM FOREST REPORT ==========")

print(classification_report(
    y_test,
    y_pred,
    digits=4
))

print("\n========== CONFUSION MATRIX ==========")

print(confusion_matrix(
    y_test,
    y_pred
))

print("\n========== ROC-AUC SCORE ==========")

auc_score = roc_auc_score(
    y_test,
    y_probability
)

print("ROC-AUC:", round(auc_score, 4))

# Save model
joblib.dump(
    rf_model,
    "../model/random_forest_model.pkl"
)

print("\nRandom Forest model saved successfully!")