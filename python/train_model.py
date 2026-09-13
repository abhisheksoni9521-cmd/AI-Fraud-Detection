import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ==========================================
# LOAD FEATURE-ENGINEERED DATA
# ==========================================

df = pd.read_csv("../data/fraud_features.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# ==========================================
# PREPARE DATA
# ==========================================

# Remove categorical column for now
df = df.drop("Amount_Category", axis=1)

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures:", X.shape)
print("Target:", y.shape)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# LOGISTIC REGRESSION
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

# Probability
y_probability = model.predict_proba(X_test_scaled)[:, 1]

# ==========================================
# MODEL EVALUATION
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

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

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "../model/fraud_model.pkl")
joblib.dump(scaler, "../model/scaler.pkl")

print("\nModel saved successfully!")
print("Files:")
print("model/fraud_model.pkl")
print("model/scaler.pkl")