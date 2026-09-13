import pandas as pd
import joblib

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("../data/fraud_features.csv")

# Remove categorical column
df = df.drop("Amount_Category", axis=1)

# Separate features
X = df.drop("Class", axis=1)

# ==========================================
# LOAD RANDOM FOREST MODEL
# ==========================================

model = joblib.load(
    "../model/random_forest_model.pkl"
)

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(X)

probabilities = model.predict_proba(X)[:, 1]

# ==========================================
# CREATE RESULT DATAFRAME
# ==========================================

results = df.copy()

results["Predicted_Fraud"] = predictions

results["Fraud_Probability"] = probabilities

results["Fraud_Risk_Score"] = (
    probabilities * 100
).round(2)


# ==========================================
# RISK LEVEL
# ==========================================

def risk_level(probability):

    if probability >= 0.80:
        return "High Risk"

    elif probability >= 0.50:
        return "Medium Risk"

    else:
        return "Low Risk"


results["Risk_Level"] = results[
    "Fraud_Probability"
].apply(risk_level)


# ==========================================
# SAVE RESULTS
# ==========================================

results.to_csv(
    "../data/fraud_predictions.csv",
    index=False
)

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n========== AI PREDICTION RESULTS ==========")

print(results[
    [
        "Amount",
        "Class",
        "Predicted_Fraud",
        "Fraud_Risk_Score",
        "Risk_Level"
    ]
].head(20))

print("\nRisk Level Distribution:")

print(
    results["Risk_Level"].value_counts()
)

print("\nAI prediction completed successfully!")

print(
    "\nFile saved: data/fraud_predictions.csv"
)