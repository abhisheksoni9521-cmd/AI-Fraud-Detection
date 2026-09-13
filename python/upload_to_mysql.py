import pandas as pd
import mysql.connector
from getpass import getpass

# ==========================================
# MYSQL CONNECTION
# ==========================================

password = getpass("Enter MySQL password: ")

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password=password,
    database="fraud_detection"
)

cursor = connection.cursor()

print("\nMySQL connected successfully!")

# ==========================================
# LOAD CSV
# ==========================================

df = pd.read_csv("../data/fraud_predictions.csv")

print("CSV loaded successfully!")
print("Total records:", len(df))

# ==========================================
# SELECT REQUIRED COLUMNS
# ==========================================

data = df[
    [
        "Time",
        "Amount",
        "Hour",
        "Log_Amount",
        "Class",
        "Predicted_Fraud",
        "Fraud_Probability",
        "Fraud_Risk_Score",
        "Risk_Level"
    ]
]

# Convert to tuples
records = list(data.itertuples(index=False, name=None))

# ==========================================
# INSERT DATA
# ==========================================

query = """
INSERT INTO transactions
(
    time_value,
    amount,
    hour,
    log_amount,
    actual_fraud,
    predicted_fraud,
    fraud_probability,
    fraud_risk_score,
    risk_level
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

batch_size = 5000

for i in range(0, len(records), batch_size):

    batch = records[i:i + batch_size]

    cursor.executemany(query, batch)
    connection.commit()

    print(
        f"Uploaded {min(i + batch_size, len(records))} "
        f"/ {len(records)} records"
    )

# ==========================================
# CLOSE CONNECTION
# ==========================================

cursor.close()
connection.close()

print("\n================================")
print("DATA UPLOAD COMPLETED!")
print("================================")
print("All transactions uploaded to MySQL.")