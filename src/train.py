import pandas as pd
from src.target_engineering import compute_rfm, cluster_rfm, label_high_risk_cluster

# ✅ Step 0: Load your real raw data
raw_df = pd.read_csv("data/raw/data.csv")

# ✅ Step 1: Compute RFM features
rfm = compute_rfm(raw_df)

# ✅ Step 2: Cluster customers
rfm_clustered = cluster_rfm(rfm)

# ✅ Step 3: Label the high-risk group
risk_labels = label_high_risk_cluster(rfm_clustered)

# ✅ Step 4: Merge risk labels back into original data
raw_df = raw_df.merge(risk_labels, on="CustomerId", how="left")
raw_df["is_high_risk"] = raw_df["is_high_risk"].fillna(0).astype(int)

# ✅ Step 5: Save processed data
raw_df.to_csv("data/processed/processed_with_labels.csv", index=False)
print("✅ Processed data with 'is_high_risk' column saved.")

# ----------------------------------------------------------------------------------
# ✅ Model Training Section
# ----------------------------------------------------------------------------------

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from src.utils.evaluation import evaluate_model
from src.utils.models import get_models

# Load processed data
df = pd.read_csv("data/processed/processed_with_labels.csv")

# ✅ Step 6: Drop non-predictive ID columns
df = df.drop(columns=[
    "TransactionId", "BatchId", "AccountId", "SubscriptionId", "CustomerId"
])

# ✅ Step 7: Convert datetime to numeric timestamp
df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
df["TransactionStartTimestamp"] = df["TransactionStartTime"].astype(int) // 10**9
df = df.drop(columns=["TransactionStartTime"])

# ✅ Step 8: One-hot encode categorical columns
categorical_cols = [
    "CurrencyCode", "CountryCode", "ProviderId", "ProductId",
    "ProductCategory", "ChannelId", "PricingStrategy"
]
df = pd.get_dummies(df, columns=categorical_cols)

# ✅ Step 9: Define X and y
X = df.drop(columns=["is_high_risk"])
y = df["is_high_risk"]

# ✅ Step 10: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ✅ Step 11: Train and log models
mlflow.set_experiment("credit-risk-modeling")
models = get_models()

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_proba)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"{name} logged to MLflow with metrics:", metrics)
