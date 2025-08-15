import os, pandas as pd, streamlit as st, mlflow, mlflow.pyfunc
from pathlib import Path

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("Credit Risk Probability — Executive Dashboard")
st.markdown("**Goal:** Estimate risk and support lending decisions with transparency.")

# Load latest processed data (for basic viz)
proc_path = Path("data/processed/processed_with_labels.csv")
df = pd.read_csv(proc_path) if proc_path.exists() else None

# Load best model from MLflow registry or local fallback
MODEL_URI = os.getenv("MODEL_URI", "models:/best_credit_model/production")
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    st.success(f"Loaded model from MLflow: {MODEL_URI}")
except Exception:
    st.warning("Could not load from MLflow. Using placeholder threshold=0.5.")
    model = None

left, right = st.columns([1,1])
with left:
    st.subheader("Enter Applicant Features")
    Recency = st.number_input("Recency (days since last tx)", min_value=0, value=45)
    Frequency = st.number_input("Frequency (tx count)", min_value=0, value=2)
    Monetary = st.number_input("Monetary (sum)", min_value=0.0, value=150.0)
    threshold = st.slider("Approval Threshold", 0.05, 0.95, 0.5, 0.01)

with right:
    st.subheader("Risk Assessment")
    if st.button("Predict Risk"):
        X = pd.DataFrame([{"Recency":Recency,"Frequency":Frequency,"Monetary":Monetary}])
        if model is not None:
            proba = float(model.predict(X)[0])
        else:
            # fallback heuristic
            proba = min(0.99, max(0.01, 0.6*(Recency>30) + 0.3*(Frequency<2) + 0.1*(Monetary<100)))
        decision = "Approve" if proba < threshold else "Reject / Review"
        st.metric("Predicted Risk", f"{proba:.3f}")
        st.metric("Decision", decision)
        st.caption("Lower is better (probability of default proxy).")

if df is not None:
    st.subheader("Portfolio Snapshot")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accounts", f"{df['CustomerId'].nunique():,}")
    col2.metric("High-Risk (label=1)", f"{int(df['is_high_risk'].sum()):,}")
    col3.metric("Avg Monetary", f"{df['Monetary'].mean():.1f}" if "Monetary" in df else "n/a")
