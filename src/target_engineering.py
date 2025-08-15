import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime

def compute_rfm(df):
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], utc=True)

    snapshot = df["TransactionStartTime"].max()

    rfm = df.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot - x.max()).days,
        "TransactionId": "count",
        "Amount": "sum"
    }).rename(columns={
        "TransactionStartTime": "Recency",
        "TransactionId": "Frequency",
        "Amount": "Monetary"
    })

    return rfm

def cluster_rfm(rfm):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    features = rfm[["Recency", "Frequency", "Monetary"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm["cluster"] = kmeans.fit_predict(scaled)

    rfm = rfm.reset_index()  # ✅ This brings back 'CustomerId' as a column

    return rfm


def label_high_risk_cluster(rfm_clustered: pd.DataFrame) -> pd.DataFrame:
    cluster_stats = rfm_clustered.groupby("cluster").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean"
    })

    # High risk: low frequency, low monetary value, high recency
    high_risk_cluster = cluster_stats.sort_values(
        by=["Frequency", "Monetary", "Recency"],
        ascending=[True, True, False]
    ).index[0]

    rfm_clustered["is_high_risk"] = (rfm_clustered["cluster"] == high_risk_cluster).astype(int)
    return rfm_clustered[["CustomerId", "is_high_risk"]]

