import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime

def compute_rfm(df: pd.DataFrame, snapshot_date: str = "2021-12-31") -> pd.DataFrame:
    snapshot = pd.to_datetime(snapshot_date)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    rfm = df.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot - x.max()).days,
        "TransactionId": "count",
        "Amount": "sum"
    }).reset_index()

    rfm.columns = ["CustomerId", "Recency", "Frequency", "Monetary"]
    return rfm

def cluster_rfm(rfm: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    return rfm

def label_high_risk_cluster(rfm_clustered: pd.DataFrame) -> pd.DataFrame:
    cluster_stats = rfm_clustered.groupby("Cluster").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean"
    })

    # High risk: low frequency, low monetary value, high recency
    high_risk_cluster = cluster_stats.sort_values(
        by=["Frequency", "Monetary", "Recency"],
        ascending=[True, True, False]
    ).index[0]

    rfm_clustered["is_high_risk"] = (rfm_clustered["Cluster"] == high_risk_cluster).astype(int)
    return rfm_clustered[["CustomerId", "is_high_risk"]]

