import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["TransactionStartTime"] = pd.to_datetime(d["TransactionStartTime"], errors="coerce", utc=True)
    snapshot = d["TransactionStartTime"].max()
    rfm = d.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot - x.max()).days if pd.notnull(x.max()) else np.nan,
        "TransactionId": "count",
        "Amount": "sum"
    }).rename(columns={"TransactionStartTime":"Recency","TransactionId":"Frequency","Amount":"Monetary"})
    return rfm.reset_index()

def cluster_rfm(rfm: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    r = rfm.copy()
    scaler = StandardScaler()
    r_scaled = scaler.fit_transform(r[["Recency","Frequency","Monetary"]].fillna(0))
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    r["Cluster"] = km.fit_predict(r_scaled)
    return r.reset_index(drop=True)

def label_high_risk_cluster(rfm_clustered: pd.DataFrame) -> pd.DataFrame:
    stats = rfm_clustered.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()
    high_cluster = stats.sort_values(["Frequency","Monetary","Recency"], ascending=[True,True,False]).index[0]
    out = rfm_clustered[["CustomerId"]].copy()
    out["is_high_risk"] = (rfm_clustered["Cluster"] == high_cluster).astype(int).values
    return out
