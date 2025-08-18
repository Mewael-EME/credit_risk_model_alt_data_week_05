import pandas as pd
from src.label.target_engineering import compute_rfm, cluster_rfm, label_high_risk_cluster

def test_compute_rfm_columns():
    df = pd.DataFrame({
        "CustomerId":[1,1,2],
        "TransactionId":[101,102,201],
        "TransactionStartTime":["2021-12-01","2021-12-10","2021-11-01"],
        "Amount":[100,200,300]
    })
    rfm = compute_rfm(df)
    for c in ["Recency","Frequency","Monetary"]:
        assert c in rfm.columns

def test_label_high_risk_has_binary():
    df = pd.DataFrame({
        "CustomerId":[1,2,3],
        "Recency":[10,100,50],
        "Frequency":[5,1,2],
        "Monetary":[500,10,100]
    })
    clustered = cluster_rfm(df)
    labels = label_high_risk_cluster(clustered)
    assert set(labels["is_high_risk"].unique()).issubset({0,1})
