import pandas as pd
from src.label.target_engineering import compute_rfm

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
