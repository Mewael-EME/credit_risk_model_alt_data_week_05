import pandas as pd
from src.target_engineering import compute_rfm

def test_compute_rfm():
    test_data = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "TransactionId": [101, 102, 201],
        "TransactionStartTime": ["2021-12-01", "2021-12-10", "2021-11-01"],
        "Amount": [100, 200, 300]
    })

    rfm = compute_rfm(test_data, snapshot_date="2021-12-31")

    assert "CustomerId" in rfm.columns
    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns
    assert rfm.shape[0] == 2  # Two customers
