import pandas as pd
from src.data.processing import extract_datetime_features, create_aggregate_features

def test_extract_datetime_features_adds_columns():
    df = pd.DataFrame({"TransactionStartTime": ["2021-01-01 10:00:00"]})
    out = extract_datetime_features(df.copy())
    for col in ["transaction_hour","transaction_day","transaction_month","transaction_year"]:
        assert col in out.columns

def test_create_aggregate_features_shapes():
    df = pd.DataFrame(
        {"CustomerId":[1,1,2], "Amount":[10,20,30], "Value":[10,20,30]}
    )
    agg = create_aggregate_features(df)
    assert agg.shape[0] == 2
    assert "amount_sum" in agg.columns
