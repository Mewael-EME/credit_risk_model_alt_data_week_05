import pandas as pd
from src.modeling.train import train_and_register

def test_train_and_register(tmp_path, monkeypatch):
    # minimal synthetic data
    df = pd.DataFrame({
        "CustomerId":[1,2,3,4],
        "Recency":[10,50,5,100],
        "Frequency":[5,1,8,0],
        "Monetary":[500,50,900,10],
        "is_high_risk":[0,1,0,1]
    })
    out = train_and_register(df, experiment="test-exp", register=False)
    assert "best_model_name" in out
