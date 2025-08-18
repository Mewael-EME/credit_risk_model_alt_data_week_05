import pandas as pd
from src.modeling.train import train_and_register

def test_train_and_register_runs():
    df = pd.DataFrame({
        "CustomerId":[1,2,3,4,5,6,7,8],
        "Recency":[10,50,5,100,40,60,20,80],
        "Frequency":[5,1,8,0,2,3,7,1],
        "Monetary":[500,50,900,10,100,200,800,30],
        "is_high_risk":[0,1,0,1,1,0,0,1]
    })
    out = train_and_register(df, experiment="ci-test", register=False)
    assert "best_model_name" in out
    assert out["best_roc_auc"] >= 0
