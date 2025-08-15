from src.utils.evaluation import evaluate_model

def test_evaluate_model():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_proba = [0.1, 0.8, 0.4, 0.3]

    metrics = evaluate_model(y_true, y_pred, y_proba)
    assert "accuracy" in metrics
    assert 0 <= metrics["roc_auc"] <= 1
