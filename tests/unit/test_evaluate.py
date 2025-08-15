from src.modeling.evaluate import metrics_dict

def test_metrics_dict_keys():
    y_true = [0,1,1,0]
    y_pred = [0,1,0,0]
    y_proba= [0.1,0.9,0.4,0.3]
    m = metrics_dict(y_true,y_pred,y_proba)
    for k in ["accuracy","precision","recall","f1","roc_auc"]:
        assert k in m
