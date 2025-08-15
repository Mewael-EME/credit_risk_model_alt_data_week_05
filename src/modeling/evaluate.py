from sklearn.metrics import confusion_matrix
import numpy as np

def ks_stat(y_true, y_proba):
    # simple KS
    pos = np.sort(y_proba[np.array(y_true)==1])
    neg = np.sort(y_proba[np.array(y_true)==0])
    import numpy as np
    from scipy import stats
    return stats.ks_2samp(pos, neg).statistic

def metrics_dict(y_true, y_pred, y_proba):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "ks": ks_stat(y_true, y_proba)
    }
