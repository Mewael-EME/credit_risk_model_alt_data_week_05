from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_models():
    return {
        "LogisticRegression": LogisticRegression(solver="liblinear"),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
