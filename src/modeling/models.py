from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_models():
    return {
        "LogisticRegression": LogisticRegression(solver="liblinear", max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }
