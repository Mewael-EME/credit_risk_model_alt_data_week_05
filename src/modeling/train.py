import mlflow, mlflow.sklearn
import pandas as pd
from src.core.config import settings
from src.core.logging import get_logger
from src.modeling.datasets import load_training_dataframe, train_test
from src.modeling.models import get_models
from src.modeling.evaluate import metrics_dict, confusion
from src.modeling.explain import shap_summary
import matplotlib.pyplot as plt

log = get_logger()

def train_and_register(df: pd.DataFrame | None = None, experiment: str | None = None, register: bool = True):
    mlflow.set_experiment(experiment or settings.EXPERIMENT)
    df = df if df is not None else load_training_dataframe()
    X_train, X_test, y_train, y_test = train_test(df)

    best = {"roc_auc": -1, "name": None, "model": None}
    for name, model in get_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:,1]

            m = metrics_dict(y_test, y_pred, y_proba)
            mlflow.log_metrics(m)
            mlflow.log_params(model.get_params())

            # SHAP
            try:
                fig = shap_summary(model, X_train.sample(min(300, len(X_train)), random_state=settings.RANDOM_STATE))
                mlflow.log_figure(fig, f"shap_summary_{name}.png")
                plt.close(fig)
            except Exception as e:
                log.warning(f"SHAP skipped for {name}: {e}")

            mlflow.sklearn.log_model(model, artifact_path="model")

            if m["roc_auc"] > best["roc_auc"]:
                best.update({"roc_auc": m["roc_auc"], "name": name, "model": model})

    log.info(f"Best model: {best['name']} (ROC-AUC={best['roc_auc']:.3f})")

    if register and best["model"] is not None:
        with mlflow.start_run(run_name=f"register_{best['name']}"):
            mlflow.sklearn.log_model(best["model"], artifact_path="model",
                                     registered_model_name=settings.MODEL_NAME)
    return {"best_model_name": best["name"], "best_roc_auc": best["roc_auc"]}

if __name__ == "__main__":
    train_and_register()
