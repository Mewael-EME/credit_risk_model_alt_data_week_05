import shap
import numpy as np
import matplotlib.pyplot as plt

def shap_summary(model, X, max_display=12):
    try:
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X)
        vals_to_plot = vals if isinstance(vals, np.ndarray) else vals[1]
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X, 20))
        vals_to_plot = explainer.shap_values(X, nsamples=200)
    shap.summary_plot(vals_to_plot, X, show=False, max_display=max_display)
    return plt.gcf()
