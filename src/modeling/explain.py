import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def explain_model(model_path="models/final_model.pkl",
                  data_path="data/processed/processed_with_labels.csv",
                  num_samples=100):
    """
    Generate SHAP explainability plots for the trained model.
    
    Args:
        model_path (str): Path to the trained model pickle file.
        data_path (str): Path to the processed dataset with labels.
        num_samples (int): Number of samples to use for SHAP analysis.
    """
    # Load trained model
    model = joblib.load(model_path)

    # Load processed data
    df = pd.read_csv(data_path)

    # Separate features from target
    if "is_high_risk" in df.columns:
        X = df.drop(columns=["is_high_risk"])
    else:
        X = df.copy()

    # Take a sample for speed
    X_sample = X.sample(min(num_samples, len(X)), random_state=42)

    # Initialize SHAP explainer
    explainer = shap.Explainer(model, X_sample)

    # Calculate SHAP values
    shap_values = explainer(X_sample)

    # Plot feature importance summary
    plt.title("SHAP Summary Plot - Feature Impact")
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig("reports/shap_summary_plot.png", bbox_inches="tight")
    plt.close()

    print("✅ SHAP summary plot saved to reports/shap_summary_plot.png")


if __name__ == "__main__":
    explain_model()
