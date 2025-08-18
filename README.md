🏦 Credit Risk Modeling – Finance Capstone Project
📌 Business Objective

In the finance sector, credit risk modeling is central to minimizing default risk while maintaining customer trust.
This project demonstrates how advanced machine learning and robust engineering practices can create a reliable, explainable, and deployable credit risk model.

The capstone highlights:

- 🔒 Modular, testable, and CI/CD integrated codebase.
- 📊 Explainability using SHAP to help finance stakeholders understand model predictions.
- ⚡ Deployable API + interactive dashboard for decision-makers.


🚀 Solution Pipeline

1. Data Preparation
   - Raw financial transactions → data/raw/data.csv
   - Cleaning + Feature Engineering (RFM analysis, clustering, labeling high-risk customers).

2. Target Engineering
   - RFM segmentation (Recency, Frequency, Monetary).
   - High-risk labeling via clustering.

3. Model Training with MLflow
   - Logistic Regression, Random Forest, XGBoost.
   - Automated experiment tracking + best model selection.

4. Model Explainability
   - SHAP plots for transparency.
   - Business-focused insights (why a customer is flagged as high-risk).

5. Deployment
   - FastAPI service for real-time inference.
   - Dockerized for portability and production readiness.
   - Optional Streamlit Dashboard for interactive decision-making.

📂 Project Structure
```
credit-risk-model/
│
├── data/
│   ├── raw/                 # Raw financial data
│   ├── processed/           # Processed + labeled data
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── target_engineering.py
│   ├── train.py             # Model training + MLflow logging
│   ├── model.py             # Model save/load helpers
│   ├── predict.py           # Prediction logic
│   ├── explainability.py    # SHAP plots + explainability
│
├── api/                     # FastAPI app
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── dashboard/               # Optional Streamlit app
│   └── app.py
│
├── tests/                   # Unit & integration tests
│   ├── test_data_preprocessing.py
│   ├── test_target_engineering.py
│   ├── test_train.py
│   └── test_api.py
│
├── .github/workflows/       # CI/CD pipelines
│   └── ci.yml
│
├── requirements.txt
├── README.md
└── mlruns/                  # MLflow experiment tracking
```

⚙️ How to Run
1️⃣ Local Setup
```
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model
pip install -r requirements.txt
```
Run preprocessing + training:
```
python -m src.train
```
Run FastAPI locally:
```
uvicorn api.main:app --reload
```

2️⃣ Docker Setup
```
docker build -t credit-risk-api ./api
docker run -p 8000:8000 credit-risk-api
```
3️⃣ Run Dashboard (Optional)
```
streamlit run dashboard/app.py
```

✅ Testing
Unit and integration tests are included. Run with:
```
pytest tests/
```
CI/CD pipeline (.github/workflows/ci.yml) ensures:
 - Code linting
 - Unit tests
 - Build verification

📊 Model Explainability (SHAP Example)

 - Global feature importance plots (which features matter most).
 - Local explanations (why this customer is predicted as high-risk).

This provides transparency to financial stakeholders.

🌟 Business Impact

 - 🎯 Helps financial institutions identify high-risk customers early.
 - 🕵️ Supports explainable decision-making with regulators and auditors.
 - 🔄 Easily deployable in production with FastAPI + Docker.