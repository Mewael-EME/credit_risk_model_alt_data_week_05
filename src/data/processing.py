import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce", utc=True)
    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year
    return df

def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("CustomerId").agg({
        "Amount": ["sum","mean","std","count"],
        "Value": ["sum","mean"]
    }).reset_index()
    g.columns = ["CustomerId","amount_sum","amount_mean","amount_std","transaction_count","value_sum","value_mean"]
    return g

def mode_or_unknown(s: pd.Series) -> str:
    m = s.mode(dropna=True)
    return m.iloc[0] if not m.empty else "Unknown"

def merge_customer_cats(df: pd.DataFrame) -> pd.DataFrame:
    cats = df.groupby("CustomerId").agg({
        "ChannelId": mode_or_unknown,
        "ProductCategory": mode_or_unknown,
        "PricingStrategy": mode_or_unknown
    }).reset_index()
    return cats

def build_preprocessor(numerical_cols, categorical_cols) -> ColumnTransformer:
    num = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, numerical_cols), ("cat", cat, categorical_cols)])
