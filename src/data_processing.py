# src/data_processing.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    agg_df = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'std', 'count'],
        'Value': ['sum', 'mean'],
    }).reset_index()

    # Flatten MultiIndex columns
    agg_df.columns = ['CustomerId', 'amount_sum', 'amount_mean', 'amount_std', 'transaction_count',
                      'value_sum', 'value_mean']
    return agg_df

def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year
    return df

def get_preprocessor(numerical_cols, categorical_cols):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    return preprocessor
