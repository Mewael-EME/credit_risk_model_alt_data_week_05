# src/eda_utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def overview(df: pd.DataFrame):
    print("Dataset shape:", df.shape)
    print("\nColumn types:\n", df.dtypes)
    print("\nFirst 5 rows:\n", df.head())

def summary_statistics(df: pd.DataFrame):
    return df.describe()

def plot_numerical_distributions(df: pd.DataFrame, numerical_cols: list):
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

def plot_categorical_distributions(df: pd.DataFrame, categorical_cols: list):
    for col in categorical_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=df, order=df[col].value_counts().index)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.show()

def correlation_heatmap(df: pd.DataFrame, numerical_cols: list):
    corr = df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

def missing_values_report(df: pd.DataFrame):
    missing = df.isnull().sum()
    return missing[missing > 0].sort_values(ascending=False)

def detect_outliers(df: pd.DataFrame, column: str):
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=column, data=df)
    plt.title(f"Outlier Detection for {column}")
    plt.show()
