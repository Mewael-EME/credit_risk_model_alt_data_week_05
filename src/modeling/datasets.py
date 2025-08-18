import pandas as pd
from sklearn.model_selection import train_test_split
from src.core.config import settings

def load_training_dataframe(path: str | None = None) -> pd.DataFrame:
    return pd.read_csv(path or settings.PROC_PATH)

def split_X_y(df: pd.DataFrame):
    y = df["is_high_risk"].astype(int)
    drop_cols = ["is_high_risk","CustomerId"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y

def train_test(df: pd.DataFrame, test_size: float | None = None, random_state: int | None = None):
    X, y = split_X_y(df)
    return train_test_split(X, y, test_size=test_size or settings.TEST_SIZE,
                            random_state=random_state or settings.RANDOM_STATE, stratify=y)
