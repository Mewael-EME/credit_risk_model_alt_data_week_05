from pydantic import BaseSettings

class Settings(BaseSettings):
    RAW_PATH: str = "data/raw/data.csv"
    PROC_PATH: str = "data/processed/processed_with_labels.csv"
    EXPERIMENT: str = "credit-risk-modeling"
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2

settings = Settings()
