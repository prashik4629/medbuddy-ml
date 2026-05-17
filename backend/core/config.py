from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Project root
    PROJECT_ROOT: Path

    # Directories
    LOG_DIR: str = "logs"
    DATASET_DIR: str = "dataset"
    MODEL_DIR: str = "model_dir"

    # File names
    DATASET_NAME: str = "heart.csv"
    LOG_NAME: str = "app.log"
    MODEL_NAME: str = "heart_disease_prediction_model.joblib"

    # ML settings
    TARGET_COL: str = "target"
    TEST_SIZE: float = 0.15
    RANDOM_STATE: int = 42

    # API settings
    API_KEY: str
    API_URL: str = "http://127.0.0.1:8000/api/v1/predict"

    @property
    def DATASET_PATH(self) -> Path:
        return self.PROJECT_ROOT / self.DATASET_DIR / self.DATASET_NAME

    @property
    def MODEL_PATH(self) -> Path:
        return self.PROJECT_ROOT / self.MODEL_DIR / self.MODEL_NAME

    @property
    def LOG_PATH(self) -> Path:
        return self.PROJECT_ROOT / self.LOG_DIR / self.LOG_NAME

    class Config:
        env_file = ".env"


settings = Settings()