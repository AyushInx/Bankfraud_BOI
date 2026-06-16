"""Application configuration using pydantic-settings."""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # App
    app_name: str = "MuleShield AI"
    app_version: str = "1.0.0"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://muleshield:muleshield@db:5432/muleshield"

    # ML Artifacts
    model_path: str = str(BASE_DIR / "ml_artifacts" / "best_model_catboost.joblib")
    scaler_path: str = str(BASE_DIR / "ml_artifacts" / "lr_scaler.joblib")
    features_path: str = str(BASE_DIR / "ml_artifacts" / "selected_features_list.json")
    model_config_path: str = str(BASE_DIR / "ml_artifacts" / "model_config.json")
    features_csv_path: str = str(BASE_DIR / "ml_artifacts" / "selected_features.csv")

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Risk Tiers
    risk_low_max: int = 39
    risk_medium_max: int = 69
    risk_high_max: int = 89

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "protected_namespaces": ()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
