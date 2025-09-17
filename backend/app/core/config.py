"""Application configuration and settings management."""
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from pydantic import AnyHttpUrl, validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Application settings
    PROJECT_NAME: str = "Stock Market Analysis API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    SECRET_KEY: str = "your-secret-key-here"  # Change in production!
    
    # API Settings
    API_PREFIX: str = "/api"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # Default frontend URL
        "http://localhost:8000",  # Default backend URL
    ]
    
    # Database Settings (for future use)
    DATABASE_URL: Optional[str] = None
    
    # API Keys
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    POLYGON_API_KEY: Optional[str] = None
    
    # Model Paths
    MODEL_DIR: Path = Path("saved_models")
    MODEL_PATH: Path = MODEL_DIR / "lstm_cnn_model.h5"
    SCALER_PATH: Path = MODEL_DIR / "scaler.pkl"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Validation for CORS origins
    @field_validator("BACKEND_CORS_ORIGINS", mode='before')
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Ensure the model directory exists
    @field_validator("MODEL_DIR", mode='before')
    @classmethod
    def ensure_model_dir_exists(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    # Configuration for environment variables
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

# Create settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the application settings."""
    return settings
