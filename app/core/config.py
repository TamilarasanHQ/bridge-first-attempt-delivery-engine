"""
Application Configuration and Settings for BRIDGE Decision Support Service.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings:
    PROJECT_NAME: str = "BRIDGE — AI-Powered First Attempt Delivery Success Engine"
    VERSION: str = "2.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Model Artifacts
    MODEL_PATH: Path = ROOT_DIR / "v2" / "models" / "phase5_candidate_model.joblib"
    
    # Risk Thresholds
    # In v2, base prevalence is 0.75%.
    # Probabilities >= 0.035 correspond to top ~3-5% highest risk deliveries.
    LOW_RISK_THRESHOLD: float = 0.015
    HIGH_RISK_THRESHOLD: float = 0.035
    
    # Static & Templates Directory
    STATIC_DIR: Path = ROOT_DIR / "app" / "static"


settings = Settings()
