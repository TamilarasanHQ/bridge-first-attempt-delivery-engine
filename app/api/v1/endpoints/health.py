"""
Health and metadata endpoint.
"""
from fastapi import APIRouter
from app.core.config import settings
from app.core.engine import engine

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_loaded": engine.model is not None,
        "feature_count": len(engine.model.feature_names_in_) if engine.model else 0,
        "features": list(engine.model.feature_names_in_) if engine.model else [],
        "risk_thresholds": {
            "low": settings.LOW_RISK_THRESHOLD,
            "high": settings.HIGH_RISK_THRESHOLD
        }
    }
