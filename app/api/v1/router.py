"""
API v1 Router aggregation.
"""
from fastapi import APIRouter
from app.api.v1.endpoints import health, predict, preferences

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health & Metadata"])
api_router.include_router(predict.router, tags=["Risk Prediction & Manifest Scoring"])
api_router.include_router(preferences.router, tags=["Customer Preferences & Interventions"])
