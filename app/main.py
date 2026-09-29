"""
Main FastAPI Application Entrypoint for BRIDGE.
AI-Powered First Attempt Delivery Success Engine.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.core.config import settings
from app.api.v1.router import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Operational decision-support engine predicting and mitigating first-attempt last-mile delivery failure prior to dispatch."
)

# CORS configuration for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 Router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# Mount static directory for CSS, JS, and UI assets
static_path = Path(__file__).resolve().parent / "static"
static_path.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serves the Dispatch Manager Dashboard."""
    index_file = static_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "service": settings.PROJECT_NAME,
        "status": "online",
        "docs_url": "/docs",
        "api_prefix": settings.API_V1_PREFIX
    }


@app.get("/portal", include_in_schema=False)
def serve_customer_portal():
    """Serves the Customer Preference Selection Portal."""
    portal_file = static_path / "portal.html"
    if portal_file.exists():
        return FileResponse(portal_file)
    return {"message": "Customer portal template"}
