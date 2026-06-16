"""
MuleShield AI — FastAPI Application Entry Point.

Bank Fraud & Mule Account Detection Platform.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.database import create_tables
from app.routers import alerts, accounts, metrics, predictions
from app.services.ml_service import get_ml_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    # ── Startup ──────────────────────────────────────────────────
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Create database tables
    try:
        await create_tables()
        logger.info("Database tables ensured ✓")
    except Exception as e:
        logger.error(f"Database setup failed: {e}")

    # Load ML model
    ml = get_ml_service()
    try:
        ml.load()
        logger.info(f"Model loaded: {ml.model_name} | {ml.n_features} features | threshold={ml.optimal_threshold}")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")

    yield

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("Shutting down MuleShield AI...")


app = FastAPI(
    title="MuleShield AI",
    description=(
        "Bank Fraud & Mule Account Detection Platform. "
        "Powered by CatBoost with SHAP explainability."
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predictions.router)
app.include_router(accounts.router)
app.include_router(alerts.router)
app.include_router(metrics.router)


@app.get("/", tags=["root"])
async def root():
    return JSONResponse({
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    })
