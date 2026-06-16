"""Metrics, health, and dashboard stats router."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.models.database import Account, Alert, AlertStatus, Prediction
from app.models.schemas import (
    AccountSummary,
    DashboardStats,
    FeatureImportanceItem,
    HealthResponse,
    ModelMetrics,
)
from app.services.ml_service import get_ml_service

router = APIRouter(prefix="/api/v1", tags=["metrics"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    ml = get_ml_service()
    return HealthResponse(
        status="healthy" if ml.is_loaded else "degraded",
        model_loaded=ml.is_loaded,
        model_name=ml.model_name if ml.is_loaded else "N/A",
        n_features=ml.n_features if ml.is_loaded else 0,
        version=settings.app_version,
    )


@router.get("/metrics", response_model=ModelMetrics)
async def get_model_metrics() -> ModelMetrics:
    """Return stored model performance metrics."""
    ml = get_ml_service()
    m = ml.model_metrics
    return ModelMetrics(**m)


@router.get("/features", response_model=list[FeatureImportanceItem])
async def get_feature_importance() -> list[FeatureImportanceItem]:
    """Return global feature importance (RF importance + MI scores)."""
    ml = get_ml_service()
    records = ml.get_global_feature_importance()
    return [FeatureImportanceItem(**r) for r in records]


@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DashboardStats:
    """Return aggregated stats for the dashboard."""
    # Total accounts
    total_accounts_r = await db.execute(select(func.count(Account.id)))
    total_accounts = total_accounts_r.scalar_one()

    # Total predictions
    total_preds_r = await db.execute(select(func.count(Prediction.id)))
    total_predictions = total_preds_r.scalar_one()

    # Flagged accounts (SUSPICIOUS label)
    flagged_r = await db.execute(
        select(func.count(func.distinct(Prediction.account_id)))
        .where(Prediction.label == "SUSPICIOUS")
    )
    flagged_accounts = flagged_r.scalar_one()

    # Fraud rate
    fraud_rate = (flagged_accounts / total_accounts * 100) if total_accounts > 0 else 0.0

    # Open alerts
    open_alerts_r = await db.execute(
        select(func.count(Alert.id)).where(Alert.status == AlertStatus.OPEN)
    )
    open_alerts = open_alerts_r.scalar_one()

    # Critical alerts
    critical_r = await db.execute(
        select(func.count(Alert.id))
        .where(Alert.status == AlertStatus.OPEN, Alert.severity == "CRITICAL")
    )
    critical_alerts = critical_r.scalar_one()

    # Average risk score
    avg_risk_r = await db.execute(select(func.avg(Prediction.risk_score)))
    avg_risk = float(avg_risk_r.scalar_one() or 0.0)

    # Recent predictions (last 10)
    recent_r = await db.execute(
        select(Account)
        .order_by(Account.uploaded_at.desc())
        .limit(10)
    )
    recent_accounts = recent_r.scalars().all()

    recent_summaries = []
    for acc in recent_accounts:
        pred_r = await db.execute(
            select(Prediction)
            .where(Prediction.account_id == acc.id)
            .order_by(Prediction.predicted_at.desc())
            .limit(1)
        )
        latest_pred = pred_r.scalar_one_or_none()

        alert_r = None
        if latest_pred:
            ar = await db.execute(
                select(Alert).where(Alert.prediction_id == latest_pred.id)
            )
            alert_r = ar.scalar_one_or_none()

        recent_summaries.append(
            AccountSummary(
                id=acc.id,
                account_ref=acc.account_ref,
                uploaded_at=acc.uploaded_at,
                latest_risk_score=latest_pred.risk_score if latest_pred else None,
                latest_label=latest_pred.label if latest_pred else None,
                latest_probability=latest_pred.probability if latest_pred else None,
                alert_status=alert_r.status.value if alert_r else None,
            )
        )

    return DashboardStats(
        total_accounts=total_accounts,
        total_predictions=total_predictions,
        flagged_accounts=flagged_accounts,
        fraud_rate=round(fraud_rate, 2),
        open_alerts=open_alerts,
        critical_alerts=critical_alerts,
        avg_risk_score=round(avg_risk, 1),
        recent_predictions=recent_summaries,
    )
