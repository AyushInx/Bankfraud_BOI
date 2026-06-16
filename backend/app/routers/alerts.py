"""Alerts router — list and manage fraud alerts."""
from __future__ import annotations

import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.database import Alert, AlertStatus, Prediction, Account
from app.models.schemas import AlertResponse, AlertUpdateRequest

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


def _alert_to_response(alert: Alert) -> AlertResponse:
    pred = alert.prediction
    account = pred.account if pred else None
    return AlertResponse(
        id=alert.id,
        prediction_id=alert.prediction_id,
        account_id=account.id if account else None,
        account_ref=account.account_ref if account else None,
        severity=alert.severity.value,
        status=alert.status.value,
        risk_score=pred.risk_score if pred else None,
        probability=pred.probability if pred else None,
        notes=alert.notes,
        created_at=alert.created_at,
        updated_at=alert.updated_at,
    )


@router.get("", response_model=list[AlertResponse])
async def list_alerts(
    db: Annotated[AsyncSession, Depends(get_db)],
    status: Optional[str] = Query(None, description="Filter: OPEN, REVIEWED, CLOSED"),
    severity: Optional[str] = Query(None, description="Filter: LOW, MEDIUM, HIGH, CRITICAL"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[AlertResponse]:
    """List all fraud alerts with optional filters."""
    q = (
        select(Alert)
        .options(
            selectinload(Alert.prediction).selectinload(Prediction.account)
        )
        .order_by(Alert.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    if status:
        try:
            q = q.where(Alert.status == AlertStatus[status.upper()])
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    if severity:
        q = q.where(Alert.severity == severity.upper())

    result = await db.execute(q)
    alerts = result.scalars().all()
    return [_alert_to_response(a) for a in alerts]


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AlertResponse:
    """Get a specific alert by ID."""
    result = await db.execute(
        select(Alert)
        .where(Alert.id == alert_id)
        .options(
            selectinload(Alert.prediction).selectinload(Prediction.account)
        )
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return _alert_to_response(alert)


@router.patch("/{alert_id}/status", response_model=AlertResponse)
async def update_alert_status(
    alert_id: uuid.UUID,
    update: AlertUpdateRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AlertResponse:
    """Update alert status (OPEN → REVIEWED → CLOSED) and add analyst notes."""
    result = await db.execute(
        select(Alert)
        .where(Alert.id == alert_id)
        .options(
            selectinload(Alert.prediction).selectinload(Prediction.account)
        )
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    try:
        alert.status = AlertStatus[update.status.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {update.status}")

    if update.notes is not None:
        alert.notes = update.notes

    await db.commit()
    await db.refresh(alert)
    return _alert_to_response(alert)
