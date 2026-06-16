"""Accounts router — query investigated accounts and their history."""
from __future__ import annotations

import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.database import Account, Alert, Prediction
from app.models.schemas import AccountDetail, AccountSummary, PaginatedAccounts, ShapFeature, PredictionResponse

router = APIRouter(prefix="/api/v1/accounts", tags=["accounts"])


async def _account_to_summary(account: Account, db: AsyncSession) -> AccountSummary:
    """Build AccountSummary from Account ORM + latest prediction."""
    # Get latest prediction
    pred_result = await db.execute(
        select(Prediction)
        .where(Prediction.account_id == account.id)
        .order_by(Prediction.predicted_at.desc())
        .limit(1)
        .options(selectinload(Prediction.alert))
    )
    latest = pred_result.scalar_one_or_none()

    return AccountSummary(
        id=account.id,
        account_ref=account.account_ref,
        uploaded_at=account.uploaded_at,
        latest_risk_score=latest.risk_score if latest else None,
        latest_label=latest.label if latest else None,
        latest_probability=latest.probability if latest else None,
        alert_status=latest.alert.status.value if (latest and latest.alert) else None,
    )


@router.get("", response_model=PaginatedAccounts)
async def list_accounts(
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    label: Optional[str] = Query(None, description="Filter by label: SUSPICIOUS or LEGITIMATE"),
) -> PaginatedAccounts:
    """List all investigated accounts with pagination."""
    offset = (page - 1) * page_size

    # Build base query
    count_q = select(func.count(Account.id))
    items_q = select(Account).order_by(Account.uploaded_at.desc()).offset(offset).limit(page_size)

    if label:
        # Filter by latest prediction label via subquery
        latest_pred_sq = (
            select(Prediction.account_id, Prediction.label)
            .distinct(Prediction.account_id)
            .order_by(Prediction.account_id, Prediction.predicted_at.desc())
            .subquery()
        )
        items_q = items_q.join(latest_pred_sq, Account.id == latest_pred_sq.c.account_id).where(
            latest_pred_sq.c.label == label.upper()
        )
        count_q = count_q.join(latest_pred_sq, Account.id == latest_pred_sq.c.account_id).where(
            latest_pred_sq.c.label == label.upper()
        )

    total_result = await db.execute(count_q)
    total = total_result.scalar_one()

    accounts_result = await db.execute(items_q)
    accounts = accounts_result.scalars().all()

    summaries = []
    for acc in accounts:
        summaries.append(await _account_to_summary(acc, db))

    return PaginatedAccounts(total=total, page=page, page_size=page_size, items=summaries)


@router.get("/flagged", response_model=PaginatedAccounts)
async def list_flagged_accounts(
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    min_risk_score: int = Query(70, ge=0, le=100),
) -> PaginatedAccounts:
    """List only high-risk/flagged accounts."""
    offset = (page - 1) * page_size

    # Get accounts with latest prediction above threshold
    subq = (
        select(Prediction.account_id, func.max(Prediction.predicted_at).label("latest"))
        .group_by(Prediction.account_id)
        .subquery()
    )
    latest_pred_sq = (
        select(Prediction)
        .join(subq, (Prediction.account_id == subq.c.account_id) & (Prediction.predicted_at == subq.c.latest))
        .where(Prediction.risk_score >= min_risk_score)
        .subquery()
    )

    total_result = await db.execute(
        select(func.count(Account.id)).join(latest_pred_sq, Account.id == latest_pred_sq.c.account_id)
    )
    total = total_result.scalar_one()

    accounts_result = await db.execute(
        select(Account)
        .join(latest_pred_sq, Account.id == latest_pred_sq.c.account_id)
        .order_by(Account.uploaded_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    accounts = accounts_result.scalars().all()

    summaries = []
    for acc in accounts:
        summaries.append(await _account_to_summary(acc, db))

    return PaginatedAccounts(total=total, page=page, page_size=page_size, items=summaries)


@router.get("/{account_id}", response_model=AccountDetail)
async def get_account(
    account_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AccountDetail:
    """Get full account details including all prediction history."""
    result = await db.execute(
        select(Account)
        .where(Account.id == account_id)
        .options(
            selectinload(Account.predictions).selectinload(Prediction.alert)
        )
    )
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    predictions = []
    for pred in sorted(account.predictions, key=lambda p: p.predicted_at, reverse=True):
        top_features = [ShapFeature(**f) for f in (pred.top_features or [])]
        predictions.append(
            PredictionResponse(
                prediction_id=pred.id,
                account_id=account.id,
                account_ref=account.account_ref,
                label=pred.label,
                probability=pred.probability,
                risk_score=pred.risk_score,
                risk_tier="HIGH" if pred.risk_score >= 70 else ("MEDIUM" if pred.risk_score >= 40 else "LOW"),
                threshold_used=pred.threshold_used,
                top_features=top_features,
                predicted_at=pred.predicted_at,
                alert_id=pred.alert.id if pred.alert else None,
            )
        )

    return AccountDetail(
        id=account.id,
        account_ref=account.account_ref,
        features=account.features,
        uploaded_at=account.uploaded_at,
        predictions=predictions,
    )
