"""
Prediction router — single and batch prediction endpoints.
"""
from __future__ import annotations

import asyncio
import io
import uuid
from datetime import datetime, timezone
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.database import Account, Alert, BatchJob, BatchJobStatus, Prediction
from app.models.schemas import (
    BatchPredictionStatus,
    PredictionRequest,
    PredictionResponse,
    ShapFeature,
)
from app.services.ml_service import get_ml_service
from app.services.risk_service import risk_tier_to_alert_severity, should_create_alert

router = APIRouter(prefix="/api/v1", tags=["predictions"])


def _build_prediction_response(
    prediction: Prediction,
    account: Account,
    alert: Alert | None,
) -> PredictionResponse:
    top_features = [ShapFeature(**f) for f in (prediction.top_features or [])]
    return PredictionResponse(
        prediction_id=prediction.id,
        account_id=account.id,
        account_ref=account.account_ref,
        label=prediction.label,
        probability=prediction.probability,
        risk_score=prediction.risk_score,
        risk_tier=prediction.top_features[0].get("risk_tier", "LOW") if prediction.top_features else "LOW",
        threshold_used=prediction.threshold_used,
        top_features=top_features,
        predicted_at=prediction.predicted_at,
        alert_id=alert.id if alert else None,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PredictionResponse:
    """
    Run fraud prediction on a single account record.
    Saves the account, prediction, and alert (if high-risk) to the database.
    """
    ml = get_ml_service()
    if not ml.is_loaded:
        raise HTTPException(status_code=503, detail="ML model not ready")

    # Run inference
    result = ml.predict_single(request.features)

    # Save account
    account = Account(
        account_ref=request.account_ref,
        features=request.features,
    )
    db.add(account)
    await db.flush()

    # Save prediction
    prediction = Prediction(
        account_id=account.id,
        probability=result["probability"],
        risk_score=result["risk_score"],
        label=result["label"],
        threshold_used=result["threshold_used"],
        top_features=result["top_features"],
    )
    db.add(prediction)
    await db.flush()

    # Create alert if warranted
    alert = None
    if should_create_alert(result["risk_tier"]):
        severity = risk_tier_to_alert_severity(result["risk_tier"])
        alert = Alert(
            prediction_id=prediction.id,
            severity=severity,
        )
        db.add(alert)
        await db.flush()

    top_features = [ShapFeature(**f) for f in result["top_features"]]
    return PredictionResponse(
        prediction_id=prediction.id,
        account_id=account.id,
        account_ref=request.account_ref,
        label=result["label"],
        probability=result["probability"],
        risk_score=result["risk_score"],
        risk_tier=result["risk_tier"],
        threshold_used=result["threshold_used"],
        top_features=top_features,
        predicted_at=prediction.predicted_at,
        alert_id=alert.id if alert else None,
    )


@router.post("/batch-predict", response_model=BatchPredictionStatus)
async def batch_predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> BatchPredictionStatus:
    """
    Upload a CSV file for batch fraud prediction.
    Returns a job_id immediately; processing runs in the background.
    Poll /batch/{job_id} for status.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    # Create job record
    job = BatchJob(
        filename=file.filename,
        total_records=len(df),
        status=BatchJobStatus.PENDING,
    )
    db.add(job)
    await db.flush()
    job_id = job.id

    background_tasks.add_task(_process_batch, job_id, df, content)

    return BatchPredictionStatus(
        job_id=job_id,
        filename=file.filename,
        status="PENDING",
        total_records=len(df),
        processed_count=0,
        flagged_count=0,
        created_at=job.created_at,
    )


async def _process_batch(job_id: uuid.UUID, df: pd.DataFrame, raw_content: bytes) -> None:
    """Background task: run batch inference and save results."""
    from app.core.database import AsyncSessionLocal

    ml = get_ml_service()

    async with AsyncSessionLocal() as db:
        try:
            # Update job to processing
            result = await db.execute(select(BatchJob).where(BatchJob.id == job_id))
            job = result.scalar_one()
            job.status = BatchJobStatus.PROCESSING
            await db.commit()

            # Run batch inference
            predictions = ml.predict_batch(df)
            flagged = 0

            for i, pred in enumerate(predictions):
                row_features = {col: float(df.iloc[i][col]) for col in df.columns if col in ml._feature_names}
                account_ref = str(df.iloc[i].get("account_id", f"row_{i}")) if "account_id" in df.columns else f"row_{i}"

                account = Account(
                    account_ref=account_ref,
                    features=row_features,
                    batch_job_id=job_id,
                )
                db.add(account)
                await db.flush()

                prediction_rec = Prediction(
                    account_id=account.id,
                    batch_job_id=job_id,
                    probability=pred["probability"],
                    risk_score=pred["risk_score"],
                    label=pred["label"],
                    threshold_used=pred["threshold_used"],
                    top_features=pred["top_features"],
                )
                db.add(prediction_rec)
                await db.flush()

                if should_create_alert(pred["risk_tier"]):
                    alert = Alert(
                        prediction_id=prediction_rec.id,
                        severity=risk_tier_to_alert_severity(pred["risk_tier"]),
                    )
                    db.add(alert)
                    flagged += 1

                # Update progress every 50 records
                if (i + 1) % 50 == 0:
                    job.processed_count = i + 1
                    job.flagged_count = flagged
                    await db.commit()

            job.status = BatchJobStatus.COMPLETED
            job.processed_count = len(predictions)
            job.flagged_count = flagged
            job.completed_at = datetime.now(timezone.utc)
            await db.commit()

        except Exception as e:
            job.status = BatchJobStatus.FAILED
            job.error_message = str(e)
            await db.commit()
            raise


@router.get("/batch/{job_id}", response_model=BatchPredictionStatus)
async def get_batch_status(
    job_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BatchPredictionStatus:
    """Poll batch job status."""
    result = await db.execute(select(BatchJob).where(BatchJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Batch job not found")

    return BatchPredictionStatus(
        job_id=job.id,
        filename=job.filename,
        status=job.status.value,
        total_records=job.total_records,
        processed_count=job.processed_count,
        flagged_count=job.flagged_count,
        created_at=job.created_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
    )
