"""Pydantic v2 schemas for MuleShield AI API."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Feature Schemas ──────────────────────────────────────────────────────────

class FeatureImportanceItem(BaseModel):
    feature: str
    rf_importance: float
    mi_score: float
    shap_importance: Optional[float] = None


# ─── SHAP Explanation ─────────────────────────────────────────────────────────

class ShapFeature(BaseModel):
    name: str
    value: float
    shap_impact: float   # positive = pushes toward fraud, negative = toward legit


# ─── Prediction Schemas ───────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Dictionary of feature_name → value. Must contain the 50 selected features.",
        example={"F3898": 0.5, "F3805": 1.2}
    )
    account_ref: Optional[str] = Field(None, description="Optional human-readable account reference")


class PredictionResponse(BaseModel):
    prediction_id: uuid.UUID
    account_id: uuid.UUID
    account_ref: Optional[str]
    label: str                        # LEGITIMATE | SUSPICIOUS
    probability: float
    risk_score: int                   # 0–100
    risk_tier: str                    # LOW | MEDIUM | HIGH | CRITICAL
    threshold_used: float
    top_features: list[ShapFeature]
    predicted_at: datetime
    alert_id: Optional[uuid.UUID] = None


class BatchPredictionStatus(BaseModel):
    job_id: uuid.UUID
    filename: str
    status: str
    total_records: int
    processed_count: int
    flagged_count: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


# ─── Account Schemas ──────────────────────────────────────────────────────────

class AccountSummary(BaseModel):
    id: uuid.UUID
    account_ref: Optional[str]
    uploaded_at: datetime
    latest_risk_score: Optional[int] = None
    latest_label: Optional[str] = None
    latest_probability: Optional[float] = None
    alert_status: Optional[str] = None

    model_config = {"from_attributes": True}


class AccountDetail(BaseModel):
    id: uuid.UUID
    account_ref: Optional[str]
    features: dict[str, Any]
    uploaded_at: datetime
    predictions: list[PredictionResponse] = []

    model_config = {"from_attributes": True}


# ─── Alert Schemas ────────────────────────────────────────────────────────────

class AlertResponse(BaseModel):
    id: uuid.UUID
    prediction_id: uuid.UUID
    account_id: Optional[uuid.UUID] = None
    account_ref: Optional[str] = None
    severity: str
    status: str
    risk_score: Optional[int] = None
    probability: Optional[float] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AlertUpdateRequest(BaseModel):
    status: str = Field(..., pattern="^(OPEN|REVIEWED|CLOSED)$")
    notes: Optional[str] = None


# ─── Metrics Schemas ──────────────────────────────────────────────────────────

class ModelMetrics(BaseModel):
    model_name: str
    pr_auc: float
    roc_auc: float
    f1_score: float
    precision: float
    recall: float
    optimal_threshold: float
    n_features: int


class DashboardStats(BaseModel):
    total_accounts: int
    total_predictions: int
    flagged_accounts: int
    fraud_rate: float              # percentage
    open_alerts: int
    critical_alerts: int
    avg_risk_score: float
    recent_predictions: list[AccountSummary]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    n_features: int
    version: str


# ─── Paginated Response ───────────────────────────────────────────────────────

class PaginatedAccounts(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AccountSummary]
