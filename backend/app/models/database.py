"""SQLAlchemy ORM models for MuleShield AI."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    DateTime, ForeignKey, Text, Enum as SAEnum, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum

from app.core.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class AlertSeverity(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertStatus(str, enum.Enum):
    OPEN = "OPEN"
    REVIEWED = "REVIEWED"
    CLOSED = "CLOSED"


class BatchJobStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    total_records = Column(Integer, default=0)
    flagged_count = Column(Integer, default=0)
    processed_count = Column(Integer, default=0)
    status = Column(SAEnum(BatchJobStatus), default=BatchJobStatus.PENDING, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    predictions = relationship("Prediction", back_populates="batch_job")


class Account(Base):
    __tablename__ = "accounts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_ref = Column(String(100), nullable=True, index=True)  # user-supplied identifier
    features = Column(JSON, nullable=False)  # dict of feature_name -> value
    uploaded_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    batch_job_id = Column(UUID(as_uuid=True), ForeignKey("batch_jobs.id"), nullable=True)

    predictions = relationship("Prediction", back_populates="account", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_id = Column(UUID(as_uuid=True), ForeignKey("accounts.id"), nullable=False)
    batch_job_id = Column(UUID(as_uuid=True), ForeignKey("batch_jobs.id"), nullable=True)

    probability = Column(Float, nullable=False)
    risk_score = Column(Integer, nullable=False)  # 0-100
    label = Column(String(20), nullable=False)   # LEGITIMATE | SUSPICIOUS
    threshold_used = Column(Float, nullable=False)
    top_features = Column(JSON, nullable=True)   # list of {name, value, shap_impact}
    predicted_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    account = relationship("Account", back_populates="predictions")
    batch_job = relationship("BatchJob", back_populates="predictions")
    alert = relationship("Alert", back_populates="prediction", uselist=False, cascade="all, delete-orphan")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id"), nullable=False, unique=True)

    severity = Column(SAEnum(AlertSeverity), nullable=False)
    status = Column(SAEnum(AlertStatus), default=AlertStatus.OPEN, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    prediction = relationship("Prediction", back_populates="alert")
