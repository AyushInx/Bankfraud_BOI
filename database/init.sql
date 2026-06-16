-- MuleShield AI - Database Initialization
-- PostgreSQL 16

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Enum types ──────────────────────────────────────────────────────────────

DO $$ BEGIN
    CREATE TYPE alert_severity AS ENUM ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE alert_status AS ENUM ('OPEN', 'REVIEWED', 'CLOSED');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE batch_job_status AS ENUM ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ── Batch Jobs ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS batch_jobs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename        VARCHAR(255) NOT NULL,
    total_records   INTEGER DEFAULT 0,
    flagged_count   INTEGER DEFAULT 0,
    processed_count INTEGER DEFAULT 0,
    status          batch_job_status NOT NULL DEFAULT 'PENDING',
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_created_at ON batch_jobs(created_at DESC);

-- ── Accounts ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS accounts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_ref     VARCHAR(100),
    features        JSONB NOT NULL,
    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    batch_job_id    UUID REFERENCES batch_jobs(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_accounts_account_ref ON accounts(account_ref);
CREATE INDEX IF NOT EXISTS idx_accounts_uploaded_at ON accounts(uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_accounts_batch_job_id ON accounts(batch_job_id);

-- ── Predictions ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS predictions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id      UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    batch_job_id    UUID REFERENCES batch_jobs(id) ON DELETE SET NULL,
    probability     FLOAT NOT NULL,
    risk_score      INTEGER NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    label           VARCHAR(20) NOT NULL CHECK (label IN ('LEGITIMATE', 'SUSPICIOUS')),
    threshold_used  FLOAT NOT NULL,
    top_features    JSONB,
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_account_id ON predictions(account_id);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_score ON predictions(risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions(label);
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at ON predictions(predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_batch_job_id ON predictions(batch_job_id);

-- ── Alerts ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS alerts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id   UUID NOT NULL UNIQUE REFERENCES predictions(id) ON DELETE CASCADE,
    severity        alert_severity NOT NULL,
    status          alert_status NOT NULL DEFAULT 'OPEN',
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at DESC);

-- Auto-update updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_alerts_updated_at ON alerts;
CREATE TRIGGER set_alerts_updated_at
BEFORE UPDATE ON alerts
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();
