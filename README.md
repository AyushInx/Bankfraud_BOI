# 🛡️ MuleShield AI — Bank Fraud & Mule Account Detection Platform

A production-style full-stack AI-powered fraud investigation platform for banking data.
Built on top of a **CatBoost** model (ROC-AUC: 0.994 | PR-AUC: 0.755) with **SHAP** explainability.

---

## 🏗️ Project Structure

```
muleshield/
├── backend/               # FastAPI backend
│   ├── app/
│   │   ├── core/          # Config, DB engine
│   │   ├── models/        # SQLAlchemy ORM + Pydantic schemas
│   │   ├── routers/       # API endpoints
│   │   └── services/      # ML inference + risk scoring
│   ├── ml_artifacts/      # CatBoost model + feature lists
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/              # React + Vite frontend
│   ├── src/
│   │   ├── components/    # Navbar, RiskGauge, ShapChart
│   │   ├── pages/         # Home, Upload, Investigate, Alerts
│   │   └── hooks/         # API hooks (useApi.js)
│   └── Dockerfile
├── database/
│   └── init.sql           # PostgreSQL schema
└── docker-compose.yml
```

---

## 🚀 Quick Start

### Option A — Docker (recommended, no local setup)

```bash
cd c:\ML\BOI\muleshield
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option B — Local Development (no Docker)

**Backend:**
```bash
# 1. Install Python deps (use your existing venv or create new)
cd c:\ML\BOI\muleshield\backend
pip install -r requirements.txt

# 2. Set DB URL (use a local PostgreSQL, or skip for SQLite workaround)
set DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/muleshield

# 3. Run
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd c:\ML\BOI\muleshield\frontend
npm install         # already done
npm run dev         # http://localhost:3000
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check + model status |
| GET | `/api/v1/metrics` | Model performance (PR-AUC, ROC-AUC…) |
| GET | `/api/v1/features` | Global feature importance |
| GET | `/api/v1/stats` | Dashboard KPI stats |
| POST | `/api/v1/predict` | Single account fraud prediction |
| POST | `/api/v1/batch-predict` | CSV batch upload |
| GET | `/api/v1/batch/{job_id}` | Batch job status poll |
| GET | `/api/v1/accounts` | List accounts (paginated) |
| GET | `/api/v1/accounts/{id}` | Account detail + prediction history |
| GET | `/api/v1/alerts` | List alerts (with filters) |
| PATCH | `/api/v1/alerts/{id}/status` | Update alert status |

Full interactive docs: http://localhost:8000/docs

---

## 🤖 ML Model

| Metric | Value |
|--------|-------|
| Model | CatBoost (best of XGBoost / LightGBM / CatBoost / LR) |
| ROC-AUC | **0.994** |
| PR-AUC | **0.755** |
| Precision | 1.0 |
| Recall | 0.55 |
| F1 | 0.71 |
| Threshold | 0.933 |
| Features | 50 selected (RF importance + MI scores) |
| Imbalance | `auto_class_weights='Balanced'` |

---

## 🎨 UI Pages

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/` | KPI cards, model metrics, recent predictions |
| Batch Upload | `/upload` | Drag-and-drop CSV, real-time progress |
| Investigate | `/investigate` | Single account form/JSON, risk gauge, SHAP chart |
| Alerts | `/alerts` | Alert management with analyst workflow |

---

## 🔍 Risk Scoring

| Score | Tier | Action |
|-------|------|--------|
| 0–39 | 🟢 LOW | Normal monitoring |
| 40–69 | 🔵 MEDIUM | Elevated review |
| 70–89 | 🟠 HIGH | Immediate review |
| 90–100 | 🔴 CRITICAL | Freeze / escalate |

Score formula: `sqrt(probability) × 100` — non-linear, emphasizes high-probability accounts.

---

## 📦 Environment Variables

Create `backend/.env` for local development:
```env
DATABASE_URL=postgresql+asyncpg://muleshield:muleshield@localhost:5432/muleshield
DEBUG=false
```

Create `frontend/.env`:
```env
VITE_API_URL=http://localhost:8000
```
