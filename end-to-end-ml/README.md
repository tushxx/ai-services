# Intelligent IT Incident Triage & Resolution Assistant

A complete IT-operations engineering project based on the supplied project specification and the supplied 750-record incident dataset.

## What it does

The application helps an engineer answer four questions quickly:

1. **What happened?** — incident information and description.
2. **How important is it?** — ML prediction of priority P1–P4.
3. **Where should it go?** — recommended support category/team from the service catalog.
4. **What should I look at?** — similar historical incidents.

The architecture follows the specification:

```text
Streamlit
   ↓
FastAPI
   ↓
Service
   ↓
SQLAlchemy
   ↓
SQLite

FastAPI
   ↓
Prediction Service
   ↓
Preprocessing
   ↓
Trained Decision Tree
```

The supplied dataset contains 750 historical incidents and reference data for categories, services, assignment groups, statuses, impacts, urgencies, criticality, priorities, and priority rules.

## Project structure

```text
it-incident-intelligence/
├── app/
│   ├── main.py
│   └── api/
│       ├── incidents.py
│       ├── predictions.py
│       └── analytics.py
├── models/
│   ├── incident.py
│   ├── schemas.py
│   └── artifacts/
├── services/
│   ├── incident_service.py
│   └── prediction_service.py
├── database/
│   └── database.py
├── ml/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── dashboard/
│   └── app.py
├── tests/
│   ├── test_incidents.py
│   └── test_predictions.py
├── data/
│   ├── raw/incidents.csv
│   ├── processed/incidents_ml_dataset.csv
│   └── reference/*.csv
├── requirements.txt
└── README.md
```

## Setup

### 1. Create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python -m ml.train
```

This creates:

- `models/artifacts/incident_priority_model.joblib`
- `models/artifacts/model_metrics.json`
- `models/artifacts/feature_importance.csv`
- `models/artifacts/confusion_matrix.csv`

The training pipeline uses Pandas, one-hot categorical encoding, a train/test split, a Decision Tree classifier, and accuracy/precision/recall/F1/confusion-matrix evaluation.

### 4. Optional: load the supplied 750 historical incidents into SQLite

The historical CSV remains the source for similarity search, but you can also populate SQLite so the dashboard immediately shows operational data:

```bash
python scripts/seed_database.py
```

This is idempotent for existing incident IDs.

### 5. Start FastAPI

```bash
uvicorn app.main:app --reload
```

Open:

- API: http://localhost:8000
- Swagger: http://localhost:8000/docs

### 6. Start Streamlit

In another terminal:

```bash
streamlit run dashboard/app.py
```

The dashboard provides:

- Operations Dashboard
- Create Incident
- Incident Intelligence
- Incident Search
- Incident Detail / update / close

## API endpoints

The supplied specification calls for:

- `POST /api/v1/incidents`
- `GET /api/v1/incidents`
- `GET /api/v1/incidents/{incident_id}`
- `PUT /api/v1/incidents/{incident_id}`
- `PATCH /api/v1/incidents/{incident_id}/close`
- `POST /api/v1/predictions/priority`
- `GET /api/v1/analytics/summary`

## Example prediction request

```json
{
  "category": "Network",
  "service": "Corporate VPN",
  "impact": "High",
  "urgency": "High",
  "affected_users": 180,
  "service_criticality": "Business_Critical",
  "reported_source": "Service Portal"
}
```

The response includes the predicted priority, confidence, and a probability distribution.

## Design notes

- The model predicts **incident priority** with four classes: P1, P2, P3 and P4.
- `predicted_priority` is stored separately from `final_priority`, so the application preserves the distinction between ML recommendation and engineer decision.
- Support team/category is determined from the supplied service catalog rather than invented by the model.
- Similar incidents use TF-IDF/cosine similarity against historical title + description text.
- Engineer judgment remains authoritative: the ML output is a recommendation, not an automatic final decision.
