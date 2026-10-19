from datetime import datetime
from pathlib import Path
import re

import pandas as pd
from sqlalchemy import or_
from sqlalchemy.orm import Session

from models.incident import Incident
from services.prediction_service import prediction_service

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "reference" / "service_catalog.csv"
HISTORICAL_PATH = PROJECT_ROOT / "data" / "raw" / "incidents.csv"


def load_catalog() -> pd.DataFrame:
    return pd.read_csv(CATALOG_PATH)


def recommend_support(service: str, category: str | None = None) -> tuple[str | None, str | None]:
    catalog = load_catalog()
    matches = catalog[catalog["service_name"].astype(str).str.lower() == service.lower()]
    if matches.empty and category:
        matches = catalog[catalog["category_name"].astype(str).str.lower() == category.lower()]
    if matches.empty:
        return category, None
    row = matches.iloc[0]
    return str(row["category_name"]), str(row["assignment_group_name"])


def next_incident_id(db: Session) -> str:
    latest = db.query(Incident).order_by(Incident.id.desc()).first()
    number = 1 if latest is None else latest.id + 1
    return f"INC-{number:06d}"


def create_incident(db: Session, payload):
    data = payload.model_dump()
    recommended_category, assignment_group = recommend_support(data["service"], data["category"])
    data["category"] = recommended_category or data["category"]

    prediction = prediction_service.predict(data)
    incident = Incident(
        incident_id=next_incident_id(db),
        **data,
        assignment_group=assignment_group,
        predicted_priority=prediction["predicted_priority"],
        prediction_confidence=prediction["confidence"],
        status="NEW",
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    return incident


def get_incident(db: Session, incident_id: str):
    return db.query(Incident).filter(Incident.incident_id == incident_id).first()


def list_incidents(
    db: Session,
    status: str | None = None,
    priority: str | None = None,
    category: str | None = None,
    search: str | None = None,
    skip: int = 0,
    limit: int = 100,
):
    query = db.query(Incident)
    if status:
        query = query.filter(Incident.status == status)
    if priority:
        query = query.filter(
            or_(Incident.final_priority == priority, Incident.predicted_priority == priority)
        )
    if category:
        query = query.filter(Incident.category == category)
    if search:
        term = f"%{search}%"
        query = query.filter(
            or_(
                Incident.incident_id.ilike(term),
                Incident.title.ilike(term),
                Incident.description.ilike(term),
                Incident.service.ilike(term),
            )
        )
    total = query.count()
    items = query.order_by(Incident.created_date.desc()).offset(skip).limit(limit).all()
    return items, total


def update_incident(db: Session, incident_id: str, payload):
    incident = get_incident(db, incident_id)
    if incident is None:
        return None

    changes = payload.model_dump(exclude_unset=True)
    for key, value in changes.items():
        setattr(incident, key, value)

    # Re-run prediction when prediction-driving fields change.
    prediction_fields = {
        "category",
        "service",
        "impact",
        "urgency",
        "affected_users",
        "service_criticality",
        "reported_source",
    }
    if prediction_fields.intersection(changes):
        data = {
            "category": incident.category,
            "service": incident.service,
            "impact": incident.impact,
            "urgency": incident.urgency,
            "affected_users": incident.affected_users,
            "service_criticality": incident.service_criticality,
            "reported_source": incident.reported_source,
        }
        prediction = prediction_service.predict(data)
        incident.predicted_priority = prediction["predicted_priority"]
        incident.prediction_confidence = prediction["confidence"]

    db.commit()
    db.refresh(incident)
    return incident


def close_incident(db: Session, incident_id: str):
    incident = get_incident(db, incident_id)
    if incident is None:
        return None
    incident.status = "CLOSED"
    if incident.resolution_time_hours is None:
        elapsed = datetime.utcnow() - incident.created_date
        incident.resolution_time_hours = round(elapsed.total_seconds() / 3600, 2)
    db.commit()
    db.refresh(incident)
    return incident


def similar_historical_incidents(title: str, description: str, limit: int = 5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    df = pd.read_csv(HISTORICAL_PATH)
    if df.empty:
        return []

    query_text = f"{title} {description}"
    corpus = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus + [query_text])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    top_idx = scores.argsort()[::-1][:limit]

    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        results.append({
            "incident_id": row["incident_id"],
            "title": row["title"],
            "category": row["category"],
            "service": row["service"],
            "priority": row["priority"],
            "status": row["status"],
            "similarity": round(float(scores[idx]), 4),
        })
    return results
