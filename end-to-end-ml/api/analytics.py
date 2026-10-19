from collections import Counter
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from database.database import get_db
from models.incident import Incident

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])


@router.get("/summary")
def summary(db: Session = Depends(get_db)):
    total = db.query(Incident).count()
    open_statuses = ["NEW", "ASSIGNED", "IN_PROGRESS"]
    open_count = db.query(Incident).filter(Incident.status.in_(open_statuses)).count()
    critical_count = db.query(Incident).filter(
        (Incident.final_priority == "P1") | (Incident.predicted_priority == "P1")
    ).count()

    def grouped(column):
        rows = db.query(column, func.count(Incident.id)).group_by(column).all()
        return {str(key): int(value) for key, value in rows if key is not None}

    # Daily trend for the last 30 days.
    
    since = datetime.utcnow() - timedelta(days=30)
    rows = (
        db.query(func.date(Incident.created_date), func.count(Incident.id))
        .filter(Incident.created_date >= since)
        .group_by(func.date(Incident.created_date))
        .order_by(func.date(Incident.created_date))
        .all()
    )
    trend = [{"date": str(day), "count": int(count)} for day, count in rows]

    return {
        "total_incidents": total,
        "open_incidents": open_count,
        "critical_incidents": critical_count,
        "by_priority": grouped(Incident.predicted_priority),
        "by_category": grouped(Incident.category),
        "by_status": grouped(Incident.status),
        "trend": trend,
        "ml_prediction_summary": {
            "predicted": db.query(Incident).filter(Incident.predicted_priority.is_not(None)).count(),
            "engineer_finalized": db.query(Incident).filter(Incident.final_priority.is_not(None)).count(),
        },
    }
