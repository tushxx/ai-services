from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from database.database import get_db
from models.schemas import IncidentCreate, IncidentListResponse, IncidentResponse, IncidentUpdate


router = APIRouter(prefix="/api/v1/incidents", tags=["Incidents"])


@router.post("", response_model=IncidentResponse, status_code=status.HTTP_201_CREATED)
def create(payload: IncidentCreate, db: Session = Depends(get_db)):
    return create_incident(db, payload)


@router.get("", response_model=IncidentListResponse)
def list_all(
    status_filter: str | None = Query(default=None, alias="status"),
    priority: str | None = None,
    category: str | None = None,
    search: str | None = None,
    skip: int = 0,
    limit: int = Query(default=100, le=500),
    db: Session = Depends(get_db),
):
    items, total = list_incidents(
        db,
        status_filter,
        priority,
        category,
        search,
        skip,
        limit,
    )
    return {"items": items, "total": total}


@router.get("/{incident_id}", response_model=IncidentResponse)
def get_one(incident_id: str, db: Session = Depends(get_db)):
    incident = get_incident(db, incident_id)

    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")

    return incident


@router.get("/{incident_id}/similar")
def similar(incident_id: str, db: Session = Depends(get_db)):
    incident = get_incident(db, incident_id)

    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")

    return {
        "items": similar_historical_incidents(
            incident.title,
            incident.description,
        )
    }


@router.put("/{incident_id}", response_model=IncidentResponse)
def update(
    incident_id: str,
    payload: IncidentUpdate,
    db: Session = Depends(get_db),
):
    incident = update_incident(db, incident_id, payload)

    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")

    return incident


@router.patch("/{incident_id}/close", response_model=IncidentResponse)
def close(incident_id: str, db: Session = Depends(get_db)):
    incident = close_incident(db, incident_id)

    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")

    return incident