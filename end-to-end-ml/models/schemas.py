from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    category: str
    service: str
    impact: str
    urgency: str
    affected_users: int = Field(ge=0)
    service_criticality: str
    reported_source: str


class PredictionResponse(BaseModel):
    predicted_priority: str
    confidence: float
    probabilities: dict[str, float]


class IncidentCreate(BaseModel):
    title: str = Field(min_length=3, max_length=255)
    description: str = Field(min_length=3)
    category: str
    service: str
    impact: str
    urgency: str
    affected_users: int = Field(ge=0)
    service_criticality: str
    reported_source: str
    final_priority: Optional[str] = None


class IncidentUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=3, max_length=255)
    description: Optional[str] = None
    impact: Optional[str] = None
    urgency: Optional[str] = None
    affected_users: Optional[int] = Field(default=None, ge=0)
    service_criticality: Optional[str] = None
    reported_source: Optional[str] = None
    status: Optional[str] = None
    final_priority: Optional[str] = None


class IncidentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    incident_id: str
    title: str
    description: str
    category: str
    service: str
    impact: str
    urgency: str
    affected_users: int
    service_criticality: str
    reported_source: str
    assignment_group: Optional[str]
    status: str
    predicted_priority: Optional[str]
    prediction_confidence: Optional[float]
    final_priority: Optional[str]
    created_date: datetime
    updated_date: datetime
    resolution_time_hours: Optional[float]


class IncidentListResponse(BaseModel):
    items: list[IncidentResponse]
    total: int
