from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from database.database import Base


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    incident_id: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    category: Mapped[str] = mapped_column(String(80), nullable=False)
    service: Mapped[str] = mapped_column(String(120), nullable=False)
    impact: Mapped[str] = mapped_column(String(30), nullable=False)
    urgency: Mapped[str] = mapped_column(String(30), nullable=False)
    affected_users: Mapped[int] = mapped_column(Integer, nullable=False)
    service_criticality: Mapped[str] = mapped_column(String(50), nullable=False)
    reported_source: Mapped[str] = mapped_column(String(80), nullable=False)

    assignment_group: Mapped[str | None] = mapped_column(String(120), nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="NEW")

    predicted_priority: Mapped[str | None] = mapped_column(String(10), nullable=True)
    prediction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_priority: Mapped[str | None] = mapped_column(String(10), nullable=True)

    created_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_date: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    resolution_time_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
