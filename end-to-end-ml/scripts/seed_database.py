from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from database.database import SessionLocal, init_db
from models.incident import Incident

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "incidents.csv"


def seed():
    init_db()
    df = pd.read_csv(RAW_PATH)
    db = SessionLocal()
    try:
        existing = {row.incident_id for row in db.query(Incident.incident_id).all()}
        added = 0
        for row in df.itertuples(index=False):
            if row.incident_id in existing:
                continue
            created = pd.to_datetime(row.created_date).to_pydatetime()
            db.add(
                Incident(
                    incident_id=row.incident_id,
                    title=row.title,
                    description=row.description,
                    category=row.category,
                    service=row.service,
                    impact=row.impact,
                    urgency=row.urgency,
                    affected_users=int(row.affected_users),
                    service_criticality=row.service_criticality,
                    reported_source=row.reported_source,
                    assignment_group=row.assignment_group,
                    status=row.status,
                    predicted_priority=row.priority,
                    final_priority=row.priority,
                    created_date=created,
                    resolution_time_hours=float(row.resolution_time_hours),
                )
            )
            added += 1
        db.commit()
        print(f"Seed complete. Added {added} historical incidents.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
