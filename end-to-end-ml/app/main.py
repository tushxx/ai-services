from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.analytics import router as analytics_router
from api.incidents import router as incidents_router
from api.predictions import router as predictions_router
from database.database import init_db

app = FastAPI(
    title="Intelligent IT Incident Triage & Resolution Assistant",
    version="1.0.0",
    description="IT incident management, ML priority prediction, historical lookup and analytics.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(incidents_router)
app.include_router(predictions_router)
app.include_router(analytics_router)

# Create the SQLite schema when the module is imported as well as on application startup.
# This also makes direct TestClient usage deterministic.
init_db()


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}
