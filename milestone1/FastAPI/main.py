#main.py
# fastapi -- > is basically use for communication and connection mainly/

from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session
import uvicorn

#loading all the content from other files into this file - 3rd step
from database import engine, Base, get_db
from models import Clinic, Appointment
from schemas import ClinicCreate, ClinicOut, ClinicWithAppointment, AppointmentCreate, AppointmentOut

Base.metadata.create_all(bind = engine)
# used in SQLAlchemy to automatically create all the database tables defined by your ORM models
#If your models are defined in a separate file and your creation step is in another file- main.py,
               # Python does not read the model classes. If it doesn't read them, Base.metadata stays empty.
#Always explicitly import your model classes before calling create_all()

#Base: declarative base class that all your database models inherit from.
#metadata: registry object under the hood that collects all table schemas when you define your model classes.
#create_all(bind=engine): method that loops through the collected tables and executes the creation logic using your specific database connection (engine).

app = FastAPI()

@app.post('/clinics', response_model = ClinicOut, status_code = 201)    # in function signature they have provided output as ClinicOut thats why response_model = ClinicOut
def create_clinic(clinic : ClinicCreate, db : Session = Depends(get_db)): -> ClinicOut
  new_clinic = Clinic(
      name = clinic.name,
      city = clinic.city
  )
  db.add(new_clinic)
  db.commit()
  db.refresh(new_clinic)
  return new_clinic





@app.post('/clinics/{clinic_id}/appointments', response_model = AppointmentOut, status_code = 201)
def add_appointment(clinic_id : int, appointment : AppointmentCreate, db : Session = Depends(get_db)): -> AppointmentOut
  c = db.query(Clinic).filter(Clinic.id == clinic_id).first()
  #WHENEVER WE ARE USING DB.QUERY THEN ALWAYS CHECK FOR HTTPEXCEPTION ALWAYS
  if not c:
    raise HTTPException(status_code = 404, detail = "Clinic not found")

  new_appointment = Appointment(
      patient_name = appointment.patient_name,
      status = appointment.status,
      duration = appointment.duration,
  )
  db.add(new_appointment)
  db.commit()
  db.refresh(new_appointment)
  return new_appointment



@app.get("/clinic/{clinic_id}", response_model = ClinicWithAppointments)
def get_clinic(clinic_id : int, status: Optional[str] = None, sort: Optional[str] = None, db.Session = Depends(get_db))-> ClinicWithAppointments:
    c = db.query(Clinic).filter(Clinic.id == clinic_id).first()
    if c is None:
        raise HTTPException(status_code = 404, details = "Clinic not find")

    query = db.query(Appoinment).filter(Appointment.clinic_id == clinic_id)
    if status:
        query = query.filter(Appointment.status == status)

    if sort:
      if sort == "patient_name" :
        query = query.order_by(Appointment.patient_name.asc())
      if sort == "duration":
        query = query.order_by(Appointment.duration)

    appointments = query.all()

    return ClinicWithAppointments(
        id = c.id,
        name = c.name,
        city = c.city
        appointments = appointments
    )


@app.delete("/clinics/{clinic_id}", status_code = 204)
def delete_clinic(clinic_id : int, db : Session = Depends(get_db)):
  c = db.query(Clinic).filter(Clinic.id == clinic_id).first()
  if not c:
    raise HTTPExcception(status_code = 404, detail = "Clinic not found")

  db.delete(c)
  db.commit()
  return Response(status_code = 204)










