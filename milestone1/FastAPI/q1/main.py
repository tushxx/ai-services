#main.py

from typing import Optional,Literal
from fastapi import FastAPI, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session
import uvicorn

from database import engine, Base, get_db
from models import Depot, Vehicle
from schemas import DepotCreate, DepotOut, DepotWithVehicle, VehicleCreate, VehicleOut

Base.metadata.create_all(bind = engine)

app = FastAPI()

@app.post('/depots', response_model = DepotOut, status_code = 201)
def create_depot(depot : DepotCreate, db : Session = Depends(get_db)):
  new_depot = Depot(name = depot.name, region = depot.region)
  db.add(new_depot)
  db.commit()
  db.refresh(new_depot)
  return new_depot

@app.post('/depots/{depot_id}/vehicles', response_model = VehicleOut, status_code = 201)
def add_vehicle(depot_id : int, vehicle : VehicleCreate, db : Session = Depends(get_db)) -> VehicleOut:

  v = db.query(Depot).filter(Depot.id == depot_id).first()

  if v is None:
    raise HTTPException(status_code = 404, detail = 'Depot not found')
  new_vehicle = Vehicle(
      plate_number = vehicle.plate_number,
      vehicle_type = vehicle.vehicle_type,
      mileage = vehicle.mileage,
      depot_id = depot_id
  ) 
  db.add(new_vehicle)
  db.commit()
  db.refresh(new_vehicle)
  return new_vehicle

@app.get('/depots/{depot_id}', response_model = DepotWithVehicle)
def get_depot(depot_id: int, vehicle_type: Optional[str] = None,sort: Optional[Literal["plate_number","mileage"]] = None, db: Session = Depends(get_db)):
  v = db.query(Depot).filter(Depot.id == depot_id).first()

  if v is None:
    raise HTTPException(status_code = 404, detail = 'Depot not found')
  
  query = db.query(Vehicle).filter(Vehicle.depot_id == depot_id)

  if vehicle_type:
    query = query.filter(Vehicle.vehicle_type == vehicle_type)
  

  if sort == 'plate_number':
    query = query.order_by(Vehicle.plate_number.asc())
  if sort == 'mileage':
    query = query.order_by(Vehicle.mileage.asc())

  vehicles = query.all()

  return DepotWithVehicle(
    id = v.id,
    name = v.name,
    region = v.region,
    vehicles = vehicles
  )


# @app.put('/depots/{depot_id}/vehicles/{vehicle_id}', response_model = VehicleOut)
# def update_vehicle(depot_id : int, vehicle_id : int, vehicle : VehicleCreate, db : Session = Depends(get_db)):
#   d = db.query(Depot).filter(Depot.id == depot_id).first()
#   if d is None:
#     raise HTTPException(status_code = 404, detail = 'Depot not found')
  
#   v = db.query(Vehicle).filter(Vehicle.id == vehicle_id).first()
#   if v is None:
#     raise HTTPException(status_code = 404, detail = 'Vehicle not found')

#   if v.depot_id != depot_id:
#     raise HTTPException(status_code = 404, detail = 'Vehicle does not belong to this depot')
  
#   v.plate_number = vehicle.plate_number
#   v.vehicle_type = vehicle.vehicle_type
#   v.mileage = vehicle.mileage

#   db.commit()
#   db.refresh(v)

#   return v
  
  
@app.put('/depots/{depot_id}/vehicles/{vehicle_id}', response_model=VehicleOut)
def update_vehicle(
    depot_id: int,
    vehicle_id: int,
    vehicle: VehicleCreate,
    db: Session = Depends(get_db)
):
    d = db.query(Depot).filter(Depot.id == depot_id).first()

    if d is None:
        raise HTTPException(status_code=404, detail='Depot not found')

    v = db.query(Vehicle).filter(Vehicle.id == vehicle_id).first()

    if v is None:
        raise HTTPException(status_code=404, detail='Vehicle not found')

    if v.depot_id != depot_id:
        raise HTTPException(
            status_code=404,
            detail='Vehicle does not belong to this depot'
        )

    v.plate_number = vehicle.plate_number
    v.vehicle_type = vehicle.vehicle_type
    v.mileage = vehicle.mileage

    db.commit()
    db.refresh(v)

    return v
