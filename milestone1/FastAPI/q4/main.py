from typing import Optional, List, Literal
from sqlalchemy.orm import Session
from fastapi import FastAPI, Query, Response, Depends, HTTPException
import uvicorn

from database import engine, get_db, Base
from models import Slip, Lot
from schemas import (
    LotCreate,
    LotOut,
    SlipCreate,
    SlipOut,
    SlipTransfer,
    LotWithSlips
)

Base.metadata.create_all(bind = engine)

app = FastAPI()


@app.post('/lots', response_model = LotOut, status_code = 201)
def create_lot(lot : LotCreate, db : Session = Depends(get_db)) -> LotOut :
    new_lot = Lot(
        name = lot.name,
        zone = lot.zone
    )

    db.add(new_lot)
    db.commit()
    db.refresh(new_lot)
    return new_lot


@app.post('/lots/{lot_id}/slips', response_model = SlipOut, status_code = 201)
def add_slip(lot_id: int, slip: SlipCreate, db: Session = Depends(get_db)) -> SlipOut:

    lot = db.query(Lot).filter(Lot.id == lot_id).first()
    if lot is None:
        raise HTTPException(status_code = 404, detail = "Lot not found")

    new_slip = Slip(
        ticket_code = slip.ticket_code,
        vehicle_class = slip.vehicle_class,
        parked_minutes = slip.parked_minutes,
        lot_id = lot_id
    )
    db.add(new_slip)
    db.commit()
    db.refresh(new_slip)
    return new_slip


@app.get('/lots/{lot_id}', response_model = LotWithSlips, status_code = 200)
def get_lot(lot_id : int, 
    vehicle_class : Optional[str] = None, 
    sort : Optional[Literal["ticket_code", "parked_minutes"]] = None,
    db : Session = Depends(get_db)
):
    lot = db.query(Lot).filter(Lot.id == lot_id).first()
    if lot is None:
        raise HTTPException(status_code = 404, detail = 'lot not found')

    slip = db.query(Slip).filter(Slip.lot_id == lot_id)
    
    if vehicle_class:
        slip = slip.filter(Slip.vehicle_class == vehicle_class)

    if sort == "ticket_code":
        slip = slip.order_by(Slip.ticket_code.asc())
    if sort == "parked_minutes":
        slip = slip.order_by(Slip.parked_minutes.asc())

    slips = slip.all()

    return LotWithSlips(
        id = lot.id,
        name = lot.name,
        zone = lot.zone,
        slips = slips
    )
    

@app.post('/lots/{lot_id}/slips/transfer', response_model = LotWithSlips, status_code = 200)
def transfer_slips(lot_id : int, transfer : SlipTransfer, db : Session = Depends(get_db)):

    lot = db.query(Lot).filter(Lot.id == lot_id).first()
    if lot is None:
        raise HTTPException(status_code = 404, detail = 'lot not found')


    target_lot = db.query(Lot).filter(Lot.id == transfer.target_lot_id).first()
    if target_lot is None:
        raise HTTPException(status_code = 404, detail = 'lot not found')

    slipey = db.query(Slip).filter(Slip.lot_id == lot_id).all()
    if slipey is None:
        raise HTTPException(status_code = 404, detail = 'lot not found')
        
    for slip in slipey:
        slip.lot_id = target_lot.id

    db.commit()
    db.refresh(target_lot)

    return LotWithSlips(
        id = target_lot.id,
        name = target_lot.name,
        zone = target_lot.zone,
        slips = target_lot.slips
    )
    

