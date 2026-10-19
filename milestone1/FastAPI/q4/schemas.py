from typing import List
from pydantic import BaseModel, ConfigDict, Field


class LotCreate(BaseModel):
    name : str = Field(..., min_length = 1)
    zone : str = Field(..., min_length = 1)

class LotOut(BaseModel):
    id : int
    name : str
    zone : str
    model_config = ConfigDict(from_attributes = True)

class SlipCreate(BaseModel):
    ticket_code : str = Field(..., min_length = 1)
    vehicle_class : str = Field(..., min_length = 1)
    parked_minutes : int = Field(..., ge = 1)

class SlipOut(BaseModel):
    id : int
    ticket_code : str
    vehicle_class : str
    parked_minutes : int
    model_config = ConfigDict(from_attributes = True)


class SlipTransfer(BaseModel):
    target_lot_id : int 


class LotWithSlips(BaseModel):
    id : int
    name : str
    zone : str
    slips : List[SlipOut]
    model_config = ConfigDict(from_attributes = True)