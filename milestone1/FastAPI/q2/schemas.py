from pydantic import BaseModel, ConfigDict, Field
from typing import List


class StoreCreate(BaseModel):
    name: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)


class StoreOut(BaseModel):
    id: int
    name: str
    city: str

    model_config = ConfigDict(from_attributes = True)

   

class ItemCreate(BaseModel):
    name: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    stock_qty: int = Field(..., ge=0)


class ItemOut(BaseModel):
    id: int
    name: str
    category: str
    stock_qty: int

    model_config = ConfigDict(from_attributes = True)

   

class StoreWithItems(BaseModel):
    id: int
    name: str
    city: str
    items: List[ItemOut]

    model_config = ConfigDict(from_attributes = True)

   