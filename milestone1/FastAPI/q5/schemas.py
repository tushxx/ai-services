from typing import List
from pydantic import BaseModel, Field, ConfigDict


class BoutiqueCreate(BaseModel):
    name : str = Field(..., min_length=1)
    city : str = Field(..., min_length=1)

class BoutiqueOut(BaseModel):
    id : int
    name : str
    city : str

    model_config = ConfigDict(from_attributes = True)

class ProductCreate(BaseModel):
    name : str = Field(..., min_length=1)
    category :str = Field(..., min_length=1)
    price : int = Field(..., ge = 1)


class ProductOut(BaseModel):
    id : int
    name : str
    category : str
    price : int
    highlighted : bool
    model_config = ConfigDict(from_attributes=True)


class BoutiqueWithProducts(BaseModel):
    id : int
    name : str
    city : str
    products : List(ProductOut)