#schemas.py
from typing import List
from pydantic import BaseModel, Field, ConfigDict

#go to error handling -- > for rules

class ClinicCreate(BaseModel):    # request does have constraint
  name : str = Field(..., min_length = 1)    # min_length means non emppty it will not be written directly as min_lenth, non empty will written
  city : str = Field(..., min_length = 1)

class ClinicOut(BaseModel): # response doesnt need any constraint
  id : int
  name : str
  city : str
  model_config = ConfigDict(from_attributes = True)  #whenever we write response model we need to write model_config
       # tells Pydantic to read data directly from regular class objects or database rows instead of just dictionaries

class ClinicWithAppointment(BaseModel):   #response model
  id : int
  name : str
  city : str
  aappointments : List[AppointmentOut]
  model_config = ConfigDict(from_attributes = True)

class AppointmentCreate(BaseModel):  #request model so there will be constraint
  patient_name : str = Field(..., min_length = 1)
  status : str = Field(..., min_length = 1)
  duration : int = Field(..., ge = 15)   # ge ---> >= ,,, gt --> greater than and vice versa

class AppointmentOut(BaseModel):  #response model
  id : int
  patient_name : str
  status : str
  duration : int
  model_config = ConfigDict(from_attributes = True)
