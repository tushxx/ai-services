#models.py
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from database import Base  ##coming from database.py

class Clinic(Base):
  __tablename__ = "clinics"
  id = Column(Integer, primary_key = True, index = True)    #index =True -->. auto generated column
  name  = Column(String, nullable = False)          #a data field or database column is required and cannot be left empty or stored as null
  city = Column(String, nullable = False)

  appointments = relationship("Appointment", back_populates = "clinics", cascade = "all, delete-orphan")
   #variable name should be of another table name, relationship with another Class name, back_populates should be on own table
   ### cascade means deleting a clinic also lead a deletion of appointment


class Appointment(Base):
  __tablename__ = "appointments"
  id = Column(Integer, primary_key = True, index = True)
  patient_name = Column(String, nullable = False)
  status = Column(String, nullable = False)
  duration = Column(Integer, nullable = False)
  clinic_id = Column(Integer, ForeignKey("clinics.id"))

  clinics = relationship("Clinic", back_populates = "appointments")
