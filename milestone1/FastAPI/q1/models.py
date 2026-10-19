from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class Depot(Base):
  __tablename__ = 'depots'
  id = Column(Integer, primary_key = True, index = True)
  name = Column(String, nullable = False)
  region = Column(String, nullable = False)

  vehicles = relationship('Vehicle', back_populates = 'depots')

class Vehicle(Base):
  __tablename__ = 'vehicles'
  id = Column(Integer, primary_key = True, index = True)
  plate_number = Column(Integer, nullable = False)
  vehicle_type = Column(String, nullable = False)
  mileage = Column(Integer, nullable = False)
  depot_id = Column(Integer, ForeignKey('depots.id'))

  depots = relationship('Depot', back_populates='vehicles')  