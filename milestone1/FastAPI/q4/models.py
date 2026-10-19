from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from database import Base


class Lot(Base):
    __tablename__ = "lots"
    id = Column(Integer, primary_key = True, index = True)
    name = Column(String, nullable=False)
    zone = Column(String, nullable=False)
    slips = relationship('Slip', back_populates = 'lots')

class Slip(Base):
    __tablename__ = "slips"
    id = Column(Integer,primary_key=True,autoincrement=True,index=True)
    ticket_code = Column(String)
    vehicle_class = Column(String)
    parked_minutes = Column(Integer)
    lot_id = Column(Integer,ForeignKey("lots.id"),nullable=False)
    lots = relationship("Lot",back_populates="slips")



