from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from database import Base


class Store(Base):
    __tablename__ = "stores"
    id = Column(Integer, primary_key = True, index = True)
    name = Column(String, nullable = False)
    city = Column(String, nullable = False)

    items = relationship('Item', back_populates = 'stores')

class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key = True, index = True)
    name = Column(String, nullable = False)
    category = Column(String, nullable = False)
    stock_qty = Column(Integer, nullable = False)
    store_id = Column(Integer, ForeignKey('stores.id'))

    stores = relationship('Store', back_populates = 'items')
