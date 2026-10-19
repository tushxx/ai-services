from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship

from database import Base


class Boutique(Base):
    __tablename__ = "boutiques"
    id = Column(Integer, primary_key = True, index = True)
    name = Column(String, nullable=False)
    city = Column(String, nullable=False)
    products = relationship("Product", back_populates="boutiques")



class Product(Base):
    __tablename__ = "products"
    id = Column(Integer,primary_key=True,index=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    price = Column(Integer, nullable=False)
    highlighted = Column(Boolean, default=False, nullable=False)
    boutique_id = Column(Integer, ForeignKey('boutiques.id'), nullable = False)
    boutiques = relationship("Boutique", back_populates="products")
    