from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from database import Base

class Notebook(Base):
    __tablename__ = "notebooks"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    author = Column(String, nullable=False)

    entries = relationship('Entry', back_populates= 'notebooks')


class Entry(Base):
    __tablename__ = "entries"
    id = Column(Integer, primary_key=True, index=True)
    heading = Column(String, nullable=False)
    mood = Column(String, nullable=False)
    word_count = Column(Integer, nullable=False)
    notebook_id = Column(Integer, ForeignKey('notebooks.id'))

    notebooks = relationship('Notebook', back_populates='entries')
