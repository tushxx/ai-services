#database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./meditrack.db"           ########## main thing in this

engine = create_engine(DATABASE_URL, connect_args = {'check_same_thread':False})

SessionLocal = sessionmaker(bind = engine, autocommit = False, autoflush = False)

Base = declarative_base()1

def get_db():
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()
