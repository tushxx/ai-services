from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


DATABASE_URL = "sqlite:///./parkwise.db"

engine = create_engine(DATABASE_URL, connect_args = {'check_same_thread' : False})

Sessionlocal = sessionmaker(bind = engine, autocommit = False, autoflush = False)

Base = declarative_base()

def get_db():

    db = Sessionlocal()
    try:
        yield db

    finally:
        db.close()
