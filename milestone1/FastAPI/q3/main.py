from typing import List, Literal, Optional
from fastapi import HTTPException, FastAPI, Response, Depends
from sqlalchemy.orm import Session
import uvicorn


from database import Base, engine, get_db
from models import Notebook, Entry
from schemas import NotebookCreate, NotebookOut, EntryCreate, EntryOut, NotebookWithEntries

Base.metadata.create_all(bind = engine)

app = FastAPI()


@app.post('/notebooks', response_model= NotebookOut, status_code=201)
def create_notebook(notebook:NotebookCreate, db:Session = Depends(get_db)):
    new_notebook = Notebook(
        title = notebook.title,
        author = notebook.author
    )
    db.add(new_notebook)
    db.commit()
    db.refresh(new_notebook)
    return new_notebook


@app.post('/notebooks/{notebook_id}/entries',response_model=EntryOut, status_code=201)
def add_entry(notebook_id: int, entry: EntryCreate, db: Session = Depends(get_db)):
    notebook = db.query(Notebook).filter(Notebook.id == notebook_id).first()

    if notebook is None:
        raise HTTPException(
            status_code=404,
            detail="notebook not found"
        )

    new_entry = Entry(
        heading = entry.heading,
        mood = entry.mood,
        word_count = entry.word_count,
        notebook_id = notebook_id
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)
    return new_entry


@app.get('/notebooks/{notebook_id}',response_model=NotebookWithEntries, status_code=200)
def get_notebook(notebook_id: int, mood: Optional[str] = None,sort: Optional[Literal["heading", "word_count"]] = None, db: Session = Depends(get_db)):
    notebook = db.query(Notebook).filter(Notebook.id == notebook_id).first()

    if notebook is None:
        raise HTTPException(
            status_code=404,
            detail="notebook not found"
        )
    
    entry = db.query(Entry).filter(Entry.notebook_id == notebook_id)

    if mood:
        entry = entry.filter(Entry.mood == mood)
    if sort == "heading":
        entry = entry.order_by(Entry.heading.asc())
    if sort == "word_count":
        entry = entry.order_by(Entry.word_count.asc())

    entries = entry.all()


    return NotebookWithEntries(
        id = notebook.id,
        title = notebook.title,
        author = notebook.author,
        entries = entries
    )


@app.delete('/notebooks/{notebook_id}/entries', status_code=204)
def clear_entries(notebook_id : int, db : Session = Depends(get_db)):

    notebook = db.query(Notebook).filter(Notebook.id == notebook_id).first()

    if notebook is None:
        raise HTTPException(
            status_code=404,
            detail="notebook not found"
        )

    db.query(Entry).filter(Entry.notebook_id == notebook_id).delete()

    db.commit()

    
