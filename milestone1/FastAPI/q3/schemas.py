from typing import List
from pydantic import Field, ConfigDict, BaseModel


class NotebookCreate(BaseModel):
    title : str = Field(..., min_length=1)
    author : str = Field(..., min_length=1)

class NotebookOut(BaseModel):
    id : int
    title : str
    author : str
    model_config = ConfigDict(from_attributes = True)


class EntryCreate(BaseModel):
    heading : str = Field(..., min_length=1)
    mood : str = Field(..., min_length=1)
    word_count : int = Field(..., ge=1)

class EntryOut(BaseModel):
    id : int
    heading : str
    mood : str
    word_count : str
    model_config = ConfigDict(from_attributes = True)

class NotebookWithEntries(BaseModel):
    id : int
    title : str
    author : str
    entries : List[EntryOut]
    model_config = ConfigDict(from_attributes = True)