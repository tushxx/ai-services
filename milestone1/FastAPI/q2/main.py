from typing import Optional, Literal
from fastapi import FastAPI, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session
import uvicorn

from database import engine, Base, get_db
import models
from schemas import (
    StoreCreate,
    StoreOut,
    StoreWithItems,
    ItemCreate,
    ItemOut,
) 

Base.metadata.create_all(bind=engine)


app = FastAPI()

@app.post('/stores', response_model = StoreOut, status_code = 201)
def create_store(store: StoreCreate, db: Session = Depends(get_db)) -> StoreOut:
    new_store = models.Store(
        name = store.name,
        city = store.city
    )
    db.add(new_store)
    db.commit()
    db.refresh(new_store)
    return new_store


@app.post('/stores/{store_id}/items', response_model = ItemOut, status_code = 201 )
def add_item(store_id: int, item: ItemCreate, db: Session = Depends(get_db)) -> ItemOut:
    st = db.query(models.Store).filter(models.Store.id == store_id).first()

    if st is None:
        raise HTTPException(status_code = 404, detail = "Store not found")

    new_item = models.Item(
        name = item.name,
        category = item.category,
        stock_qty = item.stock_qty,
        store_id = store_id
    )
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return new_item


@app.get('/stores/{store_id}', response_model = StoreWithItems, status_code = 200)
def get_store(store_id : int, category : Optional[str] = None, sort: Optional[Literal["name", "stock_qty"]] = Query(None), db : Session= Depends(get_db)):
    store = db.query(models.Store).filter(models.Store.id == store_id).first()

    if store is None:
        raise HTTPException(
            status_code=404,
            detail=f"No store found with id {store_id}",
        )

    item_q = db.query(models.Item).filter(models.Item.store_id == store_id)

    if category:
        item_q = item_q.filter(models.Item.category == category)

    if sort == "name":
        item_q = item_q.order_by(models.Item.name.asc())

    if sort == "stock_qty":
        item_q = item_q.order_by(models.Item.stock_qty.asc())

    
    items = item_q.all()

    return StoreWithItems(
        id = store.id,
        name = store.name,
        city = store.city,
        items = items
    )

@app.delete("/stores/{store_id}/items/{item_id}", status_code=204)
def delete_item(
    store_id: int, item_id: int, db: Session = Depends(get_db)
) -> None:
    store = db.query(models.Store).filter(models.Store.id == store_id).first()
    if store is None:
        raise HTTPException(
            status_code=404,
            detail=f"No store found with id {store_id}",
        )
    
    item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if item is None:
        raise HTTPException(
            status_code=404,
            detail=f"No item found with id {item_id}",
        )

    if item.store_id != store_id:
        raise HTTPException(
            status_code=404,
            detail="this item is not associated with the provided store id",
        )
    
    db.delete(item)
    db.commit()