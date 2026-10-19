from typing import List, Literal, Optional
from fastapi import FastAPI, Query, Response, HTTPException, Depends
from sqlalchemy.orm import Session
import uvicorn


from database import engine, Base, get_db
from models import Boutique, Product
from schemas import BoutiqueCreate, BoutiqueOut, ProductCreate, ProductOut, BoutiqueWithProducts


Base.metadata.create_all(bind = engine)

app = FastAPI()


@app.post('/boutiques', response_model=BoutiqueOut, status_code=201)
def create_boutique(boutique:BoutiqueCreate, db:Session=Depends(get_db)):
    new_boutique = Boutique(
        name = boutique.name,
        city = boutique.name
    )
    
    db.add(new_boutique)
    db.commit()
    db.refresh(new_boutique)
    return refresh


@app.put('/boutiques/{boutique_id}/products/{product_id}/highlight', response_model=ProductOut, status_code=200)
def highlight_product(boutique_id: int, product_id: int, db: Session = Depends(get_db)):
    boutique = db.query(Boutique).filter(Boutique.id == boutique_id).first()
    if boutique is None:
        raise HTTPException(
            status_code=404,
            detail="boutique not found"
        )

    product = db.query(Product).filter(Product.boutique_id == boutique_id).filter(Product.id == product_id).first()
    if product is None:
        raise HTTPException(
            status_code=404,
            detail="product not found"
        )

    products = db.query(Product).filter(Product.boutique_id == boutique_id).all()

    for product in products:
        product.highlighted = False


    product.highlighted = True

    db.commit()
    return product





