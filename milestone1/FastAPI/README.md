Model.py will be use to create a table -->> sqlalchemy --->> ORM model(table) class ____(Base)
              -- one to many relationship between em (cascade = "all, delete-orphan") --> deleting one table also removes its another table
main.py ---> 4 steps: 1. Import 2. instating 3. loading 4. crud operation
database.py -->> database engine --- SQLIte configuration
schema.py --->> pydantic basemodel --->> whenever we see request body schema or response schema(baseModel)

whenever you doing db.query operation next thing you need to write code for http 404 exception
