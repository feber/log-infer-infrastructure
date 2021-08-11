from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    line: str


@app.get("/")
async def root():
    return {"hello": "for every soul out there."}


@app.post("/api/infer")
async def infer(item: Item):
    return {"result": f"{item.line}, that is."}
