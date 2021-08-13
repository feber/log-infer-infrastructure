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
    return {"result": infer_model(item.line)}


def load_model():
    # TODO: load the model
    pass


def infer_model(line: str):
    # TODO: infer the `line` against the model
    return f"the infer result is => {line}"
