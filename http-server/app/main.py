from fastapi import FastAPI
from pydantic import BaseModel
from .model import get_prediction


# initiate API
app = FastAPI()


class ModelParam(BaseModel):
    """
    Defines model for POST request.
    """

    line: str


@app.get("/")
async def root():
    return {"hello": "for every single soul out there."}


@app.post("/api/predict")
async def predict(item: ModelParam):
    # make predictions based on the incoming data and
    # neural network
    preds = get_prediction(item.line)

    # return the predicted class and the probability
    return {
        "result": preds,
    }


@app.get("/api/model/load")
async def load_model(path: str):
    load_model("data")
