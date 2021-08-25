import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from fastapi import FastAPI
from pydantic import BaseModel
from data.model import get_prediction

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
    preds = []

    try:
        # make predictions based on the incoming data and neural network
        # then split by comma and strip the spaces at the beginning and the end
        preds = [command.strip() for command in get_prediction(item.line).split(",")]
    except:
        # explicitly return the error message to make it clear in Elasticsearch
        preds = ["<error>"]

    # return the generated text as an array
    # and flatten it to avoid nested array
    return preds

# TODO: unzip the model, test to call, look at the time, then run the filebeat