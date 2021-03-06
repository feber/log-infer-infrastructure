import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
from fastapi import FastAPI
from pydantic import BaseModel
from data.model import get_prediction
from typing import List

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
    """Accepts a JSON-formatted payload which contains text as infer input.

    Args:
        item (ModelParam): Payload which contains the infer input in `line` key.

    Returns:
        JSON: JSON-formatted response contains the predicted utilities, score, and the infer latency in millisecond.
    """
    utilities = []
    score = 0.0
    elapsed_ms = 0

    try:
        # make predictions based on the incoming data and neural network
        # then split by comma and strip the spaces at the beginning and the end
        utilities, score, elapsed_ms = get_prediction(item.line)

        # split utilities to array and trim wrapping whitespaces
        utilities = [utility.strip() for utility in utilities.strip().split(" ")]
    except:
        # explicitly return the error message to make it clear in Elasticsearch
        utilities = ["<error>"]

    # return the generated text and score
    return {"utilities": utilities, "score": score, "latency_ms": elapsed_ms}
