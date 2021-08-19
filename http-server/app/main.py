from fastapi import FastAPI
from pydantic import BaseModel
from data.predictions_handler import get_prediction
import random


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
async def predict():
    # prepare the input, just dummy for now
    data = {
        "variance_of_wavelet": random.uniform(0, 1),
        "skewness_of_wavelet": random.uniform(0, 1),
        "curtosis_of_wavelet": random.uniform(0, 1),
        "entropy_of_wavelet": random.uniform(0, 1),
    }

    # make predictions based on the incoming data and
    # neural network
    preds = get_prediction(data)

    # return the predicted class and the probability
    return {
        "predicted_class": round(float(preds.flatten())),
        "predicted_probability": float(preds.flatten()),
    }


@app.get("/api/model/load")
async def load_model(path: str):
    load_model("data")
