from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import joblib
from pydantic import BaseModel
import numpy as np
import pandas as pd

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

load_model = joblib.load('model.pkl')

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

special_mapping = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.post("/predict")
def pred(data: IrisData):
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df = pd.DataFrame(
        [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]],
        columns=features,
    )

    pred = load_model.predict(df)[0]
    species = special_mapping.get(int(pred), "Unknown")
    return {"predicted_species": species}
