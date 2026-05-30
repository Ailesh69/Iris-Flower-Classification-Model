from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import joblib
from pydantic import BaseModel
import numpy as np
import pandas as pd
import logging

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers =[logging.StreamHandler()]
)


try:
    load_model = joblib.load('model.pkl')
    logging.info('Model loaded successfully.')
except Exception as e:
    logging.error(f'Error loading model: {e}')
    raise RuntimeError(f'Failed to load model: {e}')


class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

special_mapping = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

@app.get("/")
async def root():
    try:
        logging.info('Root endpoint accessed.')
        return RedirectResponse(url="/static/index.html")
    except Exception as e:
        logging.error(f'Error occurred while accessing root endpoint: {e}')
        raise RuntimeError(f'Failed to access root endpoint: {e}',status_code=500)

@app.post("/predict")
def pred(data: IrisData):
    try:
        logging.info('Prediction endpoint accessed.')
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        df = pd.DataFrame(
            [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]],
            columns=features,
        )
        pred = load_model.predict(df)[0]
        species = special_mapping.get(int(pred), "Unknown")
        return {"predicted_species": species}
    except Exception as e:
        logging.error(f'Error occurred during prediction: {e}')
        raise RuntimeError(f'Failed to make prediction: {e}', status_code=500)
