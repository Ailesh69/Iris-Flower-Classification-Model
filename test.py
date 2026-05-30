import pytest 
from fastapi.testclient import TestClient
from main import app 

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/", follow_redirects=True)
    assert response.status_code == 200 , "Failed to access home endpoint"

def test_pred_page():
    response = client.get("/static/predict.html")
    assert response.status_code == 200 , "Failed to access prediction page"

def test_pred():
    payload= {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200 , "Failed to submit prediction"
    assert "predicted_species" in response.json() , "Response does not contain predicted_species field"