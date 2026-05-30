# 🌸 Iris Flower Classification

A machine learning project that predicts the species of an Iris flower from its
sepal and petal measurements. It trains a Support Vector Machine (SVM) on the
classic Iris dataset and serves predictions through a FastAPI backend with a
simple HTML frontend.

## Features

- **SVM classifier** (RBF kernel) trained with scikit-learn
- **FastAPI** REST API with a `/predict` endpoint
- **Static HTML frontend** for entering measurements and viewing predictions
- **Pytest** test suite covering the pages and the prediction endpoint

## Project Structure

```
Iris/
├── main.py            # FastAPI app (API + serves the frontend)
├── model.py           # Trains the SVM model and saves model.pkl
├── model.pkl          # Trained, serialized model
├── iris.csv           # Dataset
├── test.py            # Pytest tests
├── requirements.txt   # Python dependencies
└── static/
    ├── index.html     # Landing page
    ├── predict.html   # Prediction form
    └── style.css      # Styles
```

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

## Usage

### Train the model (optional)

`model.pkl` is already included, but you can retrain it:

```powershell
python model.py
```

### Run the API

```powershell
uvicorn main:app --reload
```

Then open <http://127.0.0.1:8000/> in your browser. The home page links to the
prediction form, where you enter the four measurements and get the predicted
species.

## API

### `POST /predict`

**Request body** (JSON):

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response**:

```json
{
  "predicted_species": "Iris-setosa"
}
```

Example with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}"
```

The model predicts one of three classes: `Iris-setosa`, `Iris-versicolor`, or
`Iris-virginica`.

## Testing

```powershell
pytest test.py -v
```

## Tech Stack

- Python
- FastAPI / Uvicorn
- scikit-learn, pandas, NumPy
- seaborn, matplotlib (for training-time visualizations)
- pytest
