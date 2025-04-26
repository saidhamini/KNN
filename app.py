from fastapi import FastAPI
from schemas import HyperParams
from model import train_model
import joblib

X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("Y_train.pkl")
y_test = joblib.load("Y_test.pkl")

app = FastAPI()

@app.post("/train/")
def train(hyperparams: HyperParams):
    try:
        result = train_model(X_train, X_test, y_train, y_test, hyperparams.dict())
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}
