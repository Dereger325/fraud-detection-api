from fastapi import FastAPI
import joblib
import pandas as pd
from app.models import Transaction

app = FastAPI(title = "Fraud Detection API")
model = joblib.load("app/ml/model.pkl")
THRESHOLD = 0.28

@app.post("/predict")
def predict(transaction: Transaction):
    df = pd.DataFrame([transaction.model_dump()])
    fraud_proba = model.predict_proba(df)[0][1]
    prediction = "FRAUD" if fraud_proba >= THRESHOLD else "LEGITIMATE"
    return {
        "fraud_probability": round(float(fraud_proba), 4),
        "prediction" : prediction,
        "threshold_used" : THRESHOLD,
        "model_version" : "v1.0"
    }