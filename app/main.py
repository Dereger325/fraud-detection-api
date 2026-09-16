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
import os 
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

import shap

background = pd.read_pickle("app/ml/background.pkl")
explainer = shap.TreeExplainer(
    model,
    data=background,
    feature_perturbation="interventional",
    model_output="probability"
)
@app.post("/explain")       

def explain(transaction : Transaction):
    df = pd.DataFrame([transaction.model_dump()])
    fraud_proba = model.predict_proba(df)[0][1]
    prediction = "FRAUD" if fraud_proba >= THRESHOLD else "LEGITIMATE"

    shap_values = explainer.shap_values(df)[0]
    shap_df = pd.DataFrame({
        "feature":df.columns,
        "shap_values":shap_values
    }).sort_values("shap_values",ascending=False,key=abs)

    top_5_values = shap_df.head(5).to_dict(orient="records")
    return{
        "fraud_probability": round(float(fraud_proba), 4),
        "prediction" : prediction,
        "threshold_used" : THRESHOLD,
        "top_features" : top_5_values
    }