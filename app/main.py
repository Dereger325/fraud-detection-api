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
from google import genai
client = genai.Client(api_key=api_key)

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

    prompt = f"""A fraud detection model flagged a transaction as {prediction} with {fraud_proba:.4f} probability.

The top contributing factors (SHAP values, in probability terms, meaning each number is the approximate percentage-point contribution to the fraud probability) were:
{top_5_values}

Important constraints:
- These features (V1-V28) are anonymized PCA components from the original dataset provider. Their real-world meaning was never disclosed and is not derivable from the data. Do NOT speculate, guess, or state what these features "typically represent," "commonly capture," or "correlate with" in real-world terms.
- Only describe the relative magnitude and direction (positive/negative) of each feature's contribution, using the numbers given.
- Do not invent any fact, statistic, or domain claim not explicitly present in the data above.
- fraud_proba is the probability the transaction IS fraud, not the probability it is legitimate — do not invert this.

Explain in 2-3 plain-English sentences why this transaction was likely flagged, for a non-technical fraud analyst, using only what's stated above."""

    response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
    )
    return{
        "fraud_probability": round(float(fraud_proba), 4),
        "prediction" : prediction,
        "threshold_used" : THRESHOLD,
        "top_features" : top_5_values,
        "response" : response.text
    }



#print(response.text)