from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_pipeline_v1.pkl"

# Load the full pipeline (preprocessing + model)
model = joblib.load(MODEL_PATH)

class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

app = FastAPI(title="Fraud Detection API", version="2.0")

@app.post("/score-transaction")
def score_transaction(tx: Transaction):
    df = pd.DataFrame([tx.dict()])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "fraud_probability": round(float(prob), 4),
        "flagged": bool(pred)
    }
