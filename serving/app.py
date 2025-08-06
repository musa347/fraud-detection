from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
import os
import logging
import psycopg2

# -------------------------------
#  Paths & Model Loading
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_pipeline_v1.pkl"

model = joblib.load(MODEL_PATH)

# -------------------------------
#  Database Setup
# -------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

def init_db():
    """Create the transactions table if it does not exist."""
    if not DATABASE_URL:
        print("⚠️ No DATABASE_URL provided, skipping DB init")
        return
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id SERIAL PRIMARY KEY,
                step INT,
                type VARCHAR(50),
                amount FLOAT,
                oldbalanceOrg FLOAT,
                newbalanceOrig FLOAT,
                oldbalanceDest FLOAT,
                newbalanceDest FLOAT,
                fraud_probability FLOAT,
                flagged BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("Transactions table initialized successfully")
    except Exception as e:
        print("⚠️ DB init failed:", e)


def log_transaction(data, prob, flagged):
    """Insert transaction record into Postgres with improved logging."""
    if not DATABASE_URL:
        logging.error("DATABASE_URL not set. Skipping DB logging.")
        return

    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            conn.autocommit = True  # Ensures immediate insert
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO transactions (
                        step, type, amount, oldbalanceOrg, newbalanceOrig,
                        oldbalanceDest, newbalanceDest, fraud_probability, flagged
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    data['step'], data['type'], data['amount'],
                    data['oldbalanceOrg'], data['newbalanceOrig'],
                    data['oldbalanceDest'], data['newbalanceDest'],
                    prob, flagged
                ))
        logging.info(f"Transaction logged: {data} | Prob={prob}, Flagged={flagged}")
    except Exception as e:
        logging.exception("DB logging failed")


# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="Fraud Detection API", version="2.0")

class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


@app.on_event("startup")
def on_startup():
    """Initialize DB table on app startup."""
    init_db()


@app.post("/score-transaction")
def score_transaction(tx: Transaction):
    """Score a transaction and log it."""
    df = pd.DataFrame([tx.dict()])
    pred = model.predict(df)[0]
    prob = float(model.predict_proba(df)[0][1])
    flagged = bool(pred)

    log_transaction(tx.dict(), prob, flagged)

    return {
        "fraud_probability": round(prob, 4),
        "flagged": flagged
    }


# -------------------------------
# Local Run (for Render uses $PORT)
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
