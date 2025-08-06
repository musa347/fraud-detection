 Fraud Detection API

A real-time fraud detection API that uses Machine Learning to identify suspicious financial transactions.  

This service exposes a REST API using FastAPI to score transactions and logs predictions to a PostgreSQL database. It is packaged with Docker for easy deployment to platforms like Render.

 Features

- Real-time fraud prediction via REST API
- Transaction logging with fraud probability
- PostgreSQL integration for transaction storage
- Automatic table creation on startup
- Dockerized for seamless deployment

 Tech Stack

- Python 3.13
- FastAPI
- Pandas and Scikit-learn for the ML pipeline
- Joblib for model serialization
- PostgreSQL
- Docker
- Render for cloud deployment

 Setup and Run Locally

```bash
 Clone the repository
git clone git@github.com:imusa654/fraud-detection.git
cd fraud-detection

 Create and activate virtual environment
python -m venv venv
source venv/bin/activate   For macOS/Linux
 venv\Scripts\activate    For Windows

 ->Install dependencies
pip install -r requirements.txt

 Run FastAPI locally
uvicorn serving.app:app --reload

->Docker Build and Run
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api

->Sample Api Request
{
  "step": 1,
  "type": "PAYMENT",
  "amount": 5000,
  "oldbalanceOrg": 10000,
  "newbalanceOrig": 5000,
  "oldbalanceDest": 0,
  "newbalanceDest": 5000
}

-> Sample Response
{
  "fraud_probability": 0.02,
  "flagged": false
}


