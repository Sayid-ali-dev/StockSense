from fastapi import FastAPI 
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

app = FastAPI(
    title="StockSense API",
    description="ML-powered stock price direction predictor",
    version="1.0.0"
)

model = joblib.load("app/stock_model.pkl")

class StockRequest(BaseModel):
    ticker: str

def get_features(ticker_symbol):
    raw_data = yf.download(ticker_symbol, period="3mo")
    raw_data.columns = raw_data.columns.get_level_values(0)

    raw_data["moving_avg_7"] = raw_data["Close"].rolling(window=7).mean()
    raw_data["moving_avg_21"] = raw_data["Close"].rolling(window=21).mean()
    raw_data["momentum"] = raw_data["Close"].pct_change(periods=7)
    raw_data["volatility"] = raw_data["Close"].rolling(window=7).std()
    raw_data["volume_change"] = raw_data["Volume"].pct_change()

    raw_data = raw_data.dropna()

    latest_row = raw_data.iloc[-1]

    features = pd.DataFrame([{
        "moving_avg_7": latest_row["moving_avg_7"],
        "moving_avg_21": latest_row["moving_avg_21"],
        "momentum": latest_row["momentum"],
        "volatility": latest_row["volatility"],
        "volume_change": latest_row["volume_change"],
    }])

    return features

@app.get("/")

def home():
    return {"message": "Welcome to StockSense API", "status": "running"}

@app.post("/predict")

def predict(request:StockRequest):
    features = get_features(request.ticker)
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0].max()

    direction = "UP" if prediction == 1 else "DOWN"

    return {
        "ticker": request.ticker.upper(),
        "prediction": direction,
        "confidence": f"{confidence:.2%}",
        "message": f"StockSense predict {request.ticker.upper()} will go {direction} tomorrow."
    }


