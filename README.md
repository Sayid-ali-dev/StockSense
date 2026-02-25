# StockSense — ML-Powered Stock Direction Predictor

A production-ready machine learning API that predicts whether a stock price will go **UP** or **DOWN** the next trading day. Built end-to-end with a trained Random Forest classifier, a FastAPI REST API, a clean frontend interface, and a fully automated CI/CD pipeline.

**Live Demo:** https://stocksense-k2tl.onrender.com

---

## Tech Stack

- **Machine Learning:** Scikit-learn (Random Forest Classifier)
- **Data:** yfinance (Yahoo Finance OHLCV data)
- **API:** FastAPI + Uvicorn
- **Frontend:** HTML, CSS, JavaScript (served via Jinja2)
- **Containerization:** Docker
- **Cloud Deployment:** Render
- **CI/CD:** GitHub Actions (automated model evaluation on every push)

---

## How It Works

1. **Training** — Historical OHLCV stock data is pulled from Yahoo Finance and transformed into engineered features: 7-day and 21-day moving averages, momentum, volatility, and volume change. A Random Forest classifier is trained on these features to predict next-day price direction.

2. **Serving** — The trained model is saved to disk and loaded into a FastAPI application at startup. When a request hits the `/predict` endpoint, the API pulls live data for the requested ticker, engineers the same features, and returns a prediction with a confidence score.

3. **CI/CD Pipeline** — Every push to main triggers a GitHub Actions workflow that automatically evaluates the model's accuracy against a minimum threshold. If the model passes, Render auto-redeploys the updated application. If it fails, deployment is blocked.

---

## Project Structure
```
StockSense/
├── app/
│   ├── main.py          # FastAPI application and endpoints
│   ├── train.py         # Model training and feature engineering
│   ├── evaluate.py      # Automated model evaluation for CI/CD
│   ├── stock_model.pkl  # Saved trained model
│   └── templates/
│       └── index.html   # Frontend UI
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml  # GitHub Actions CI/CD workflow
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Running Locally
```bash
# Clone the repo
git clone https://github.com/Sayid-ali-dev/StockSense.git
cd StockSense

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python app/train.py

# Start the API
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000` to use the app locally.

---

## API Endpoints

### `GET /`
Returns the frontend UI.

### `POST /predict`
Returns a stock price direction prediction.

**Request:**
```json
{
  "ticker": "AAPL"
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "prediction": "UP",
  "confidence": "63.00%",
  "message": "StockSense predicts AAPL will go UP tomorrow"
}
```

---

## CI/CD Pipeline

Every push to `main` triggers the GitHub Actions workflow:

1. Checks out the code on a fresh Ubuntu runner
2. Installs all dependencies
3. Runs `evaluate.py` to test model accuracy
4. If accuracy meets the minimum threshold — Render auto-redeploys
5. If accuracy falls below threshold — deployment is blocked

---

*Built by Sayid-Mohamed Ali — Georgia Tech MS CS Candidate (ML/AI)*