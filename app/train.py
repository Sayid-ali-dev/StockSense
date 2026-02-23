import yfinance as yf
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def download_stock_data(ticker_symbol, period_of_time):
    raw_data = yf.download(ticker_symbol, period=period_of_time)
    raw_data.columns = raw_data.columns.get_level_values(0)
    print(raw_data.columns)
    print(raw_data.head())
    return raw_data

def engineer_features(raw_data):
    feature_data = raw_data.copy()

    # Moving averages — smooths out noise, captures trend direction
    feature_data["moving_avg_7"] = feature_data["Close"].rolling(window=7).mean()
    feature_data["moving_avg_21"] = feature_data["Close"].rolling(window=21).mean()

    # Momentum — how much the price changed over the last 7 days
    feature_data["momentum"] = feature_data["Close"].pct_change(periods=7)

    # Volatility — how much the price swings day to day (risk signal)
    feature_data["volatility"] = feature_data["Close"].rolling(window=7).std()

    # Volume change — unusual volume often precedes big moves
    feature_data["volume_change"] = feature_data["Volume"].pct_change()

    # Target — this is what we're predicting
    # 1 = price goes UP tomorrow, 0 = price goes DOWN
    feature_data["target"] = (
        feature_data["Close"].shift(-1) > feature_data["Close"]
    ).astype(int)

    # Drop rows with NaN values created by rolling calculations
    feature_data = feature_data.dropna()

    return feature_data

def train_model(feature_columns):

    feature_columns = [
        "moving_avg_7",
        "moving_avg_21",
        "momentum",
        "volatility",
        "volume_change"
    ]

    X = feature_data[feature_columns]
    y = feature_data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2%}")

    return model 

def save_model(model, file_path):
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}.")

if __name__ == "__main__":
    print("Downloading Stock Data ...")
    raw_data = download_stock_data("AAPL", "5y")

    print("Engineering Features ...")
    feature_data = engineer_features(raw_data)

    print("Train Model ...")
    trained_model = train_model(feature_data)

    save_model(trained_model, "app/stock_model.pkl")
