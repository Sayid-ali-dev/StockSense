import yfinance as yf
import pandas as pd 
import joblib 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MINIMUM_ACCURACY = 0.50

def get_data():
    raw_data = yf.download("AAPL", period="5y")
    raw_data.columns = raw_data.columns.get_level_values(0)
    return raw_data

def engineer_features(raw_data):
    feature_data = raw_data.copy()

    feature_data["moving_avg_7"] = feature_data["Close"].rolling(window=7).mean()
    feature_data["moving_avg_21"] = feature_data["Close"].rolling(window=21).mean()
    feature_data["momentum"] = feature_data["Close"].pct_change(periods=7)
    feature_data["volatility"] = feature_data["Close"].rolling(window=7).std()
    feature_data["volume_change"] = feature_data["Volume"].pct_change()
    feature_data["target"] = (
        feature_data["Close"].shift(-1) >  feature_data["Close"]
    ).astype(int)

    feature_data = feature_data.dropna()
    return feature_data

def evaluate_model():
    print("Loading model....")

    model = joblib.load("app/stock_model.pkl")

    print("Downloading evaluation data...")
    raw_data = get_data()

    print("Engineering features...")
    feature_data = engineer_features(raw_data)

    feature_columns = [
        "moving_avg_7",
        "moving_avg_21",
        "momentum",
        "volatility",
        "volume_change"
    ]

    X = feature_data[feature_columns]
    y = feature_data["target"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42
    )
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test,predictions)

    print(f"Model accuracy: {accuracy:.2%}")
    print(f"Minimum required: {MINIMUM_ACCURACY:.2%}")

    if accuracy < MINIMUM_ACCURACY:
        print("FAILED — accuracy below threshold. Blocking deployment.")
        raise SystemExit(1)
    else:
        print("PASSED — model meets accuracy threshold. Proceeding with deployment.")

if __name__ == "__main__":
    evaluate_model()


    







