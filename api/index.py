import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import joblib

# ── Feature engineering (inlined to avoid cross-directory import issues on Vercel) ──

def add_features(df):
    df = df.copy()

    # Flatten MultiIndex if exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0

    # Returns
    df["Return"] = df["Close"].pct_change()

    # SMA
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # EMA
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # Volatility
    df["Volatility"] = df["Return"].rolling(10).std()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Target for ML
    if "Target" not in df.columns:
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    return df


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Stock ML Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ACCURACY_PATH = os.path.join(BASE_DIR, "model_accuracy.txt")

PREDICTION_THRESHOLD = 0.6
FEATURES = ["Open", "High", "Low", "Close", "Volume",
            "Return", "MA10", "MA50", "EMA20", "Volatility", "RSI"]


@app.get("/api/analyze")
def analyze(ticker: str = Query(..., description="Stock ticker symbol e.g. AAPL")):
    ticker = ticker.upper().strip()

    try:
        df = yf.download(ticker, start="2018-01-01", progress=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {e}")

    if df.empty:
        raise HTTPException(status_code=404, detail="Invalid ticker or no data available.")

    df.reset_index(inplace=True)
    df = add_features(df)

    # ── Performance metrics ──────────────────────────────────────────────────
    start_price = float(df["Close"].iloc[0])
    end_price = float(df["Close"].iloc[-1])
    total_return = ((end_price - start_price) / start_price) * 100
    volatility = float(df["Return"].std() * 100)

    # ── Model accuracy ───────────────────────────────────────────────────────
    accuracy = None
    try:
        with open(ACCURACY_PATH, "r") as f:
            accuracy = float(f.read().strip())
    except Exception:
        pass

    # ── ML Prediction ────────────────────────────────────────────────────────
    try:
        model = joblib.load(MODEL_PATH)
        X_latest = df[FEATURES].iloc[-1:].values
        prob_up = float(model.predict_proba(X_latest)[0][1])
        prediction = "UP" if prob_up > PREDICTION_THRESHOLD else "DOWN"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # ── Chart data (last 500 rows) ───────────────────────────────────────────
    chart_df = df.tail(500).copy()

    def safe_list(series):
        return [None if pd.isna(v) else round(float(v), 4) for v in series]

    return {
        "ticker": ticker,
        "dates": chart_df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "open":  safe_list(chart_df["Open"]),
        "high":  safe_list(chart_df["High"]),
        "low":   safe_list(chart_df["Low"]),
        "close": safe_list(chart_df["Close"]),
        "ma10":  safe_list(chart_df["MA10"]),
        "ema20": safe_list(chart_df["EMA20"]),
        "total_return": round(total_return, 2),
        "volatility":   round(volatility, 2),
        "accuracy":     round(accuracy * 100, 2) if accuracy else None,
        "prediction":   prediction,
        "confidence":   round(prob_up, 4),
        "current_price": round(end_price, 2),
        "start_price":   round(start_price, 2),
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}
