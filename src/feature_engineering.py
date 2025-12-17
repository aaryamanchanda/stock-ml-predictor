import pandas as pd
import numpy as np


def add_features(df):
    df = df.copy()

    # Flatten columns if yfinance returns MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Required columns
    price_cols = ["Open", "High", "Low", "Close"]

    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Volume is optional
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    else:
        df["Volume"] = 0

    # Returns
    df["Return"] = df["Close"].pct_change()

    # Moving averages
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

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

    # Target (only used during training)
    if "Target" not in df.columns:
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    return df
