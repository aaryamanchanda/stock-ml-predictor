import pandas as pd
import numpy as np


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


if __name__ == "__main__":
    df = pd.read_csv("data/stock_data.csv")
    df = add_features(df)
    df.to_csv("data/processed_data.csv", index=False)
    print("Processed data saved to data/processed_data.csv")
