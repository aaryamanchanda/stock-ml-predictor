import yfinance as yf
import pandas as pd


def load_stock_data(ticker, start="2015-01-01"):
    df = yf.download(ticker, start=start, progress=False)
    df.reset_index(inplace=True)
    return df


if __name__ == "__main__":
    df = load_stock_data("AAPL")
    df.to_csv("data/stock_data.csv", index=False)
    print("Data saved to data/stock_data.csv")
