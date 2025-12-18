import pandas as pd
import joblib

df = pd.read_csv("data/processed_data.csv")
model = joblib.load("model.pkl")

X = df.drop(columns=["Target", "Date"])
df["prob_up"] = model.predict_proba(X)[:, 1]

thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]

print("\nBacktest Results\n")

for t in thresholds:
    temp = df.copy()
    temp["Signal"] = (temp["prob_up"] > t).astype(int)
    temp["Strategy_Return"] = temp["Signal"].shift(1) * temp["Return"]
    temp.dropna(inplace=True)

    if len(temp) == 0:
        continue

    market = (1 + temp["Return"]).cumprod().iloc[-1]
    strategy = (1 + temp["Strategy_Return"]).cumprod().iloc[-1]

    print(f"Threshold {t} | Strategy: {round(strategy,2)} | Trades: {temp['Signal'].sum()}")
