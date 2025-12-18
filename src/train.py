import pandas as pd
import joblib
from model import get_model

df = pd.read_csv("data/processed_data.csv")

X = df.drop(columns=["Target", "Date"])
y = df["Target"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
y_train = y.iloc[:split]

model = get_model()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")
