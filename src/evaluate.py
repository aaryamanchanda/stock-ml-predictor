import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load data and model
df = pd.read_csv("data/processed_data.csv")
model = joblib.load("model.pkl")

X = df.drop(columns=["Target", "Date"])
y = df["Target"]

# Time-based split
split = int(len(df) * 0.8)
X_test = X.iloc[split:]
y_test = y.iloc[split:]

preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)

# Save accuracy to file
with open("model_accuracy.txt", "w") as f:
    f.write(f"{accuracy:.4f}")

print(f"Model accuracy saved: {accuracy:.4f}")
