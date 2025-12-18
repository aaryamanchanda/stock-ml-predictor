# 📈 ML-Based Stock Trend Prediction System

An end-to-end machine learning project that predicts **next-day stock price direction (UP/DOWN)** using historical market data, technical indicators, and a confidence-based trading strategy. The model is deployed using **Streamlit** for real-time inference.

---

## 🚀 Features
- Historical stock data ingestion using Yahoo Finance
- Feature engineering with technical indicators (MA, RSI, volatility)
- Machine learning model (Random Forest)
- Time-series aware training (no data leakage)
- Confidence-based trading signals
- Backtesting against buy-and-hold strategy
- Interactive Streamlit web application

---

## 🧠 Tech Stack
- Python
- pandas, numpy
- scikit-learn
- yfinance
- matplotlib
- Streamlit
- joblib

---

## 📂 Project Structure
```bash
stock-ml-predictor/
│
├── data/
│ ├── stock_data.csv
│ └── processed_data.csv
│
├── src/
│ ├── data_loader.py
│ ├── feature_engineering.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ └── backtest.py
│
├── app.py
├── model.pkl
├── requirements.txt
└── README.md
```

---

## 📊 Model Overview
- **Target:** Predict whether the next trading day closes higher than today
- **Model:** Random Forest Classifier
- **Evaluation:** Accuracy, precision/recall, confusion matrix
- **Strategy:** Trade only when model confidence > 0.6

---

## 📈 Backtesting Results (Sample)
- Buy & Hold Return: ~9.7×
- ML Strategy Return: ~4–5×
- Trades: ~100
- Lower drawdown and controlled risk compared to naive strategies

---

## 🖥️ Run the Application

### 1️⃣ Install dependencies
```
pip install -r requirements.txt
```
### 2️⃣ Train the model
```
python src/data_loader.py
python src/feature_engineering.py
python src/train.py
```

### 3️⃣ Run Streamlit app
```
streamlit run app.py
```

## 📌 Notes

This project focuses on realistic ML evaluation, not perfect price prediction

Accuracy is not optimized aggressively to avoid overfitting

Designed for educational and internship demonstration purposes

## 📜 Disclaimer

This project is for educational use only and does not constitute financial advice.