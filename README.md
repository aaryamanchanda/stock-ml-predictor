# 📈 ML-Based Stock Trend Prediction System

An end-to-end machine learning project that predicts **next-day stock price direction (UP/DOWN)** using historical market data, technical indicators, and a confidence-based trading strategy. Deployed as a modern **web dashboard** on **Vercel** with a FastAPI backend and a sleek dark-themed frontend.

🔗 **Live Demo:** [stock-ml-predictor.vercel.app](https://stock-ml-predictor.vercel.app)

---

## 🚀 Features
- Real-time stock data fetching via Yahoo Finance
- Feature engineering with technical indicators (SMA, EMA, RSI, Volatility)
- Machine learning model (Random Forest Classifier)
- Time-series aware training (no data leakage)
- Confidence-based trading signals (threshold > 60%)
- Interactive candlestick charts with Plotly
- Backtesting against buy-and-hold strategy
- CSV data export
- **Two interfaces:**
  - 🌐 **Web Dashboard** — Modern dark-themed UI deployed on Vercel
  - 🖥️ **Streamlit App** — Local interactive application

---

## 🧠 Tech Stack

| Category | Technology |
|---|---|
| Backend API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript, Plotly.js |
| ML / Data | scikit-learn, pandas, numpy |
| Data Source | yfinance (Yahoo Finance) |
| Model Persistence | joblib |
| Local App | Streamlit |
| Deployment | Vercel (Serverless Python) |

---

## 📂 Project Structure
```
stock-ml-predictor/
│
├── api/
│   └── index.py                ← FastAPI backend (Vercel serverless function)
│
├── public/
│   └── index.html              ← Web dashboard frontend
│
├── data/
│   ├── stock_data.csv          ← Raw OHLCV data from Yahoo Finance
│   └── processed_data.csv      ← Engineered features for training
│
├── src/
│   ├── __init__.py             ← Package initializer
│   ├── data_loader.py          ← Downloads stock data via yfinance
│   ├── feature_engineering.py  ← Computes MA, RSI, volatility features
│   ├── model.py                ← Model definition
│   ├── train.py                ← Training pipeline
│   ├── evaluate.py             ← Evaluation metrics and confusion matrix
│   └── backtest.py             ← Backtesting vs buy-and-hold
│
├── notebooks/
│   └── EDA.ipynb               ← Exploratory data analysis
│
├── app.py                      ← Streamlit web application (local)
├── model.pkl                   ← Serialized trained Random Forest model
├── model_accuracy.txt          ← Recorded model accuracy
├── requirements.txt            ← Python dependencies
├── vercel.json                 ← Vercel deployment configuration
├── project_report.md           ← Detailed project report
└── README.md
```

---

## 📊 Model Overview
- **Target:** Predict whether the next trading day closes higher than today
- **Model:** Random Forest Classifier
- **Features:** Open, High, Low, Close, Volume, Return, MA10, MA50, EMA20, Volatility, RSI
- **Evaluation:** Accuracy, precision/recall, confusion matrix
- **Strategy:** Trade only when model confidence > 0.6

---

## 📈 Backtesting Results (Sample)
- Buy & Hold Return: ~9.7×
- ML Strategy Return: ~4–5×
- Trades: ~100
- Lower drawdown and controlled risk compared to naive strategies

---

## 🌐 Deployment (Vercel)

The project is deployed on [Vercel](https://vercel.com) as a serverless application:
- **Backend:** `api/index.py` runs as a Vercel Serverless Function using `@vercel/python`
- **Frontend:** `public/index.html` is served as a static asset
- **Config:** `vercel.json` routes `/api/*` requests to the FastAPI backend and all other requests to the static frontend

The deployed app provides the same analysis as the local Streamlit app — enter any ticker, get real-time technical analysis with an interactive candlestick chart and ML-based trend prediction.

---

## 🖥️ Run Locally

### Option 1: Web Dashboard (FastAPI)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn api.index:app --reload

# Open http://localhost:8000 in your browser
```

### Option 2: Streamlit App
```bash
# Install dependencies
pip install -r requirements.txt
pip install streamlit

# Run the Streamlit app
streamlit run app.py
```

### Train the Model from Scratch
```bash
python src/data_loader.py
python src/feature_engineering.py
python src/train.py
```

---

## 📌 Notes

- This project focuses on realistic ML evaluation, not perfect price prediction
- Accuracy is not optimized aggressively to avoid overfitting
- Designed for educational and internship demonstration purposes

## 📜 Disclaimer

This project is for educational use only and does not constitute financial advice. Past performance is not indicative of future results. Trading and investing in financial markets carries significant risk.