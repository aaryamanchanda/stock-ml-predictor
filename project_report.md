# ML-Based Stock Trend Prediction System — Project Report

**Developer:** Aaryaman Chanda
**Repository:** [github.com/aaryamanchanda/stock-ml-predictor](https://github.com/aaryamanchanda/stock-ml-predictor)
**Live Demo:** [stock-ml-predictor.vercel.app](https://stock-ml-predictor.vercel.app)
**Report Date:** March 2026
**Language:** Python (100%)

---

## 1. Project Overview

| Field | Details |
|---|---|
| Project Title | ML-Based Stock Trend Prediction System |
| Developer | Aaryaman Chanda |
| Repository | github.com/aaryamanchanda/stock-ml-predictor |
| Language | Python (100%) |
| Purpose | Educational / Internship Portfolio Demonstration |
| Deployment | Vercel (Serverless Python + Static Frontend) |
| Local Interface | Streamlit Web Application |
| Model Used | Random Forest Classifier |
| Prediction Target | Next-day stock price direction (UP / DOWN) |

---

## 2. Introduction

Stock market prediction is one of the most challenging problems in finance, characterized by high volatility, non-linearity, and sensitivity to external factors. This project presents an end-to-end machine learning pipeline that predicts the next-day directional movement of a stock — whether it will close higher (UP) or lower (DOWN) — using historical market data and derived technical indicators.

Rather than aiming for precise price prediction (a notoriously difficult regression problem), this system frames the task as a **binary classification problem**, which is more tractable and practically useful for trading signal generation. A confidence threshold is applied to the model outputs to filter out low-certainty predictions, resulting in a more disciplined trading strategy.

The project is delivered as a **dual-interface application**:
- A **production web dashboard** deployed on Vercel, powered by a FastAPI backend and a modern dark-themed JavaScript frontend with interactive Plotly candlestick charts
- A **local Streamlit application** for rapid prototyping and development-time inference

This architecture makes the project suitable both as a demonstrable portfolio piece and as a template for further research and development.

---

## 3. Objectives

- Ingest and preprocess historical stock price data using Yahoo Finance (`yfinance`)
- Engineer meaningful technical features — Moving Averages (SMA/EMA), Relative Strength Index (RSI), and price volatility measures — from raw OHLCV data
- Train a Random Forest Classifier using time-series-aware cross-validation to avoid data leakage
- Evaluate model performance using accuracy, precision, recall, and a confusion matrix
- Implement a confidence-based trading strategy that only acts on high-certainty predictions (confidence > 0.6)
- Backtest the ML-driven strategy against a passive buy-and-hold benchmark
- Deploy the trained model as a production web application on Vercel for public access
- Provide an alternative local Streamlit interface for development and exploration

---

## 4. Technology Stack

| Category | Library / Tool | Purpose |
|---|---|---|
| Backend API | FastAPI | RESTful API serving ML predictions and stock data |
| ASGI Server | Uvicorn | Production ASGI server for FastAPI |
| Frontend | HTML, CSS, JavaScript | Modern dark-themed web dashboard |
| Charting | Plotly.js | Interactive candlestick charts with technical overlays |
| Data Collection | yfinance | Fetch historical OHLCV stock data from Yahoo Finance |
| Data Processing | pandas, numpy | Data wrangling, feature computation, and array operations |
| Machine Learning | scikit-learn | Random Forest model training, evaluation, and cross-validation |
| Model Persistence | joblib | Serialize and deserialize the trained model (`model.pkl`) |
| Local Application | Streamlit | Interactive local front-end for development |
| Deployment | Vercel | Serverless Python functions + static asset hosting |
| Language | Python | Core programming language (100% of codebase) |

---

## 5. Project Structure

```
stock-ml-predictor/
├── api/
│   └── index.py                ← FastAPI backend (Vercel serverless function)
│
├── public/
│   └── index.html              ← Production web dashboard (static frontend)
│
├── data/
│   ├── stock_data.csv          ← Raw OHLCV data fetched from Yahoo Finance
│   └── processed_data.csv      ← Engineered features ready for model training
│
├── src/
│   ├── __init__.py             ← Package initializer
│   ├── data_loader.py          ← Downloads and saves stock data via yfinance
│   ├── feature_engineering.py  ← Computes MA, RSI, volatility features
│   ├── model.py                ← Model definition and training logic
│   ├── train.py                ← Training pipeline orchestrator
│   ├── evaluate.py             ← Evaluation metrics and confusion matrix
│   └── backtest.py             ← Backtesting vs buy-and-hold benchmark
│
├── notebooks/
│   └── EDA.ipynb               ← Jupyter notebook for exploratory data analysis
│
├── app.py                      ← Streamlit web application (local interface)
├── model.pkl                   ← Serialized trained Random Forest model
├── model_accuracy.txt          ← Recorded model accuracy after training
├── requirements.txt            ← Python dependency list
├── vercel.json                 ← Vercel deployment configuration
├── project_report.md           ← This report
└── README.md                   ← Project documentation
```

---

## 6. Architecture

### 6.1 Deployment Architecture

The application follows a **serverless architecture** on Vercel:

```
                     ┌─────────────────────────────────┐
                     │           Vercel CDN             │
                     └──────────┬──────────┬────────────┘
                                │          │
                    /api/*      │          │    /*
                                │          │
                     ┌──────────▼──┐  ┌────▼───────────┐
                     │  Serverless  │  │  Static Assets  │
                     │  Function    │  │  (public/)      │
                     │  (FastAPI)   │  │                 │
                     │  api/index.py│  │  index.html     │
                     └──────┬──────┘  └─────────────────┘
                            │
                    ┌───────▼───────┐
                    │  model.pkl    │
                    │  (bundled)    │
                    └───────────────┘
```

- **Frontend:** A single-page HTML application (`public/index.html`) served as a static asset. It uses vanilla JavaScript to call the API and Plotly.js for interactive charting.
- **Backend:** A FastAPI application (`api/index.py`) deployed as a Vercel Serverless Function. It handles stock data fetching, feature engineering, and ML inference.
- **Model:** The trained `model.pkl` file is bundled into the serverless function's deployment package.
- **Routing:** `vercel.json` routes `/api/*` to the FastAPI handler and all other paths to the static frontend.

### 6.2 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/analyze?ticker=AAPL` | GET | Fetches stock data, computes features, runs prediction, returns JSON |
| `/api/health` | GET | Health check endpoint |

### 6.3 API Response Schema

```json
{
  "ticker": "AAPL",
  "dates": ["2024-01-02", "..."],
  "open": [185.12, "..."],
  "high": [186.50, "..."],
  "low": [184.80, "..."],
  "close": [185.90, "..."],
  "ma10": [184.50, "..."],
  "ema20": [183.20, "..."],
  "total_return": 42.15,
  "volatility": 1.85,
  "accuracy": 52.50,
  "prediction": "UP",
  "confidence": 0.6523,
  "current_price": 185.90,
  "start_price": 130.75
}
```

---

## 7. Methodology

### 7.1 Data Ingestion

Historical stock price data is fetched programmatically from Yahoo Finance using the `yfinance` library via `data_loader.py`. The data includes standard OHLCV fields: Open, High, Low, Close, and Volume, spanning from January 2018 to present. The raw data is saved locally to `data/stock_data.csv` for reproducibility. In the deployed API, data is fetched on-demand for any user-specified ticker.

### 7.2 Feature Engineering

Raw price data alone is insufficient for effective prediction. The `feature_engineering.py` module (also inlined in the API handler for serverless compatibility) derives the following technical indicators:

- **Simple Moving Average (SMA 10, SMA 50):** Short-term and long-term rolling averages to capture trend direction and momentum
- **Exponential Moving Average (EMA 20):** Weighted moving average giving more importance to recent prices
- **Relative Strength Index (RSI):** A momentum oscillator measuring the speed and magnitude of price changes on a scale of 0–100
- **Volatility:** 10-day rolling standard deviation of returns to capture market uncertainty
- **Daily Returns:** Percentage change in closing price
- **Binary Target Variable:** The label is `1` (UP) if the next day's close is higher than today's, else `0` (DOWN)

Processed features are saved to `data/processed_data.csv`.

### 7.3 Model Training

A **Random Forest Classifier** from scikit-learn is used as the predictive model. Random Forest was selected for its robustness to overfitting, ability to handle non-linear feature relationships, and built-in feature importance ranking. Training is performed in a time-series-aware manner — training data always precedes validation data chronologically — to prevent data leakage, a common pitfall in financial ML.

**Feature vector** used for prediction:
```
Open, High, Low, Close, Volume, Return, MA10, MA50, EMA20, Volatility, RSI
```

### 7.4 Evaluation

Model performance is assessed through several metrics computed in `evaluate.py`: overall accuracy, precision, recall, and F1 score. A confusion matrix is generated to visualize the distribution of true positives, false positives, true negatives, and false negatives. The model accuracy is persisted to `model_accuracy.txt` for reference and is displayed in the web dashboard.

### 7.5 Trading Strategy

A confidence-based trading signal is generated from the model's predicted class probabilities. A trade is only executed when the model's confidence exceeds a threshold of **0.6 (60%)**. This filters out borderline predictions and reduces the number of trades, prioritizing precision over recall — an important consideration in live trading contexts.

### 7.6 Backtesting

The `backtest.py` module simulates the performance of the confidence-filtered ML strategy over the historical dataset and compares cumulative returns against a passive buy-and-hold baseline. This provides an empirical measure of the strategy's practical utility beyond raw classification accuracy.

---

## 8. Results & Performance

| Metric | Value (Sample) |
|---|---|
| Buy & Hold Cumulative Return | ~9.7× the initial investment |
| ML Strategy Cumulative Return | ~4–5× the initial investment |
| Number of Trades Executed | ~100 trades |
| Confidence Threshold | > 0.60 (60%) |
| Drawdown Characteristic | Lower than passive buy-and-hold |
| Risk Profile | Controlled and conservative relative to market |

> **Note:** While the ML strategy underperforms the buy-and-hold return on a raw absolute basis in this sample, it achieves lower drawdown and executes fewer, higher-confidence trades. In volatile or bear market conditions, this risk-controlled approach would likely outperform relative to the passive baseline.

---

## 9. Web Dashboard

The production web dashboard (`public/index.html`) provides a modern, dark-themed interface featuring:

- **Search bar** — Enter any valid stock ticker (AAPL, TSLA, MSFT, etc.)
- **Performance summary cards** — Current price, total return, volatility, and model accuracy
- **ML prediction panel** — Next-day directional prediction with a visual confidence bar
- **Interactive candlestick chart** — Plotly.js chart with SMA 10 and EMA 20 overlays
- **CSV export** — Download the processed analysis data

The dashboard communicates with the FastAPI backend via the `/api/analyze` endpoint.

---

## 10. Streamlit Application (Local)

The project also includes a Streamlit web application (`app.py`) that provides an interactive local interface for stock direction prediction. Users can:

- Enter any valid stock ticker symbol (e.g., AAPL, TSLA, MSFT)
- Fetch the latest market data on-demand via `yfinance`
- Compute technical features in real time
- Load the pre-trained `model.pkl` and generate a next-day directional prediction
- View the model's confidence score alongside the UP/DOWN signal
- Download processed data as CSV

To run the application locally:

```bash
pip install streamlit
streamlit run app.py
```

---

## 11. Limitations & Considerations

- Stock market returns are influenced by macroeconomic events, news sentiment, and geopolitical factors not captured by technical indicators alone
- The ML strategy's lower absolute return compared to buy-and-hold reflects the conservative confidence threshold; tuning this value will directly impact the trade-off between precision and coverage
- The model is not retrained incrementally — in a production system, periodic retraining on recent data would be necessary to maintain predictive validity
- Vercel serverless functions have a 10-second execution timeout on the free tier; complex tickers with extensive historical data may approach this limit
- Past backtesting performance is not indicative of future returns; financial markets are non-stationary
- This project is explicitly designed for educational and portfolio demonstration purposes and does not constitute financial advice

---

## 12. Conclusion

The ML-Based Stock Trend Prediction System successfully demonstrates the application of machine learning techniques to financial time-series data. By framing the problem as a binary classification task, engineering informative technical features, and applying a confidence-based trading filter, the project delivers a coherent and reproducible end-to-end ML pipeline.

The dual-interface approach — a production-grade web dashboard deployed on Vercel alongside a local Streamlit application — showcases both software engineering best practices and data science methodology. The FastAPI backend provides a clean REST API for ML inference, while the modern frontend delivers an engaging user experience with interactive visualizations.

Although the ML strategy does not outperform the raw buy-and-hold return in this sample, it demonstrates meaningful risk control — a key consideration in practical trading systems. This project serves as a strong foundation for further research into ensemble methods, sentiment-augmented models, and reinforcement learning-based trading strategies.

---

## 13. Setup & Installation

**Prerequisites:** Python 3.8+, pip

```bash
# 1. Clone the repository
git clone https://github.com/aaryamanchanda/stock-ml-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download stock data
python src/data_loader.py

# 4. Engineer features
python src/feature_engineering.py

# 5. Train the model
python src/train.py

# 6a. Run the web dashboard locally
uvicorn api.index:app --reload
# Open http://localhost:8000

# 6b. Or run the Streamlit app
pip install streamlit
streamlit run app.py
```

### Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

The `vercel.json` configuration handles routing automatically — no additional setup required.

---

> **Disclaimer:** This project is for educational use only and does not constitute financial advice. Past performance is not indicative of future results. Trading and investing in financial markets carries significant risk.
