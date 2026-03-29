# ML-Based Stock Trend Prediction System — Project Report

**Developer:** Aaryaman Chanda
**Repository:** [github.com/aaryamanchanda/stock-ml-predictor](https://github.com/aaryamanchanda/stock-ml-predictor)
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
| Deployment | Streamlit Web Application |
| Model Used | Random Forest Classifier |
| Prediction Target | Next-day stock price direction (UP / DOWN) |

---

## 2. Introduction

Stock market prediction is one of the most challenging problems in finance, characterized by high volatility, non-linearity, and sensitivity to external factors. This project presents an end-to-end machine learning pipeline that predicts the next-day directional movement of a stock — whether it will close higher (UP) or lower (DOWN) — using historical market data and derived technical indicators.

Rather than aiming for precise price prediction (a notoriously difficult regression problem), this system frames the task as a **binary classification problem**, which is more tractable and practically useful for trading signal generation. A confidence threshold is applied to the model outputs to filter out low-certainty predictions, resulting in a more disciplined trading strategy.

The project is structured as a modular Python codebase with a Streamlit front-end for interactive, real-time inference — making it suitable both as a demonstrable portfolio project and as a template for further research and development.

---

## 3. Objectives

- Ingest and preprocess historical stock price data using Yahoo Finance (`yfinance`)
- Engineer meaningful technical features — Moving Averages (MA), Relative Strength Index (RSI), and price volatility measures — from raw OHLCV data
- Train a Random Forest Classifier using time-series-aware cross-validation to avoid data leakage
- Evaluate model performance using accuracy, precision, recall, and a confusion matrix
- Implement a confidence-based trading strategy that only acts on high-certainty predictions (confidence > 0.6)
- Backtest the ML-driven strategy against a passive buy-and-hold benchmark
- Deploy the trained model as an interactive Streamlit web application for real-time inference

---

## 4. Technology Stack

| Category | Library / Tool | Purpose |
|---|---|---|
| Data Collection | yfinance | Fetch historical OHLCV stock data from Yahoo Finance |
| Data Processing | pandas, numpy | Data wrangling, feature computation, and array operations |
| Machine Learning | scikit-learn | Random Forest model training, evaluation, and cross-validation |
| Visualization | matplotlib | Plotting backtesting results and performance charts |
| Model Persistence | joblib | Serialize and deserialize the trained model (`model.pkl`) |
| Web Application | Streamlit | Interactive front-end for real-time stock prediction |
| Language | Python | Core programming language (100% of codebase) |

---

## 5. Project Structure

```
stock-ml-predictor/
├── data/
│   ├── stock_data.csv          ← Raw OHLCV data fetched from Yahoo Finance
│   └── processed_data.csv      ← Engineered features ready for model training
│
├── src/
│   ├── data_loader.py          ← Downloads and saves stock data via yfinance
│   ├── feature_engineering.py  ← Computes MA, RSI, volatility features
│   ├── model.py                ← Model definition and training logic
│   ├── train.py                ← Training pipeline orchestrator
│   ├── evaluate.py             ← Evaluation metrics and confusion matrix
│   └── backtest.py             ← Backtesting vs buy-and-hold benchmark
│
├── notebooks/                  ← Jupyter notebooks for exploration/EDA
├── app.py                      ← Streamlit web application entry point
├── model.pkl                   ← Serialized trained Random Forest model
├── model_accuracy.txt          ← Recorded model accuracy after training
├── requirements.txt            ← Python dependency list
└── README.md                   ← Project documentation
```

---

## 6. Methodology

### 6.1 Data Ingestion

Historical stock price data is fetched programmatically from Yahoo Finance using the `yfinance` library via `data_loader.py`. The data includes standard OHLCV fields: Open, High, Low, Close, and Volume, spanning a user-defined historical window. The raw data is saved locally to `data/stock_data.csv` for reproducibility.

### 6.2 Feature Engineering

Raw price data alone is insufficient for effective prediction. The `feature_engineering.py` module derives the following technical indicators:

- **Moving Averages (MA):** Short-term and long-term rolling averages to capture trend direction and momentum
- **Relative Strength Index (RSI):** A momentum oscillator measuring the speed and magnitude of price changes on a scale of 0–100
- **Volatility:** Rolling standard deviation of returns to capture market uncertainty
- **Binary Target Variable:** The label is `1` (UP) if the next day's close is higher than today's, else `0` (DOWN)

Processed features are saved to `data/processed_data.csv`.

### 6.3 Model Training

A **Random Forest Classifier** from scikit-learn is used as the predictive model. Random Forest was selected for its robustness to overfitting, ability to handle non-linear feature relationships, and built-in feature importance ranking. Training is performed in a time-series-aware manner — training data always precedes validation data chronologically — to prevent data leakage, a common pitfall in financial ML.

### 6.4 Evaluation

Model performance is assessed through several metrics computed in `evaluate.py`: overall accuracy, precision, recall, and F1 score. A confusion matrix is generated to visualize the distribution of true positives, false positives, true negatives, and false negatives. The model accuracy is persisted to `model_accuracy.txt` for reference.

### 6.5 Trading Strategy

A confidence-based trading signal is generated from the model's predicted class probabilities. A trade is only executed when the model's confidence exceeds a threshold of **0.6 (60%)**. This filters out borderline predictions and reduces the number of trades, prioritizing precision over recall — an important consideration in live trading contexts.

### 6.6 Backtesting

The `backtest.py` module simulates the performance of the confidence-filtered ML strategy over the historical dataset and compares cumulative returns against a passive buy-and-hold baseline. This provides an empirical measure of the strategy's practical utility beyond raw classification accuracy.

---

## 7. Results & Performance

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

## 8. Streamlit Web Application

The project includes a Streamlit web application (`app.py`) that provides an interactive interface for real-time stock direction prediction. Users can:

- Enter any valid stock ticker symbol (e.g., AAPL, TSLA, MSFT)
- Fetch the latest market data on-demand via `yfinance`
- Compute technical features in real time
- Load the pre-trained `model.pkl` and generate a next-day directional prediction
- View the model's confidence score alongside the UP/DOWN signal

To run the application locally, install dependencies and execute:

```bash
streamlit run app.py
```

---

## 9. Limitations & Considerations

- Stock market returns are influenced by macroeconomic events, news sentiment, and geopolitical factors not captured by technical indicators alone
- The ML strategy's lower absolute return compared to buy-and-hold reflects the conservative confidence threshold; tuning this value will directly impact the trade-off between precision and coverage
- The model is not retrained incrementally — in a production system, periodic retraining on recent data would be necessary to maintain predictive validity
- Past backtesting performance is not indicative of future returns; financial markets are non-stationary
- This project is explicitly designed for educational and portfolio demonstration purposes and does not constitute financial advice

---

## 10. Conclusion

The ML-Based Stock Trend Prediction System successfully demonstrates the application of machine learning techniques to financial time-series data. By framing the problem as a binary classification task, engineering informative technical features, and applying a confidence-based trading filter, the project delivers a coherent and reproducible end-to-end ML pipeline.

The accompanying Streamlit application makes the model accessible to non-technical users, while the modular codebase structure ensures maintainability and extensibility. Although the ML strategy does not outperform the raw buy-and-hold return in this sample, it demonstrates meaningful risk control — a key consideration in practical trading systems.

This project serves as a strong foundation for further research into ensemble methods, sentiment-augmented models, and reinforcement learning-based trading strategies.

---

## 11. Setup & Installation

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

# 6. Launch the Streamlit app
streamlit run app.py
```

---

> **Disclaimer:** This project is for educational use only and does not constitute financial advice. Past performance is not indicative of future results. Trading and investing in financial markets carries significant risk.
