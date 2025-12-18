import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import plotly.graph_objects as go

from src.feature_engineering import add_features

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    layout="centered"
)

st.title("📈 Stock Analysis Dashboard")
st.write(
    "Interactive stock analysis dashboard with technical indicators and an "
    "ML-based next-day trend prediction (educational use only)."
)

# -----------------------------
# User input
# -----------------------------
ticker = st.text_input(
    "Enter Stock Ticker (e.g. AAPL, MSFT, TSLA)",
    value="AAPL"
).upper()

PREDICTION_THRESHOLD = 0.6

# -----------------------------
# Analyze button
# -----------------------------
if st.button("Analyze"):
    try:
        # -----------------------------
        # Fetch data
        # -----------------------------
        with st.spinner("Fetching stock data..."):
            df = yf.download(ticker, start="2018-01-01", progress=False)

            if df.empty:
                st.error("❌ Invalid ticker or no data available.")
                st.stop()

            df.reset_index(inplace=True)
            df = add_features(df)

        # -----------------------------
        # Price chart & indicators
        # -----------------------------
        st.subheader("📊 Price Chart & Indicators")

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        ))

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["MA10"],
            name="SMA 10",
            line=dict(width=1)
        ))

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["EMA20"],
            name="EMA 20",
            line=dict(width=1)
        ))

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Performance summary
        # -----------------------------
        st.subheader("📈 Performance Summary")

        start_price = df["Close"].iloc[0]
        end_price = df["Close"].iloc[-1]

        total_return = ((end_price - start_price) / start_price) * 100
        volatility = df["Return"].std() * 100

        c1, c2 = st.columns(2)
        c1.metric("Total Return (%)", f"{total_return:.2f}")
        c2.metric("Volatility (%)", f"{volatility:.2f}")

        # -----------------------------
        # Model reliability (accuracy)
        # -----------------------------
        st.subheader("📊 Model Reliability")

        try:
            with open("model_accuracy.txt", "r") as f:
                acc = float(f.read())

            st.metric(
                label="Historical Test Accuracy",
                value=f"{acc * 100:.2f}%"
            )

            st.caption(
                "Accuracy measured on unseen historical data. "
                "In financial markets, accuracy alone does not guarantee profitability."
            )
        except:
            st.warning("Model accuracy data not found. Run evaluate.py first.")

        # -----------------------------
        # ML Prediction
        # -----------------------------
        st.subheader("🤖 ML-Based Trend Prediction (Extension)")

        model = joblib.load("model.pkl")

        # MUST match training features exactly (order matters)
        FEATURES = [
            "Open", "High", "Low", "Close", "Volume",
            "Return", "MA10", "MA50", "EMA20",
            "Volatility", "RSI"
        ]

        X_latest = df[FEATURES].iloc[-1:].values
        prob_up = model.predict_proba(X_latest)[0][1]

        prediction = "📈 UP" if prob_up > PREDICTION_THRESHOLD else "📉 DOWN"

        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Confidence:** {prob_up:.2f}")

        st.caption(
            f"The model predicts **UP** only when confidence exceeds {PREDICTION_THRESHOLD} "
            "to reduce false signals."
        )

        # -----------------------------
        # CSV export
        # -----------------------------
        st.subheader("⬇️ Download Data")

        st.download_button(
            label="Download processed stock data as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_stock_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("⚠️ An error occurred while running the analysis.")
        st.exception(e)
