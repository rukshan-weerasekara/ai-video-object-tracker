import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# --- 1. Page Config ---
st.set_page_config(page_title="AI Market Intelligence", layout="wide")

st.title("📊 Advanced AI Market Intelligence Tool")
st.markdown("Developed by **Rukshan Weerasekara** | Creative Technologist")
st.markdown("---")

# --- 2. Sidebar Inputs ---
st.sidebar.header("Configuration")
asset_ticker = st.sidebar.text_input("Enter Ticker (e.g., BTC-USD, TSLA, NVDA)", "BTC-USD").upper()
time_period = '2y'

if st.sidebar.button("Run AI Forecast"):
    try:
        with st.spinner(f"Fetching data for {asset_ticker}..."):
            # Data Acquisition
            df = yf.download(asset_ticker, period=time_period, interval='1d')

        if df.empty:
            st.error("Invalid ticker or no data found.")
        else:
            # Feature Engineering
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_30'] = df['Close'].rolling(window=30).mean()

            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df = df.dropna()

            X = df[['SMA_10', 'SMA_30', 'RSI']]
            y = df['Close']

            # Model Training
            split = int(0.8 * len(df))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_train.values, y_train.values.ravel())

            # Prediction
            predictions = model.predict(X_test.values)
            mae = mean_absolute_error(y_test, predictions)

            latest_data = [[float(df['SMA_10'].iloc[-1]), float(df['SMA_30'].iloc[-1]), float(df['RSI'].iloc[-1])]]
            next_day_prediction = model.predict(latest_data)[0]

            # Display Results
            current_price = float(df['Close'].iloc[-1])
            change_pct = ((next_day_prediction - current_price) / current_price) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:,.2f}")
            col2.metric("AI Prediction", f"${next_day_prediction:,.2f}", f"{change_pct:+.2f}%")
            col3.metric("Model Error (MAE)", f"${mae:,.2f}")

            # Visualization
            st.subheader(f"Historical Trend: {asset_ticker}")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index[-120:], df['Close'][-120:], label='Actual Price', color='#2c3e50')
            ax.plot(df.index[-120:], df['SMA_10'][-120:], label='10-Day Trend', linestyle='--')
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Status Error: {e}")
