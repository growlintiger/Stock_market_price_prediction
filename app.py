import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import streamlit as st
import plotly.graph_objects as go

# List of popular stock tickers for dropdown selection (including Indian stocks)
STOCKS = [
    "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NFLX", "META", "NVDA", "BABA", "IBM",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"
]

# Data Collection Function
def get_stock_data(ticker):
    data = yf.download(ticker, start="2015-01-01", end=pd.to_datetime('today').strftime('%Y-%m-%d'))
    data['Date'] = data.index
    data['Price'] = data['Close']
    data.dropna(inplace=True)
    return data

# Data Preparation for LSTM
def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Price']].values)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])

    return np.array(X), np.array(y), scaler

# Reshape for LSTM
def reshape_data(X):
    return np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build and Train Improved LSTM Model
def train_model(X, y):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(X.shape[1], 1)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=10)
    return model

# Streamlit Web Interface
def main():
    st.title("Stock Price Prediction App")

    # Scrolling Ticker Bar with Top Stocks (Color Coded)
    ticker_data = " | ".join([
        f"<span style='color:{'green' if yf.Ticker(stock).history(period='1d')['Close'].iloc[-1] >= yf.Ticker(stock).history(period='2d')['Close'].iloc[0] else 'red'}'>{stock}: {yf.Ticker(stock).history(period='1d')['Close'].iloc[-1]:.2f}</span>"
        for stock in STOCKS
    ])
    st.markdown(f"<marquee>{ticker_data}</marquee>", unsafe_allow_html=True)

    selected_stock = st.selectbox("Select a Stock Ticker", STOCKS)
    custom_ticker = st.text_input("Or Enter Custom Stock Ticker (e.g., TATAMOTORS.NS)")
    ticker = custom_ticker.strip() if custom_ticker else selected_stock

    data = get_stock_data(ticker)

    if st.button("Predict"):
        X, y, scaler = prepare_data(data)
        X = reshape_data(X)
        model = train_model(X, y)

        # Prediction for the next day
        latest_data = X[-1].reshape(1, X.shape[1], 1)
        predicted_price = model.predict(latest_data)
        predicted_price = scaler.inverse_transform(predicted_price)[0, 0]

        # Accuracy Calculation
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)
        mse = mean_squared_error(data['Price'].values[-len(predictions):], predictions)

        latest_price = data['Price'].iloc[-1]
        prev_day_price = data['Price'].iloc[-2]
        actual_trend_color = 'green' if latest_price > prev_day_price else 'red'
        price_trend = "⬆ Up" if predicted_price > latest_price else "⬇ Down"
        price_color = "green" if predicted_price > latest_price else "red"

        st.markdown(f"### <span style='color:{actual_trend_color}'>Latest Price: ${latest_price:.2f}</span>", unsafe_allow_html=True)
        st.markdown(f"### <span style='color:{price_color}'>Predicted Next Day Price: ${predicted_price:.2f} ({price_trend})</span>", unsafe_allow_html=True)
        st.write(f"### Model MSE (Lower is better): {mse:.4f}")

        # Improved Visuals using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Price'], mode='lines', name='Actual Price'))
        fig.add_trace(go.Scatter(x=[data['Date'].iloc[-1]], y=[predicted_price], mode='markers', marker=dict(color='red', size=10), name='Predicted Price'))
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()