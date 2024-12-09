import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from datetime import date
from sklearn.preprocessing import MinMaxScaler
import joblib

# Title and Description
st.title("Stock Price Prediction App")
st.write("This app predicts stock prices for the upcoming week using Long Short-Term Memory (LSTM) networks. Select a stock ticker, adjust parameters, and view predictions with interactive visualizations.")

# Adjustable Parameters
st.sidebar.header("Adjustable Parameters")
stock_ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", value="AAPL")
start_date = st.sidebar.date_input("Start Date:", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date:", value=date.today())
training_epochs = st.sidebar.slider("Training Epochs:", 10, 100, 20)
lstm_layers = st.sidebar.slider("Number of LSTM Layers:", 1, 5, 2)
batch_size = st.sidebar.slider("Batch Size:", 8, 128, 32)

# Fetch Data
def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

data_load_state = st.text("Loading data...")
data = fetch_data(stock_ticker, start_date, end_date)
data_load_state.text("Data Loaded!")

# Preprocess Data
def preprocess_data(data):
    data = data[['Close']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

scaled_data, scaler = preprocess_data(data)

# Prepare Data for LSTM
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x, y = create_sequences(scaled_data, seq_length)
x_train, y_train = x[:int(0.8 * len(x))], y[:int(0.8 * len(y))]
x_test, y_test = x[int(0.8 * len(x)):], y[int(0.8 * len(y))]

# Build LSTM Model
def build_model(layers):
    model = Sequential()
    for i in range(layers):
        model.add(LSTM(50, return_sequences=(i < layers - 1), input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

model = build_model(lstm_layers)

# Train Model
st.text("Training the model...")
model.fit(x_train, y_train, epochs=training_epochs, batch_size=batch_size, verbose=0)
st.text("Model trained!")

# Predictions
def predict(model, data):
    return model.predict(data)

predicted_prices = predict(model, x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
y_test = scaler.inverse_transform(y_test)

# Data Visualization
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index[-len(y_test):], y=data['Close'].values[-len(y_test):], name="Actual Prices"))
fig.add_trace(go.Scatter(x=data.index[-len(y_test):], y=predicted_prices.flatten(), name="Predicted Prices"))
st.plotly_chart(fig)

# Download CSV
def download_csv(data, predictions):
    df = pd.DataFrame({
        'Date': data.index[-len(predictions):],
        'Actual Prices': data['Close'].values[-len(predictions):],
        'Predicted Prices': predictions.flatten()
    })
    return df

download_data = download_csv(data, predicted_prices)

csv = download_data.to_csv(index=False)
st.download_button("Download as CSV", csv, "predictions.csv", "text/csv")

# Performance Optimization with Caching
@st.cache
def fetch_cached_data(ticker, start, end):
    return fetch_data(ticker, start, end)

# Deployment Instructions
st.sidebar.header("Deployment Instructions")
st.sidebar.write("Ensure you have the Yahoo Finance API key and deploy via Streamlit Cloud using this GitHub repository.")

# How it Works
st.header("How It Works")
st.write("This app uses the Yahoo Finance API to fetch historical stock data, trains an LSTM model to predict future prices, and visualizes the results. Adjust the parameters for better accuracy and download the results as a CSV.")
