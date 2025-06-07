import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBRegressor

# --------------- Technical Indicator Functions ------------------

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, min_periods=1).mean()
    ema_slow = series.ewm(span=slow, min_periods=1).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=1).mean()
    return macd_line - signal_line

# ---------------------- Load Data ------------------------------

df = pd.read_csv("Nifty_Stocks.csv")

# Ensure 'Date' column exists and convert it
df['Date'] = pd.to_datetime(df['Date'])

# Feature Engineering
df['Daily_Return'] = df['Close'].pct_change()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['Volatility'] = df['Daily_Return'].rolling(window=14).std()
df['RSI'] = calculate_rsi(df['Close'])
df['MACD'] = calculate_macd(df['Close'])

# Drop NaNs created during rolling computations
df = df.dropna()

# Encoding
label = LabelEncoder()
df['Symbol'] = df['Symbol'].astype(str)
df['Symbol_enc'] = label.fit_transform(df['Symbol'])

if 'Category' in df.columns:
    df['Category'] = df['Category'].astype(str)
    df['Category_enc'] = label.fit_transform(df['Category'])

# ---------------------- Feature Selection ----------------------

features = ['Open', 'High', 'Low', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Volatility']
X = df[features]
y = df['Close']

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Load model
model = joblib.load('xgb_model.pkl')

# Predict
df['Predicted_Close'] = model.predict(X_scaled)

# ---------------------- Streamlit UI ---------------------------

st.set_page_config(layout="wide")
st.title("📊 Global Stock Index: Price Prediction Dashboard")

selected_symbol = st.sidebar.selectbox("Select a Stock Symbol", sorted(df['Symbol'].unique()))
df_selected = df[df['Symbol'] == selected_symbol]

st.subheader(f"Predicted vs Actual Close Prices for {selected_symbol}")
st.dataframe(df_selected[['Date', 'Close', 'Predicted_Close']].sort_values('Date', ascending=False).head(10))

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_selected['Date'], df_selected['Close'], label="Actual Close", color="royalblue")
ax.plot(df_selected['Date'], df_selected['Predicted_Close'], label="Predicted Close", color="darkorange")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title(f"{selected_symbol} - Close Price Trend")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
