import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score

# ------------------- Technical Indicator Functions -------------------

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, min_periods=1).mean()
    ema_slow = series.ewm(span=slow, min_periods=1).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=1).mean()
    return macd_line - signal_line

# ------------------- Load Data -------------------

df = pd.read_csv("Nifty_Stocks.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Sort data by Symbol and Date to maintain time order
df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

# ------------------- Feature Engineering -------------------

# Calculate SMA_50 and SMA_200 per symbol
df['SMA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=50).mean())
df['SMA_200'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=200).mean())

# RSI and MACD per symbol
df['RSI'] = df.groupby('Symbol')['Close'].transform(calculate_rsi)
df['MACD'] = df.groupby('Symbol')['Close'].transform(calculate_macd)

# Drop rows with NaN from rolling calculations
df = df.dropna().reset_index(drop=True)

# Encode categorical columns
le_symbol = LabelEncoder()
df['Symbol_enc'] = le_symbol.fit_transform(df['Symbol'].astype(str))

le_category = LabelEncoder()
df['Category_enc'] = le_category.fit_transform(df['Category'].astype(str))

# ------------------- Prepare Data for ML -------------------

features = [
    'Open', 'High', 'Low',
    'SMA_50', 'SMA_200',
    'RSI', 'MACD',
    'Volatility',
    'Symbol_enc',
    'Category_enc'
]

X = df[features]
y = df['Close']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ------------------- Train Models -------------------

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ------------------- Predict & Evaluate -------------------

lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

lr_r2 = r2_score(y_test, lr_preds)
rf_r2 = r2_score(y_test, rf_preds)

# Predict full dataset for visualization
df['Predicted_Close_RF'] = rf_model.predict(X_scaled)
df['Predicted_Close_LR'] = lr_model.predict(X_scaled)

# ------------------- Streamlit Dashboard -------------------

st.set_page_config(layout="wide")
st.title("Stock Price Prediction Dashboard")

selected_symbol = st.sidebar.selectbox("Select Stock Symbol", sorted(df['Symbol'].unique()))

df_symbol = df[df['Symbol'] == selected_symbol].sort_values('Date')

st.subheader(f"Close Price: Actual vs Predicted ({selected_symbol})")
st.dataframe(df_symbol[['Date', 'Close', 'Predicted_Close_RF', 'Predicted_Close_LR']].tail(10))

st.markdown("### Model Performance (R² on Test Set)")
st.write({
    "Linear Regression R²": round(lr_r2, 4),
    "Random Forest R²": round(rf_r2, 4)
})

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_symbol['Date'], df_symbol['Close'], label='Actual Close', color='blue')
ax.plot(df_symbol['Date'], df_symbol['Predicted_Close_RF'], label='Predicted Close (Random Forest)', color='orange')
ax.plot(df_symbol['Date'], df_symbol['Predicted_Close_LR'], label='Predicted Close (Linear Regression)', color='green')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title(f"{selected_symbol} Close Price Prediction")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
