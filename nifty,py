import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# Load your data
df = pd.read_csv("Nifty_Stocks.csv")

# Preprocessing (same as your training code)
df['Daily_Return'] = df['Close'].pct_change()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['Volatility'] = df['Daily_Return'].rolling(window=14).std()

# You must also have RSI, MACD, etc. added here like you did before
# (To keep this short, reuse your RSI, MACD, EMA code here)

# Drop rows with NaNs created during indicator calculation
df = df.dropna()

# Encode labels
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df['Symbol_enc'] = label.fit_transform(df['Symbol'])
df['Category_enc'] = label.fit_transform(df['Category'])

# Features and target
features = ['Open', 'High', 'Low', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Volatility']
X = df[features]
y = df['Close']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Load the best model from GridSearchCV
# (You must save this model beforehand using joblib.dump(best_model, 'xgb_model.pkl'))
model = joblib.load('xgb_model.pkl')

# Predict using model
df['Predicted_Close'] = model.predict(X_scaled)

# ---------------------- Streamlit UI ----------------------

st.title("📈 Stock Price Prediction Dashboard")

# Sidebar stock selection
selected_symbol = st.sidebar.selectbox("Choose a Stock Symbol", sorted(df['Symbol'].unique()))
df_selected = df[df['Symbol'] == selected_symbol]

st.subheader(f"Predicted vs Actual Close Prices for {selected_symbol}")
st.write(df_selected[['Date', 'Close', 'Predicted_Close']].tail(10))

# Plot
fig, ax = plt.subplots()
ax.plot(df_selected['Date'], df_selected['Close'], label="Actual", color="blue")
ax.plot(df_selected['Date'], df_selected['Predicted_Close'], label="Predicted", color="orange")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title(f"{selected_symbol} Price Trend")
ax.legend()
st.pyplot(fig)
