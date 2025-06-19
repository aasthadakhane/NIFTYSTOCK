import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Set Streamlit Page Config
st.set_page_config(page_title="Nifty Stock Predictor", layout="wide")

st.title("📊 Nifty Stocks Prediction Dashboard")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Nifty_Stocks.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.fillna(0, inplace=True)
    return df

df = load_data()

# ---------------- Feature Engineering ----------------
def compute_indicators(df):
    df = df.copy()
    
    # SMA
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(14).std()
    
    return df.dropna()

df = compute_indicators(df).reset_index(drop=True)

# Label Encoding
label = LabelEncoder()
df['Symbol'] = label.fit_transform(df['Symbol'])
df['Category'] = label.fit_transform(df['Category'])

# ---------------- Model Training ----------------
features_to_drop = ['Close', 'Adj Close', 'Volume', 'Date', 'Price_Range', 'Cumulative_Return', 'Average_Price']
X = df.drop(columns=features_to_drop, errors='ignore')
y = df['Close']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

# Predictions
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

r2_lr = round(r2_score(y_test, y_pred_lr), 4)
r2_rf = round(r2_score(y_test, y_pred_rf), 4)

# ---------------- Sidebar UI ----------------
symbols = df['Symbol'].unique()
selected_symbol = st.sidebar.selectbox("🔎 Select Stock Symbol", symbols)

df_selected = df[df['Symbol'] == selected_symbol].copy()
df_selected['Predicted_Close'] = rf.predict(scaler.transform(df_selected[X.columns]))

# ---------------- Display Data ----------------
st.subheader("📅 Latest Predictions")
st.dataframe(df_selected[['Date', 'Close', 'Predicted_Close', 'RSI', 'SMA_50', 'SMA_200']].sort_values(by='Date', ascending=False).head(10), use_container_width=True)

# ---------------- Evaluation ----------------
with st.expander("📈 Model Performance"):
    st.metric("Linear Regression R²", r2_lr)
    st.metric("Random Forest R²", r2_rf)

# ---------------- Chart ----------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_selected['Date'], df_selected['Close'], label='Actual Close', color='#1f77b4', linewidth=2)
ax.plot(df_selected['Date'], df_selected['Predicted_Close'], label='Predicted Close', color='#ff7f0e', linestyle='--', linewidth=2)
ax.set_title("Actual vs Predicted Close Price", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# ---------------- Footer ----------------
st.markdown("""
---
**Disclaimer:** This tool is for educational purposes only. Do not use it for actual trading decisions.
""")
