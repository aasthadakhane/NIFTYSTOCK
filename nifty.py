import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBRegressor

# ---------------------- Technical Indicator Functions ------------------

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

# ---------------------- Load & Preprocess Data ------------------

df = pd.read_csv("Nifty_Stocks.csv")
df['Date'] = pd.to_datetime(df['Date'])

df['Daily_Return'] = df['Close'].pct_change()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['Volatility'] = df['Daily_Return'].rolling(window=14).std()
df['RSI'] = calculate_rsi(df['Close'])
df['MACD'] = calculate_macd(df['Close'])

df = df.dropna()

label = LabelEncoder()
df['Symbol'] = df['Symbol'].astype(str)
df['Symbol_enc'] = label.fit_transform(df['Symbol'])

if 'Category' in df.columns:
    df['Category'] = df['Category'].astype(str)
    df['Category_enc'] = label.fit_transform(df['Category'])

features = ['Open', 'High', 'Low', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Volatility']
X = df[features]
y = df['Close']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------- Train Models ----------------------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# ---------------------- Evaluate Models -------------------------

rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_preds)
xgb_r2 = r2_score(y_test, xgb_preds)

# ---------------------- Predict full dataset with XGBoost -------------------------

df['Predicted_Close'] = xgb_model.predict(X_scaled)

# ---------------------- Streamlit Dashboard -----------------------

st.set_page_config(layout="wide")
st.title("📈 Stock Price Prediction Dashboard")

selected_symbol = st.sidebar.selectbox("Select a Stock Symbol", sorted(df['Symbol'].unique()))
df_selected = df[df['Symbol'] == selected_symbol]

st.subheader(f"Predicted vs Actual Close Prices for {selected_symbol}")
st.dataframe(df_selected[['Date', 'Close', 'Predicted_Close']].sort_values('Date', ascending=False).head(10))

# ---------------------- Metrics -----------------------

st.markdown("### 🔍 Model Evaluation (R² Score Only)")
st.write({
    "Random Forest R²": round(rf_r2, 4),
    "XGBoost R²": round(xgb_r2, 4),
})

# ---------------------- Display Parameters -----------------------

st.markdown("### ⚙️ Model Parameters")

st.subheader("🔹 Random Forest Parameters")
st.json({
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42
})

st.subheader("🔹 XGBoost Parameters")
st.json({
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42
})

# ---------------------- Plot -----------------------

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_selected['Date'], df_selected['Close'], label="Actual Close", color="blue")
ax.plot(df_selected['Date'], df_selected['Predicted_Close'], label="Predicted Close (XGBoost)", color="orange")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title(f"{selected_symbol} - Close Price Trend")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
