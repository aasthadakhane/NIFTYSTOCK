import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# -------------- Page Config --------------
st.set_page_config(page_title=" Advanced Nifty Stock Predictor", layout="wide")
st.title(" Nifty Stocks Prediction with Advanced Features & Models")

# -------------- Load Data --------------
@st.cache_data
def load_data():
    df = pd.read_csv("Nifty_Stocks.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.fillna(0, inplace=True)
    return df

df = load_data()

# -------------- Feature Engineering --------------
def add_technical_indicators(df):
    df = df.copy()
    
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

    # EMA
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()

    # Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = rolling_mean + (rolling_std * 2)
    df['BB_Lower'] = rolling_mean - (rolling_std * 2)

    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df.loc[i, 'Close'] > df.loc[i - 1, 'Close']:
            obv.append(obv[-1] + df.loc[i, 'Volume'])
        elif df.loc[i, 'Close'] < df.loc[i - 1, 'Close']:
            obv.append(obv[-1] - df.loc[i, 'Volume'])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    return df.dropna()

df = add_technical_indicators(df)

# -------------- Encode --------------
label = LabelEncoder()
df['Symbol'] = label.fit_transform(df['Symbol'])
df['Category'] = label.fit_transform(df['Category'])

# -------------- Prepare Data --------------
drop_cols = ['Close', 'Adj Close', 'Volume', 'Date', 'Price_Range', 'Cumulative_Return', 'Average_Price']
X = df.drop(columns=drop_cols, errors='ignore')
y = df['Close']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -------------- Train Models --------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "SVR": SVR(kernel='rbf')
}

model_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_results[name] = {
        "model": model,
        "r2": round(r2_score(y_test, y_pred), 4),
        "y_pred": y_pred
    }

# -------------- Sidebar ----------------
selected_model_name = st.sidebar.selectbox("🔍 Choose Prediction Model", list(model_results.keys()))
selected_symbol = st.sidebar.selectbox("Select Stock Symbol", df['Symbol'].unique())

# Filter Data
df_selected = df[df['Symbol'] == selected_symbol].copy()
df_selected = df_selected.reset_index(drop=True)
model = model_results[selected_model_name]["model"]
df_selected['Predicted_Close'] = model.predict(scaler.transform(df_selected[X.columns]))

# -------------- Display Table --------------
st.subheader(f"Recent Predictions ({selected_model_name})")
st.dataframe(
    df_selected[['Date', 'Close', 'Predicted_Close', 'RSI', 'SMA_50', 'EMA_20', 'BB_Upper', 'BB_Lower']]
    .sort_values(by='Date', ascending=False)
    .head(10),
    use_container_width=True
)

# -------------- Display Metrics --------------
st.markdown("### Model Performance")
for name, res in model_results.items():
    st.metric(label=f"{name} R² Score", value=res["r2"])

# -------------- Plot ----------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_selected['Date'], df_selected['Close'], label="Actual", color="#1f77b4", linewidth=2)
ax.plot(df_selected['Date'], df_selected['Predicted_Close'], label="Predicted", color="#ff7f0e", linestyle="--", linewidth=2)
ax.set_title(f"{selected_model_name}: Actual vs Predicted Close Price", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# -------------- Footer ----------------
st.markdown("""
---
 **Note:** This dashboard is for educational purposes only. The predictions do not constitute investment advice.
""")
