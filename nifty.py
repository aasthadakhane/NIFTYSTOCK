import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ---------------------- Load Custom Dataset ------------------
df = pd.read_csv("Nifty_Stocks.csv")  # Replace with the correct file name

# Preprocess
df['Date'] = pd.to_datetime(df['Date'])
df = df.fillna(0)
df = df.reset_index(drop=True)

# Initialize technical indicators
for col in ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal', 'Volatility']:
    df[col] = np.nan

# SMA
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()

# RSI
period = 14
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=period).mean()
avg_loss = loss.rolling(window=period).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Volatility
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility'] = df['Daily_Return'].rolling(window=14).std()

# Clean
df = df.dropna()

# Label Encode
label = LabelEncoder()
df['Symbol'] = label.fit_transform(df['Symbol'])
df['Category'] = label.fit_transform(df['Category'])

# Feature & Target
drop_cols = ['Close', 'Adj Close', 'Volume', 'Date', 'Price_Range', 'Cumulative_Return', 'Average_Price']
X = df.drop(columns=drop_cols)
y = df['Close']

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------- Models ------------------
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)

# ---------------------- Evaluation ------------------
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)

# ---------------------- Streamlit App ------------------
st.set_page_config(layout="wide")
st.title("Nifty Stocks Prediction Dashboard")

selected_symbol = st.sidebar.selectbox("Select Symbol", df['Symbol'].unique())
df_selected = df[df['Symbol'] == selected_symbol]
df_selected['Predicted_Close'] = rf.predict(scaler.transform(df_selected[X.columns]))

# Display
st.subheader("Latest Predictions")
st.dataframe(df_selected[['Date', 'Close', 'Predicted_Close']].sort_values(by='Date', ascending=False).head(10))

st.markdown("### Model Evaluation")
st.write({
    "Linear Regression R²": round(r2_lr, 4),
    "Random Forest R²": round(r2_rf, 4)
})

# Plot
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_selected['Date'], df_selected['Close'], label='Actual', color='blue')
ax.plot(df_selected['Date'], df_selected['Predicted_Close'], label='Predicted', color='orange')
ax.set_title("Actual vs Predicted Close Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
