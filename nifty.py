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
df = pd.read_csv("Nifty_Stocks.csv")

# Preprocess
df['Date'] = pd.to_datetime(df['Date'])
df = df.fillna(0)
df = df.reset_index(drop=True)

# Initialize technical indicators
tech_cols = [
    'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal', 'Volatility',
    'Momentum', 'Log_Return', 'Rolling_Mean_20', 'Rolling_Std_20',
    'Price_Range_Change', 'Price_Ratio_10', 'Expanding_Mean', 'Expanding_Std',
    'Weekly_High', 'Weekly_Low', 'Bollinger_Upper', 'Bollinger_Lower'
]
for col in tech_cols:
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

# Momentum
df['Momentum'] = df['Close'] - df['Close'].shift(periods=5)

# Log Return
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

# Rolling stats
rolling_window = 20
df['Rolling_Mean_20'] = df['Close'].rolling(window=rolling_window).mean()
df['Rolling_Std_20'] = df['Close'].rolling(window=rolling_window).std()

# Expanding stats
df['Expanding_Mean'] = df['Close'].expanding().mean()
df['Expanding_Std'] = df['Close'].expanding().std()

# Weekly High & Low
df['Weekly_High'] = df['Close'].rolling(window=5).max()
df['Weekly_Low'] = df['Close'].rolling(window=5).min()

# Bollinger Bands
df['Bollinger_Upper'] = df['Rolling_Mean_20'] + (2 * df['Rolling_Std_20'])
df['Bollinger_Lower'] = df['Rolling_Mean_20'] - (2 * df['Rolling_Std_20'])

# New creative features
price_range_today = df['High'] - df['Low']
price_range_yesterday = price_range_today.shift(1)
df['Price_Range_Change'] = price_range_today - price_range_yesterday
df['Price_Ratio_10'] = df['Close'] / df['Close'].rolling(window=10).mean()

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
st.subheader("Recent Predictions Snapshot")
st.dataframe(df_selected[['Date', 'Close', 'Predicted_Close']].sort_values(by='Date', ascending=False).head(10))

st.markdown("### Model Comparison")
st.write({
    "Linear Regression R² Score": round(r2_lr, 4),
    "Random Forest R² Score": round(r2_rf, 4)
})

# Plotting
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_selected['Date'], df_selected['Close'], label='Actual Closing Price', color='navy')
ax.plot(df_selected['Date'], df_selected['Predicted_Close'], label='Predicted Closing Price', color='crimson')
ax.set_title("Stock Price Projection Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# Additional Visuals
st.markdown("### Technical Indicators")
st.line_chart(df_selected.set_index('Date')[['RSI', 'MACD', 'Momentum', 'Volatility']])

st.markdown("### Bollinger Bands")
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(df_selected['Date'], df_selected['Close'], label='Close', color='black')
ax2.plot(df_selected['Date'], df_selected['Bollinger_Upper'], label='Upper Band', linestyle='--', color='green')
ax2.plot(df_selected['Date'], df_selected['Bollinger_Lower'], label='Lower Band', linestyle='--', color='red')
ax2.fill_between(df_selected['Date'], df_selected['Bollinger_Lower'], df_selected['Bollinger_Upper'], color='gray', alpha=0.1)
ax2.set_title("Bollinger Bands")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)
