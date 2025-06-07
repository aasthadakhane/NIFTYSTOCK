import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    'Weekly_High', 'Weekly_Low', 'Bollinger_Upper', 'Bollinger_Lower',
    'Price_Change_Pct', 'Rolling_Median_10', 'Z_Score', 'Price_Diff', 'Daily_Mean_Dev'
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
df['Rolling_Mean_20'] = df['Close'].rolling(window=20).mean()
df['Rolling_Std_20'] = df['Close'].rolling(window=20).std()

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
df['Price_Range_Change'] = (df['High'] - df['Low']) - (df['High'] - df['Low']).shift(1)
df['Price_Ratio_10'] = df['Close'] / df['Close'].rolling(window=10).mean()
df['Price_Change_Pct'] = df['Close'].pct_change(periods=5) * 100
df['Rolling_Median_10'] = df['Close'].rolling(window=10).median()
df['Z_Score'] = (df['Close'] - df['Rolling_Mean_20']) / df['Rolling_Std_20']
df['Price_Diff'] = df['Close'] - df['Open']
df['Daily_Mean_Dev'] = df['Close'] - df['Close'].expanding().mean()

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
df_selected = df[df['Symbol'] == selected_symbol].copy()
df_selected['Predicted_Close'] = rf.predict(scaler.transform(df_selected[X.columns]))

# Display Table
st.subheader("Recent Predictions Snapshot")
st.dataframe(df_selected[['Date', 'Close', 'Predicted_Close']].sort_values(by='Date', ascending=False).head(10), use_container_width=True)

st.markdown("### Model Comparison")
st.write({
    "Linear Regression R² Score": round(r2_lr, 4),
    "Random Forest R² Score": round(r2_rf, 4)
})

# Plotting with pastel style
sns.set(style="whitegrid", palette="pastel")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_selected['Date'], df_selected['Close'], label='Actual Closing Price', color='#87CEEB')
ax.plot(df_selected['Date'], df_selected['Predicted_Close'], label='Predicted Closing Price', color='#FFB6C1')
ax.set_title("Stock Price Projection Over Time", fontsize=14)
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
ax2.plot(df_selected['Date'], df_selected['Bollinger_Upper'], label='Upper Band', linestyle='--', color='#98FB98')
ax2.plot(df_selected['Date'], df_selected['Bollinger_Lower'], label='Lower Band', linestyle='--', color='#F08080')
ax2.fill_between(df_selected['Date'], df_selected['Bollinger_Lower'], df_selected['Bollinger_Upper'], color='gray', alpha=0.1)
ax2.set_title("Bollinger Bands", fontsize=14)
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

st.markdown("---")
st.markdown("#### Insights:")
st.markdown("- This dashboard provides a forecast using Random Forest and visualizes key technical indicators.")
st.markdown("- Features like RSI, MACD, Bollinger Bands, and Z-Score help in interpreting stock trends.")
st.markdown("- Pastel visuals enhance readability and create an elegant analytic experience.")
