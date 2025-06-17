import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# ---------------------- Load Data ------------------
df = pd.read_csv("Nifty_Stocks.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.fillna(0)
df = df.reset_index(drop=True)

# ---------------------- Data Engineering ------------------
df['Price_Change'] = df['Close'] - df['Open']
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)

df['Rolling_Mean_20'] = df['Close'].rolling(window=20).mean()
df['Rolling_Std_20'] = df['Close'].rolling(window=20).std()
df['Z_Score'] = (df['Close'] - df['Rolling_Mean_20']) / df['Rolling_Std_20']

df['Weekly_High'] = df['Close'].rolling(window=5).max()
df['Weekly_Low'] = df['Close'].rolling(window=5).min()

df['Bollinger_Upper'] = df['Rolling_Mean_20'] + (2 * df['Rolling_Std_20'])
df['Bollinger_Lower'] = df['Rolling_Mean_20'] - (2 * df['Rolling_Std_20'])

df = df.fillna(method='bfill')

# ---------------------- Encoding ------------------
label = LabelEncoder()
df['Symbol'] = label.fit_transform(df['Symbol'])
df['Category'] = label.fit_transform(df['Category'])

# ---------------------- Feature Selection ------------------
drop_cols = ['Close', 'Adj Close', 'Volume', 'Date']
X = df.drop(columns=drop_cols)
y = df['Close']

# ---------------------- Train/Test Split ------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------- Models ------------------
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)
xgb = XGBRegressor(random_state=42).fit(X_train, y_train)

# ---------------------- Evaluation ------------------
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

r2_scores = {
    "Linear Regression R²": round(r2_score(y_test, y_pred_lr), 4),
    "Random Forest R²": round(r2_score(y_test, y_pred_rf), 4),
    "XGBoost R²": round(r2_score(y_test, y_pred_xgb), 4),
}
rmse_scores = {
    "Linear Regression RMSE": round(mean_squared_error(y_test, y_pred_lr, squared=False), 4),
    "Random Forest RMSE": round(mean_squared_error(y_test, y_pred_rf, squared=False), 4),
    "XGBoost RMSE": round(mean_squared_error(y_test, y_pred_xgb, squared=False)),
}

# ---------------------- Save Models ------------------
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ---------------------- Streamlit App ------------------
st.set_page_config(layout="wide")
st.title(" Nifty Stocks Forecast Dashboard")

selected_symbol = st.sidebar.selectbox("Select Symbol", df['Symbol'].unique())
df_selected = df[df['Symbol'] == selected_symbol].copy()

with open("rf_model.pkl", "rb") as f:
    rf_loaded = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler_loaded = pickle.load(f)

df_selected['Predicted_Close'] = rf_loaded.predict(scaler_loaded.transform(df_selected[X.columns]))

# ---------------------- Dashboard Display ------------------
st.subheader("Recent Forecast")
st.dataframe(df_selected[['Date', 'Close', 'Predicted_Close']].sort_values(by='Date', ascending=False).head(10), use_container_width=True)

st.markdown("### Model Performance")
st.write("R² Scores:", r2_scores)
st.write("RMSE Scores:", rmse_scores)

# ---------------------- Price Plot ------------------
sns.set(style="whitegrid", palette="pastel")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_selected['Date'], df_selected['Close'], label='Actual Close', color='#87CEEB')
ax.plot(df_selected['Date'], df_selected['Predicted_Close'], label='Predicted Close', color='#FFB6C1')
ax.set_title("Stock Price Forecast", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# ---------------------- Technical Indicators ------------------
st.markdown("### Technical Indicators")
st.line_chart(df_selected.set_index('Date')[['Z_Score', 'Rolling_Mean_20', 'Rolling_Std_20']])

# ---------------------- Feature Engineering Visuals ------------------
st.markdown("Feature Engineering Visuals")

# Price Change
st.subheader("Daily Price Change")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df_selected['Date'], df_selected['Price_Change'], color="#ADD8E6")
ax1.axhline(0, color='gray', linestyle='--')
st.pyplot(fig1)

# Z-Score
st.subheader("Z-Score")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df_selected['Date'], df_selected['Z_Score'], color="#FF69B4")
ax2.axhline(0, color='black', linestyle='--')
ax2.axhline(2, color='red', linestyle=':')
ax2.axhline(-2, color='green', linestyle=':')
st.pyplot(fig2)

# Bollinger Bands
st.subheader("Bollinger Bands")
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(df_selected['Date'], df_selected['Close'], label='Close', color='black')
ax3.plot(df_selected['Date'], df_selected['Bollinger_Upper'], linestyle='--', color='#00FA9A', label='Upper Band')
ax3.plot(df_selected['Date'], df_selected['Bollinger_Lower'], linestyle='--', color='#FF6347', label='Lower Band')
ax3.fill_between(df_selected['Date'], df_selected['Bollinger_Lower'], df_selected['Bollinger_Upper'], color='gray', alpha=0.2)
ax3.legend()
st.pyplot(fig3)

# Log Return Distribution
st.subheader("Log Return Distribution")
fig4, ax4 = plt.subplots(figsize=(8, 4))
sns.histplot(df_selected['Log_Return'], kde=True, color="#87CEFA", ax=ax4)
st.pyplot(fig4)

# Weekly High and Low
st.subheader("Weekly High and Low")
fig5, ax5 = plt.subplots(figsize=(10, 4))
ax5.plot(df_selected['Date'], df_selected['Weekly_High'], label='Weekly High', color='#6A5ACD')
ax5.plot(df_selected['Date'], df_selected['Weekly_Low'], label='Weekly Low', color='#FFB6C1')
ax5.fill_between(df_selected['Date'], df_selected['Weekly_Low'], df_selected['Weekly_High'], color='lightgray', alpha=0.2)
ax5.legend()
st.pyplot(fig5)

# ---------------------- Insights ------------------
st.markdown("---")
st.markdown("#### Insights:")
st.markdown("- Model predicts stock closing prices using advanced features and XGBoost/Random Forest models.")
st.markdown("- RSI, Bollinger Bands, and Z-Score help spot overbought/oversold levels.")
st.markdown("- Visual insights from pastel charts enhance readability.")
