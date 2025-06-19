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

# Page setup
st.set_page_config(page_title="Nifty Stock Prediction", layout="wide")
st.title("Nifty Stock Prediction Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Nifty_Stocks.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.fillna(0, inplace=True)
    return df

df = load_data()

# Feature engineering
def add_indicators(df):
    df = df.copy()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(14).std()

    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()

    rolling_mean = df['Close'].rolling(20).mean()
    rolling_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = rolling_mean + (rolling_std * 2)
    df['BB_Lower'] = rolling_mean - (rolling_std * 2)

    df['Momentum'] = df['Close'] - df['Close'].shift(10)

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

df = add_indicators(df)

# Encode labels
label = LabelEncoder()
df['Symbol'] = label.fit_transform(df['Symbol'])
df['Category'] = label.fit_transform(df['Category'])

# Feature and target
drop_cols = ['Close', 'Adj Close', 'Volume', 'Date', 'Price_Range', 'Cumulative_Return', 'Average_Price']
X = df.drop(columns=drop_cols, errors='ignore')
y = df['Close']

# Scale and split
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
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

# Sidebar
selected_model_name = st.sidebar.selectbox("Select Prediction Model", list(model_results.keys()))
selected_symbol = st.sidebar.selectbox("Select Stock Symbol", df['Symbol'].unique())

# Filtered Data
df_selected = df[df['Symbol'] == selected_symbol].copy().reset_index(drop=True)
model = model_results[selected_model_name]["model"]
df_selected['Predicted_Close'] = model.predict(scaler.transform(df_selected[X.columns]))

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Indicators", "Model Insights", "Data Table"])

# Tab 1: Prediction chart
with tab1:
    st.subheader("Actual vs Predicted Close Price")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df_selected['Date'], df_selected['Close'], label="Actual", color='blue')
    ax1.plot(df_selected['Date'], df_selected['Predicted_Close'], label="Predicted", color='orange', linestyle='--')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.set_title("Actual vs Predicted Close Price")
    ax1.legend()
    st.pyplot(fig1)

# Tab 2: Indicators chart
with tab2:
    st.subheader("RSI and Bollinger Bands")
    fig2, ax2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Bollinger Bands
    ax2[0].plot(df_selected['Date'], df_selected['Close'], label='Close', color='blue')
    ax2[0].plot(df_selected['Date'], df_selected['BB_Upper'], label='Upper Band', linestyle='--', color='red')
    ax2[0].plot(df_selected['Date'], df_selected['BB_Lower'], label='Lower Band', linestyle='--', color='green')
    ax2[0].legend()
    ax2[0].set_title("Bollinger Bands")

    # RSI
    ax2[1].plot(df_selected['Date'], df_selected['RSI'], color='purple')
    ax2[1].axhline(70, linestyle='--', color='red')
    ax2[1].axhline(30, linestyle='--', color='green')
    ax2[1].set_title("RSI Indicator")
    ax2[1].set_xlabel("Date")
    ax2[1].set_ylabel("RSI")

    st.pyplot(fig2)

# Tab 3: Model insights
with tab3:
    st.subheader("Model Performance")
    for name, result in model_results.items():
        st.write(f"{name} R² Score: {result['r2']}")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
        feat_df = feat_df.sort_values(by="Importance", ascending=False)
        st.bar_chart(feat_df.set_index("Feature"))

# Tab 4: Filterable data table
with tab4:
    st.subheader("Filtered Data Table")
    with st.expander("Add Date Filter"):
        start_date = st.date_input("Start Date", pd.to_datetime(df_selected['Date'].min()))
        end_date = st.date_input("End Date", pd.to_datetime(df_selected['Date'].max()))

    mask = (df_selected['Date'] >= pd.to_datetime(start_date)) & (df_selected['Date'] <= pd.to_datetime(end_date))
    filtered_df = df_selected.loc[mask]
    st.dataframe(filtered_df[['Date', 'Close', 'Predicted_Close', 'RSI', 'SMA_50', 'EMA_20', 'BB_Upper', 'BB_Lower']], use_container_width=True)

# Footer
st.markdown("""
---
This dashboard is for analysis and learning purposes. It does not constitute financial advice.
""")
