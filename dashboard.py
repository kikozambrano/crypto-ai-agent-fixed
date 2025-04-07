import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

client = Client()

# === Settings ===
st.sidebar.title("🔧 Settings")
symbols = {
    "Bitcoin (BTC)": "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT",
    "Solana (SOL)": "SOLUSDT"
}
symbol_name = st.sidebar.selectbox("Crypto Asset", list(symbols.keys()))
symbol = symbols[symbol_name]

days = st.sidebar.selectbox("Lookback Period", ["1", "7", "14", "30", "90", "180", "365"], index=3)
short_ma = st.sidebar.slider("Short MA", 2, 50, 10)
long_ma = st.sidebar.slider("Long MA", 5, 100, 30)
refresh_rate = st.sidebar.slider("Auto-Refresh Every (seconds)", 10, 300, 60)

# Auto-refresh
st_autorefresh(interval=refresh_rate * 1000, key="refresh")

# === Data Fetching ===
def fetch_data(coin_id, days):
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df[["time", "price"]]

# === Add Technical Indicators ===
def add_indicators(df):
    df = df.copy()
    df["short_ma"] = df["price"].rolling(window=short_ma).mean()
    df["long_ma"] = df["price"].rolling(window=long_ma).mean()
    df["rsi"] = ta.momentum.RSIIndicator(df["price"]).rsi()
    df["ema_20"] = ta.trend.EMAIndicator(df["price"], window=20).ema_indicator()
    df["macd_diff"] = ta.trend.MACD(df["price"]).macd_diff()
    try:
        df["stoch_rsi"] = ta.momentum.StochRSIIndicator(df["price"]).stochrsi()
    except:
        df["stoch_rsi"] = np.nan
    bb = ta.volatility.BollingerBands(df["price"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    return df

# === Signal Logic ===
def generate_signal(df):
    if df["short_ma"].iloc[-1] > df["long_ma"].iloc[-1]:
        return "BUY"
    elif df["short_ma"].iloc[-1] < df["long_ma"].iloc[-1]:
        return "SELL"
    else:
        return "HOLD"

# === ML Labeling ===
def add_target_label(df, lookahead=1):
    df = df.copy()
    df["target"] = df["price"].shift(-lookahead)
    return df.dropna()

def train_model(df):
    features = ["rsi", "macd_diff", "short_ma", "long_ma", "ema_20", "stoch_rsi"]
    df = df.dropna(subset=features + ["target"])
    if df.empty:
        return None
    X = df[features]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# === Main App ===
df = fetch_data(coin_id, days)
df = add_indicators(df)
signal = generate_signal(df)

ml_signal = "N/A"
predicted_price = np.nan
expected_return = np.nan

if len(df) >= 50:
    try:
        df = add_target_label(df)
        model = train_model(df)
        latest = df.dropna().iloc[-1:][["rsi", "macd_diff", "short_ma", "long_ma", "ema_20", "stoch_rsi"]]
        latest = latest.dropna()
        if not latest.empty:
            ml_prediction = model.predict(latest)[0]
            ml_signal = "BUY" if ml_prediction == 1 else "SELL"
            st.write("📊 ML raw prediction:", ml_prediction)
            st.write("📊 ML final signal:", ml_signal)
        else:
            st.warning("ML input row has NaN values. Cannot predict.")
    except Exception as e:
        st.error(f"ML prediction failed: {e}")
else:
    st.warning("Not enough data to train ML model. Use a longer lookback period.")

# === UI ===
st.title(f"📈 ML + Technical Signal for {symbol_name}")
st.subheader(f"📌 MA Signal: `{signal}`")
st.subheader(f"🤖 ML Prediction: `{ml_signal}`")
if not np.isnan(predicted_price):
    st.subheader(f"🎯 ML Target Price: ${predicted_price:,.2f}")
if not np.isnan(expected_return):
    st.subheader(f"📈 Expected Return: {expected_return:.2f}%")

st.subheader("📊 Price + Moving Averages")
st.line_chart(df.set_index("time")[["price", "short_ma", "long_ma"]])

st.subheader("📈 RSI")
if df["rsi"].notna().sum() > 0:
    st.line_chart(df.set_index("time")[["rsi"]])
else:
    st.warning("⚠️ RSI not available.")

st.subheader("📉 MACD")
if df["macd_diff"].notna().sum() > 0:
    st.line_chart(df.set_index("time")[["macd_diff"]])
else:
    st.warning("⚠️ MACD not available.")

st.subheader("🎯 Bollinger Bands")
if df[["bb_upper", "bb_lower"]].notna().sum().sum() > 0:
    st.line_chart(df.set_index("time")[["bb_upper", "price", "bb_lower"]])
else:
    st.warning("⚠️ Bollinger Bands not available.")

st.subheader("🌀 Stochastic RSI")
if df["stoch_rsi"].notna().sum() > 0:
    st.line_chart(df.set_index("time")[["stoch_rsi"]])
else:
    st.warning("⚠️ Stochastic RSI not available.")

st.subheader("⚡ EMA (20)")
st.line_chart(df.set_index("time")[["price", "ema_20"]])
