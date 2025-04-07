import streamlit as st
from streamlit_autorefresh import st_autorefresh
from pycoingecko import CoinGeckoAPI
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# === Settings ===
cg = CoinGeckoAPI()

st.sidebar.title("🔧 Settings")
coins = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Solana (SOL)": "solana"
}
coin_name = st.sidebar.selectbox("Crypto Asset", list(coins.keys()))
coin_id = coins[coin_name]

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
    df["future_price"] = df["price"].shift(-lookahead)
    df["target"] = np.where(df["future_price"] > df["price"], 1, 0)
    return df.dropna()

def train_model(df):
    features = ["rsi", "macd_diff", "short_ma", "long_ma", "ema_20", "stoch_rsi"]
    df = df.dropna(subset=features + ["target"])
    X = df[features]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# === Main App ===
df = fetch_data(coin_id, days)
df = add_indicators(df)
signal = generate_signal(df)

ml_signal = "N/A"
if len(df) >= 50:
    try:
        df = add_target_label(df)
        model = train_model(df)
        latest = df.dropna().iloc[-1:][["rsi", "macd_diff", "short_ma", "long_ma", "ema_20", "stoch_rsi"]]
        latest = latest.dropna()
        if not latest.empty:
            ml_prediction = model.predict(latest)[0]
            ml_signal = "BUY" if ml_prediction == 1 else "SELL"

    # Friendly explanation
    rsi_val = latest["rsi"].values[0]
    macd_val = latest["macd_diff"].values[0]
    sma = latest["short_ma"].values[0]
    lma = latest["long_ma"].values[0]

    explanation = []
    if rsi_val < 30:
        explanation.append(f"🟢 RSI is low ({rsi_val:.1f}) → asset may be oversold")
    elif rsi_val > 70:
        explanation.append(f"🔴 RSI is high ({rsi_val:.1f}) → asset may be overbought")
    else:
        explanation.append(f"ℹ️ RSI is moderate ({rsi_val:.1f})")

    if macd_val > 0:
        explanation.append(f"🟢 MACD is positive ({macd_val:.4f}) → upward momentum")
    else:
        explanation.append(f"🔴 MACD is negative ({macd_val:.4f}) → downward pressure")

    if sma > lma:
        explanation.append(f"🟢 Short MA ({sma:.0f}) > Long MA ({lma:.0f}) → bullish crossover")
    else:
        explanation.append(f"🔴 Short MA ({sma:.0f}) < Long MA ({lma:.0f}) → bearish crossover")
    
            st.write("📊 ML raw prediction:", ml_prediction)
            st.write("📊 ML final signal:", ml_signal)
        else:
            st.warning("ML input row has NaN values. Cannot predict.")
    except Exception as e:
        st.error(f"ML prediction failed: {e}")
else:
    st.warning("Not enough data to train ML model. Use a longer lookback period.")

# === Streamlit UI ===
st.title(f"📈 ML + Technical Signal for {coin_name}")
st.subheader(f"📌 MA Signal: `{signal}`")
st.subheader(f"🤖 ML Prediction: `{ml_signal}`")
for reason in explanation:
    st.write(reason)

st.subheader("📊 Price + Moving Averages")
st.line_chart(df.set_index("time")[["price", "short_ma", "long_ma"]])

st.subheader("📈 RSI")
st.line_chart(df.set_index("time")[["rsi"]])

st.subheader("📉 MACD")
st.line_chart(df.set_index("time")[["macd_diff"]])

st.subheader("🎯 Bollinger Bands")
st.line_chart(df.set_index("time")[["bb_upper", "price", "bb_lower"]])

st.subheader("🌀 Stochastic RSI")
st.line_chart(df.set_index("time")[["stoch_rsi"]])

st.subheader("⚡ EMA (20)")
st.line_chart(df.set_index("time")[["price", "ema_20"]])
