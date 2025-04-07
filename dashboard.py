import streamlit as st
from streamlit_autorefresh import st_autorefresh
from pycoingecko import CoinGeckoAPI
import pandas as pd
import numpy as np
import ta

# Initialize API
cg = CoinGeckoAPI()

# Sidebar Controls
st.sidebar.title("🔧 Settings")
coins = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Solana (SOL)": "solana"
}
coin_name = st.sidebar.selectbox("Crypto Asset", list(coins.keys()))
coin_id = coins[coin_name]

days = st.sidebar.selectbox("Lookback Period", ["1", "7", "14", "30", "90", "180", "365"], index=1)
short_ma = st.sidebar.slider("Short MA Window", 2, 50, 10)
long_ma = st.sidebar.slider("Long MA Window", 5, 100, 30)
refresh_rate = st.sidebar.slider("Auto-Refresh Every (seconds)", 10, 300, 60)

# Auto-refresh
st_autorefresh(interval=refresh_rate * 1000, key="refresh")

# Fetch Data from CoinGecko
def fetch_data(coin_id, days):
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
    prices = pd.DataFrame(data['prices'], columns=["timestamp", "price"])
    prices["time"] = pd.to_datetime(prices["timestamp"], unit='ms')
    return prices[["time", "price"]]

# Add Technical Indicators
def add_indicators(df):
    df = df.copy()
    df["short_ma"] = df["price"].rolling(window=short_ma).mean()
    df["long_ma"] = df["price"].rolling(window=long_ma).mean()
    df["rsi"] = ta.momentum.RSIIndicator(df["price"]).rsi()
    macd = ta.trend.MACD(df["price"])
    df["macd_diff"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["price"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    return df

# Generate basic signal
def generate_signal(df):
    if df["short_ma"].iloc[-1] > df["long_ma"].iloc[-1]:
        return "BUY"
    elif df["short_ma"].iloc[-1] < df["long_ma"].iloc[-1]:
        return "SELL"
    else:
        return "HOLD"

# Main Logic
df = fetch_data(coin_id, days)
df = add_indicators(df)
signal = generate_signal(df)

# UI Output
st.title(f"📈 Signal for {coin_name}")
st.subheader(f"📌 Current Signal: `{signal}`")

# Charts
st.line_chart(df.set_index("time")[["price", "short_ma", "long_ma"]])
st.line_chart(df.set_index("time")[["rsi"]])
st.line_chart(df.set_index("time")[["macd_diff"]])
st.line_chart(df.set_index("time")[["bb_upper", "price", "bb_lower"]])
