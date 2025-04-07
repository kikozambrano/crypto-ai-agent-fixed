import streamlit as st
from streamlit_autorefresh import st_autorefresh
from pycoingecko import CoinGeckoAPI
import pandas as pd
import numpy as np
import datetime

cg = CoinGeckoAPI()

# Sidebar
st.sidebar.title("🔧 Settings")
symbol_map = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Solana (SOL)": "solana"
}
symbol_label = st.sidebar.selectbox("Choose Asset", list(symbol_map.keys()))
symbol = symbol_map[symbol_label]

interval_map = {
    "1 Day": 1,
    "7 Days": 7,
    "14 Days": 14,
    "30 Days": 30,
}
interval_label = st.sidebar.selectbox("Interval", list(interval_map.keys()))
days = interval_map[interval_label]

short_window = st.sidebar.slider("Short MA Window", 3, 30, 10)
long_window = st.sidebar.slider("Long MA Window", 10, 100, 30)
refresh_rate = st.sidebar.slider("Refresh Every (seconds)", 10, 300, 60)

# Auto-refresh
st_autorefresh(interval=refresh_rate * 1000, key="refresh")

# Fetch prices
def fetch_prices(symbol, days):
    data = cg.get_coin_market_chart_by_id(id=symbol, vs_currency='usd', days=days)
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df[['time', 'price']]

def generate_signal(df, short_window, long_window):
    df['short_ma'] = df['price'].rolling(window=short_window).mean()
    df['long_ma'] = df['price'].rolling(window=long_window).mean()
    if df['short_ma'].iloc[-1] > df['long_ma'].iloc[-1]:
        return "BUY"
    elif df['short_ma'].iloc[-1] < df['long_ma'].iloc[-1]:
        return "SELL"
    else:
        return "HOLD"

df = fetch_prices(symbol, days)
signal = generate_signal(df, short_window, long_window)

# Display
st.title(f"📈 Live Crypto Signal for {symbol_label}")
st.subheader(f"Signal: {signal}")
st.line_chart(df.set_index('time')['price'])
