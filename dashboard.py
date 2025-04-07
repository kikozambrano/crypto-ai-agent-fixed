import streamlit as st
from binance.client import Client
import pandas as pd
import numpy as np
from streamlit_autorefresh import st_autorefresh

# Binance client
client = Client()

# Sidebar controls
st.sidebar.title("🔧 Settings")
symbol = st.sidebar.selectbox("Choose Asset", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d", "1w"])
short_window = st.sidebar.slider("Short MA", 5, 30, 10)
long_window = st.sidebar.slider("Long MA", 10, 100, 30)
refresh_rate = st.sidebar.slider("Refresh Every (seconds)", 10, 300, 60)

# Auto-refresh the page every X seconds
st_autorefresh(interval=refresh_rate * 1000, key="refresh")

# Fetch prices
def fetch_prices(symbol, interval, limit=100):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['close'] = df['close'].astype(float)
    return df[['time', 'close']]

# Generate signal
def generate_signal(df, short_window, long_window):
    df['short_ma'] = df['close'].rolling(window=short_window).mean()
    df['long_ma'] = df['close'].rolling(window=long_window).mean()
    if df['short_ma'].iloc[-1] > df['long_ma'].iloc[-1]:
        return "BUY"
    elif df['short_ma'].iloc[-1] < df['long_ma'].iloc[-1]:
        return "SELL"
    else:
        return "HOLD"

# Run app
df = fetch_prices(symbol, interval)
signal = generate_signal(df, short_window, long_window)

st.title(f"📈 Live Crypto Signal for {symbol}")
st.subheader(f"Current Signal: {signal}")
st.line_chart(df.set_index('time')['close'])
