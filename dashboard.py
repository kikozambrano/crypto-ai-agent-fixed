 import streamlit as st
 from streamlit_autorefresh import st_autorefresh
 from binance.client import Client
 from binance.enums import KLINE_INTERVAL_1MINUTE, KLINE_INTERVAL_1HOUR, KLINE_INTERVAL_1DAY
 import pandas as pd
 import numpy as np
 import ta
 from sklearn.ensemble import RandomForestRegressor
 from sklearn.model_selection import train_test_split
 
 # === Settings ===
 st.sidebar.title("🔧 Settings")
symbols = {
    "Bitcoin (BTC)": "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT",
    "Solana (SOL)": "SOLUSDT"
}
symbol_name = st.sidebar.selectbox("Crypto Asset", list(symbols.keys()))
symbol = symbols[symbol_name]

interval = st.sidebar.selectbox("Interval", ["1m", "1h", "1d"], index=1)
interval_map = {"1m": KLINE_INTERVAL_1MINUTE, "1h": KLINE_INTERVAL_1HOUR, "1d": KLINE_INTERVAL_1DAY}
binance_interval = interval_map[interval]

limit = st.sidebar.slider("Data Points (candles)", 100, 1000, 500)
short_ma = st.sidebar.slider("Short MA", 2, 50, 10)
long_ma = st.sidebar.slider("Long MA", 5, 100, 30)
refresh_rate = st.sidebar.slider("Auto-Refresh Every (seconds)", 10, 300, 60)

# Auto-refresh
st_autorefresh(interval=refresh_rate * 1000, key="refresh")

# === Data Fetching from Binance ===
def fetch_data(symbol, interval, limit):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])
    df["time"] = pd.to_datetime(df["close_time"], unit="ms")
    df["price"] = df["close"].astype(float)
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
     df["target"] = df["price"].shift(-1)
     return df.dropna()
 
 def train_model(df):
     features = ["rsi", "macd_diff", "short_ma", "long_ma", "ema_20", "stoch_rsi"]
     df = df.dropna(subset=features + ["target"])
     X = df[features]
     y = df["target"]
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
     model = RandomForestRegressor(n_estimators=100, random_state=42)
     model.fit(X_train, y_train)
     return model
 
 # === Main App ===
 df = fetch_data(symbol, binance_interval, limit)
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
            predicted_price = model.predict(latest)[0]
            latest_price = df["price"].iloc[-1]
            price_diff = predicted_price - latest_price
            expected_return = (price_diff / latest_price) * 100

            if price_diff > 0:
                ml_signal = "BUY"
            elif price_diff < 0:
                ml_signal = "SELL"
            else:
                ml_signal = "HOLD"

            st.write("📊 ML raw prediction:", f"${predicted_price:,.2f}")
            st.write("📊 ML final signal:", ml_signal)
        else:
            st.warning("ML input row has NaN values. Cannot predict.")
    except Exception as e:
        st.error(f"ML prediction failed: {e}")
else:
    st.warning("Not enough data to train ML model. Increase candle limit.")
 
 # === Streamlit UI ===
 st.title(f"📈 ML + Technical Signal for {coin_name}")
 st.subheader(f"📌 MA Signal: `{signal}`")
 st.subheader(f"🤖 ML Prediction: `{ml_signal}`")
 if not np.isnan(predicted_price):
     st.subheader(f"🎯 ML Target Price: ${predicted_price:,.2f}")
 if not np.isnan(expected_return):
     st.subheader(f"📈 Expected Return: {expected_return:.2f}%")
 
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
