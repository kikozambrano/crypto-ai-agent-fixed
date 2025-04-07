import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import ta
from pycoingecko import CoinGeckoAPI
from sklearn.ensemble import RandomForestRegressor
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

days = st.sidebar.selectbox("Lookback Period (days)", ["30", "90", "180", "365"], index=3)
short_ma = st.sidebar.slider("Short MA", 2, 50, 10)
long_ma = st.sidebar.slider("Long MA", 5, 100, 30)
refresh_rate = st.sidebar.slider("Auto-Refresh Every (seconds)", 10, 300, 60)

# Auto-refresh
st_autorefresh(interval=refresh_rate * 1000, key="refresh")

# === Data Fetching from CoinGecko ===
def fetch_data(coin_id="bitcoin", days="365"):
    try:
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df[["time", "price"]]
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame(columns=["time", "price"])

# === Add Technical Indicators ===
def add_indicators(df):
    df = df.copy()
    try:
        df["short_ma"] = df["price"].rolling(window=short_ma).mean()
    except Exception as e:
        df["short_ma"] = np.nan
        st.warning(f"⚠️ Failed to calculate Short MA: {e}")

    try:
        df["long_ma"] = df["price"].rolling(window=long_ma).mean()
    except Exception as e:
        df["long_ma"] = np.nan
        st.warning(f"⚠️ Failed to calculate Long MA: {e}")

    try:
        df["rsi"] = ta.rsi(df["price"])
    except Exception as e:
        df["rsi"] = np.nan
        st.warning(f"⚠️ Failed to calculate RSI: {e}")

    try:
        df["ema_20"] = ta.ema(df["price"], length=20)
    except Exception as e:
        df["ema_20"] = np.nan
        st.warning(f"⚠️ Failed to calculate EMA: {e}")

    try:
        macd = ta.macd(df["price"])
        df["macd_diff"] = macd["MACD_12_26_9"] - macd["MACDs_12_26_9"]
    except Exception as e:
        df["macd_diff"] = np.nan
        st.warning(f"⚠️ Failed to calculate MACD: {e}")

    try:
        stochrsi = ta.stochrsi(df["price"])
        df["stoch_rsi"] = stochrsi["STOCHRSIk_14_14_3_3"] if "STOCHRSIk_14_14_3_3" in stochrsi else np.nan
    except Exception as e:
        df["stoch_rsi"] = np.nan
        st.warning(f"⚠️ Failed to calculate Stochastic RSI: {e}")

    try:
        bb = ta.bbands(df["price"])
        df["bb_upper"] = bb["BBU_20_2.0"]
        df["bb_lower"] = bb["BBL_20_2.0"]
    except Exception as e:
        df["bb_upper"] = np.nan
        df["bb_lower"] = np.nan
        st.warning(f"⚠️ Failed to calculate Bollinger Bands: {e}")

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
if df.empty:
    st.stop()

df = add_indicators(df)
signal = generate_signal(df)

ml_signal = "N/A"
predicted_price = np.nan
expected_return = np.nan

if len(df) >= 50:
    try:
        df = add_target_label(df)
        model = train_model(df)
        if model:
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
    except Exception as e:
        st.error(f"ML prediction failed: {e}")
else:
    st.warning("Not enough data to train ML model. Increase lookback period.")

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
if df["ema_20"].notna().sum() > 0:
    st.line_chart(df.set_index("time")[["price", "ema_20"]])
else:
    st.warning("⚠️ EMA not available.")

# === Backtesting ===
if st.sidebar.checkbox("Run Backtest"):
    st.subheader("📈 Backtest Results")

    backtest_df = df.dropna().copy()
    backtest_df = add_target_label(backtest_df)
    backtest_df = backtest_df.dropna(subset=["rsi", "macd_diff", "short_ma", "long_ma", "ema_20", "stoch_rsi", "target"])

    features = ["rsi", "macd_diff", "short_ma", "long_ma", "ema_20", "stoch_rsi"]
    if backtest_df[features].empty:
        st.warning("⚠️ Not enough indicator data to run backtest.")
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(backtest_df[features], backtest_df["target"])

        initial_cash = 10000
        cash = initial_cash
        position = 0
        portfolio_values = []
        trade_log = []

        for i in range(len(backtest_df) - 1):
            row = backtest_df.iloc[i]
            next_price = backtest_df.iloc[i + 1]["price"]
            input_row = pd.DataFrame([row[features]])
            pred = model.predict(input_row)[0]

            if pred > row["price"] and position == 0:
                # BUY
                position = cash / row["price"]
                cash = 0
                trade_log.append({"date": row["time"], "action": "BUY", "price": row["price"]})
            elif pred < row["price"] and position > 0:
                # SELL
                cash = position * row["price"]
                position = 0
                trade_log.append({"date": row["time"], "action": "SELL", "price": row["price"]})

            portfolio_value = cash + position * row["price"]
            portfolio_values.append(portfolio_value)

        final_value = cash + position * backtest_df.iloc[-1]["price"]
        total_return = ((final_value - initial_cash) / initial_cash) * 100

        st.metric("💰 Final Portfolio Value", f"${final_value:,.2f}")
        st.metric("📈 Total Return", f"{total_return:.2f}%")

        chart_df = backtest_df.iloc[:len(portfolio_values)].copy()
        chart_df["Portfolio Value"] = portfolio_values
        st.line_chart(chart_df.set_index("time")[["price", "Portfolio Value"]])

        st.subheader("📋 Trade Log")
        if trade_log:
            st.dataframe(pd.DataFrame(trade_log))
        else:
            st.write("No trades executed.")
