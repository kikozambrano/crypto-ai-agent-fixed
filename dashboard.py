# Initialization of Binance Client
client = Client(api_key, api_secret)

# Correcting the variable name in the title
st.title(f"📈 ML + Technical Signal for {symbol_name}")

# Improved exception handling for ML prediction
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
except ValueError as ve:
    st.error(f"Value Error in ML prediction: {ve}")
except Exception as e:
    st.error(f"ML prediction failed: {e}")
