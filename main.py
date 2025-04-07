from binance.client import Client
import numpy as np
import time

client = Client()

def get_prices(symbol='BTCUSDT', interval='1m', limit=100):
    klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=limit)
    return [float(kline[4]) for kline in klines]

def generate_signal(prices, short_window=10, long_window=30):
    short_ma = np.mean(prices[-short_window:])
    long_ma = np.mean(prices[-long_window:])
    
    if short_ma > long_ma:
        return "BUY"
    elif short_ma < long_ma:
        return "SELL"
    else:
        return "HOLD"

while True:
    try:
        prices = get_prices()
        signal = generate_signal(prices)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Signal: {signal}")
        time.sleep(60)
    except Exception as e:
        print("Error:", e)
        time.sleep(60)

