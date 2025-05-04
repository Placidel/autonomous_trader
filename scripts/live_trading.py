import pandas as pd
import numpy as np
import os
from stable_baselines3 import PPO
from scripts.trading_env import TradingEnv
from utils.data_utils import download_stock_data
from utils.alpaca_connector import submit_order

# 🔵 Simulation control flag
simulate = True  # Set to False to send real orders via Alpaca

def prepare_data(stock_symbol):
    df = download_stock_data([stock_symbol], period="6mo")

    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close'][stock_symbol].to_frame(name='Close')
    elif stock_symbol in df.columns:
        df = df[[stock_symbol]].rename(columns={stock_symbol: 'Close'})
    else:
        df = df[['Close']]

    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    df = df.dropna()

    return df

def compute_rsi(series, window=14):
    delta = series.diff(1).dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def run_live_trading(stock_symbols):
    portfolio = {symbol: {'cash': 500, 'shares': 0} for symbol in stock_symbols}
    logs = []

    for stock in stock_symbols:
        model_path = f"models/trading_agent_{stock}.zip"
        if not os.path.exists(model_path):
            print(f"Model not found for {stock}, skipping...")
            continue

        model = PPO.load(model_path)

        df = prepare_data(stock)
        today_data = df.iloc[-1:]  # latest day data
        env = TradingEnv(today_data)

        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)

        # Map action integer
        action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
        decision = action_map[int(action)]

        price = today_data['Close'].values[0]

        if simulate:
            # Simulation mode
            if decision == "Buy" and portfolio[stock]['cash'] >= price:
                portfolio[stock]['cash'] -= price
                portfolio[stock]['shares'] += 1
            elif decision == "Sell" and portfolio[stock]['shares'] > 0:
                portfolio[stock]['cash'] += price
                portfolio[stock]['shares'] -= 1
        else:
            # Real trading mode (Alpaca)
            if decision == "Buy":
                submit_order(stock, qty=1, side="buy")
            elif decision == "Sell":
                submit_order(stock, qty=1, side="sell")

        total_value = portfolio[stock]['cash'] + portfolio[stock]['shares'] * price

        log_entry = {
            'stock': stock,
            'decision': decision,
            'price': price,
            'cash': portfolio[stock]['cash'],
            'shares': portfolio[stock]['shares'],
            'total_value': total_value
        }
        logs.append(log_entry)

        print(f"Stock: {stock}, Action: {decision}, Cash: {portfolio[stock]['cash']:.2f}, Shares: {portfolio[stock]['shares']}, Total: {total_value:.2f}")

    # Save daily log
    log_df = pd.DataFrame(logs)
    os.makedirs("logs", exist_ok=True)
    log_df.to_csv("logs/live_trading_log.csv", index=False)
    print("Daily trading log saved to logs/live_trading_log.csv")

if __name__ == "__main__":
    stock_symbols = ['APA', 'APO']  # Replace with your current selected stocks
    run_live_trading(stock_symbols)
