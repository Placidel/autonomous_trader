from stable_baselines3 import PPO
from scripts.trading_env import TradingEnv
from utils.data_utils import download_stock_data
import pandas as pd

def prepare_data(stock_symbol):
    df = download_stock_data([stock_symbol], period="1y")

    # Force a single clean column
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


def train_agent(stock_symbol):
    df = prepare_data(stock_symbol)
    env = TradingEnv(df)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save(f"models/trading_agent_{stock_symbol}")


if __name__ == "__main__":
    selected_stocks = ['APA', 'APO']  # From your last stock selector
    for stock in selected_stocks:
        train_agent(stock)
