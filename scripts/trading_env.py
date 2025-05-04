import gym
import numpy as np
from gym import spaces
import pandas as pd


class TradingEnv(gym.Env):
    def __init__(self, data, initial_cash=500):
        super(TradingEnv, self).__init__()
        self.data = data
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.stock_held = 0
        self.current_step = 0
        self.max_steps = len(data) - 1

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observations: [close_price, 10-day MA, 50-day MA, RSI, cash, stock_held]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32
        )

    def _next_observation(self):
        row = self.data.iloc[self.current_step]

        # Safely extract each value and force float
        close_price = float(row['Close']) if not pd.isna(row['Close']) else 0.0
        ma_10 = float(row['MA_10']) if not pd.isna(row['MA_10']) else 0.0
        ma_50 = float(row['MA_50']) if not pd.isna(row['MA_50']) else 0.0
        rsi = float(row['RSI']) if not pd.isna(row['RSI']) else 0.0

        obs = np.array([
            close_price,
            ma_10,
            ma_50,
            rsi,
            self.cash,
            self.stock_held
        ], dtype=np.float32)

        return obs

    def reset(self):
        self.cash = self.initial_cash
        self.stock_held = 0
        self.current_step = 0
        return self._next_observation()

    def step(self, action):
        row = self.data.iloc[self.current_step]
        price = row['Close']

        if action == 1 and self.cash >= price:  # Buy
            self.stock_held += 1
            self.cash -= price
        elif action == 2 and self.stock_held > 0:  # Sell
            self.stock_held -= 1
            self.cash += price
        # else: Hold (do nothing)

        self.current_step += 1
        done = self.current_step >= self.max_steps

        next_obs = self._next_observation()
        portfolio_value = self.cash + self.stock_held * price
        reward = portfolio_value - self.initial_cash  # Profit relative to start
        info = {"portfolio_value": portfolio_value}

        return next_obs, reward, done, info
