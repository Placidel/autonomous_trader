#!/usr/bin/env python3

import os
import pandas as pd
from dotenv import load_dotenv

from scripts.sentiment_analysis import run_sentiment_analysis
from scripts.stock_selector import run_stock_selector
from scripts.live_trading import run_live_trading

load_dotenv()

# Configuration
SENTIMENT_THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD", 0.1))
SENTIMENT_SOURCES   = ["newsapi", "stocktwits", "google"]
SIMULATE_TRADING    = os.getenv("SIMULATE_TRADING", "true").lower() in ("1", "true", "yes")

def main():
    # 1. Load AI/tech stock list
    df = pd.read_csv("data/ai_tech_stock_list.csv")
    # CSV header is "symbol"
    stock_list = df["symbol"].dropna().tolist()

    print(f"Evaluating {len(stock_list)} AI/tech stocks...")

    # 2. Run multi-source sentiment analysis
    sentiment_config = {
        "threshold": SENTIMENT_THRESHOLD,
        "sources": SENTIMENT_SOURCES
    }
    sentiment_scores = run_sentiment_analysis(sentiment_config, stock_list)
    if not sentiment_scores:
        print("No stocks passed sentiment threshold. Exiting.")
        return

    # 3. Rank & select stocks
    stocks_to_rank = list(sentiment_scores.keys())
    try:
        selected = run_stock_selector(sentiment_scores, stocks_to_rank)
    except Exception as e:
        print(f"No valid training data: {e}. Exiting.")
        return
    if not selected:
        print("No stocks selected for trading. Exiting.")
        return

    print(f"Selected stocks for trading: {selected}")

    # 4. Execute live or simulated trades
    run_live_trading(selected, simulate=SIMULATE_TRADING)

if __name__ == "__main__":
    main()
