# run_pipeline.py

import time
from dotenv import load_dotenv

from scripts.get_ai_stocks import fetch_ai_stocks
from scripts.sentiment_analysis import run_sentiment_analysis
from scripts.stock_selector import run_stock_selector
from scripts.record_analysis import save_analysis_to_log
from scripts.live_trading import run_live_trading

load_dotenv()

SENTIMENT_THRESHOLD = 0.1
CYCLE_INTERVAL_SECONDS = 60 * 60  # 1 hour

def main():
    while True:
        print("[INFO] Starting trading cycle...")

        # 1. Fetch updated AI/tech stock list
        stock_list = fetch_ai_stocks()
        if not stock_list:
            print("[WARN] No stocks retrieved. Skipping cycle.")
            time.sleep(CYCLE_INTERVAL_SECONDS)
            continue

        # 2. Run sentiment analysis across all sources
        sentiment_config = {
            "threshold": SENTIMENT_THRESHOLD,
            "sources": ["newsapi", "stocktwits", "google"]
        }
        sentiment_scores = run_sentiment_analysis(sentiment_config, stock_list)
        if not sentiment_scores:
            print("[WARN] No sentiment scores computed. Skipping cycle.")
            time.sleep(CYCLE_INTERVAL_SECONDS)
            continue

        # 3. Rank and select stocks (guard against no training data)
        stocks_to_rank = list(sentiment_scores.keys())
        try:
            selected_stocks = run_stock_selector(sentiment_scores, stocks_to_rank)
        except ValueError as e:
            print(f"[WARN] {e} Skipping trading this cycle.")
            time.sleep(CYCLE_INTERVAL_SECONDS)
            continue

        if not selected_stocks:
            print("[WARN] No stocks selected for trading. Skipping cycle.")
            time.sleep(CYCLE_INTERVAL_SECONDS)
            continue

        # 4. Record sentiment results
        save_analysis_to_log(sentiment_scores)

        # 5. Execute live or simulated trading
        run_live_trading(selected_stocks)

        print(f"[INFO] Cycle complete. Sleeping for {CYCLE_INTERVAL_SECONDS} seconds...\n")
        time.sleep(CYCLE_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
