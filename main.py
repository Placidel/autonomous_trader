import json
from scripts.sentiment_analysis import run_sentiment_analysis
from scripts.stock_selector import run_stock_selector
from utils.data_utils import fetch_sp500_symbols


def load_config(path='config/settings.json'):
    with open(path) as f:
        return json.load(f)


def main():
    config = load_config()
    all_stocks = fetch_sp500_symbols()

    sentiment_scores = run_sentiment_analysis(config['sentiment'], all_stocks)

    if sentiment_scores:
        # Pass only the selected stocks to stock selector
        selected_stocks = list(sentiment_scores.keys())
        ranked_stocks = run_stock_selector(sentiment_scores, selected_stocks)
        print("Final selected stocks:", ranked_stocks[:2])
    else:
        print("No stocks met the sentiment threshold today.")


if __name__ == "__main__":
    main()
