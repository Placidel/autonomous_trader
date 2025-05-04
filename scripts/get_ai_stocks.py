import os
import pandas as pd
from alpaca_trade_api.rest import REST
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

def fetch_ai_stocks():
    api = REST(API_KEY, API_SECRET, BASE_URL)

    try:
        assets = api.list_assets(status="active")
    except Exception as e:
        print(f"Error retrieving assets: {e}")
        return []

    ai_keywords = [
        "AI", "Artificial", "Machine Learning", "Neural", "Deep", "Cognitive",
        "Automation", "Autonomous", "Vision", "Robot", "Speech", "NLP", "Language", "Semiconductor"
    ]

    matching_symbols = set()

    for asset in assets:
        description = asset.name.lower() if asset.name else ""
        symbol = asset.symbol.upper()

        if any(keyword.lower() in description for keyword in ai_keywords):
            matching_symbols.add(symbol)

    if not os.path.exists("data"):
        os.makedirs("data")

    output_path = "data/ai_tech_stock_list.csv"
    pd.Series(sorted(matching_symbols)).to_csv(output_path, index=False, header=["symbol"])
    print(f"Saved {len(matching_symbols)} AI/tech stocks to {output_path}")

    return list(matching_symbols)
