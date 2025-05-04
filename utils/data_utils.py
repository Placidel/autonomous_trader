import os
import time
import requests
import pandas as pd
import yfinance as yf
import feedparser
from dotenv import load_dotenv

load_dotenv()

# ———————— Market Data Utility ————————

def download_stock_data(symbols, period="1y"):
    """
    Download historical price data for given symbols.
    Returns a DataFrame of adjusted close prices.
    """
    df = yf.download(symbols, period=period, auto_adjust=True, progress=False)
    if isinstance(symbols, list) and len(symbols) > 1:
        return df['Close']
    series = df['Close'] if 'Close' in df else df
    return series.to_frame(name='Close')


# ———————— News / Sentiment Utilities ————————

def fetch_newsapi(ticker, api_key):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&language=en&sortBy=publishedAt&apiKey={api_key}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        return [
            f"{a['title']}. {a.get('description','')}"
            for a in articles if a.get("description")
        ]
    except Exception as e:
        print(f"[ERROR] NewsAPI for {ticker}: {e}")
        return []

def fetch_stocktwits(ticker):
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code != 200:
            print(f"[WARN] StockTwits returned status {r.status_code} for {ticker}")
            return []
        data = r.json()
        return [m["body"] for m in data.get("messages", []) if m.get("body")]
    except Exception as e:
        print(f"[ERROR] StockTwits for {ticker}: {e}")
        return []

def fetch_google_news(ticker):
    """
    Fetch the latest headlines from Google News RSS for the ticker.
    """
    rss_url = (
        f"https://news.google.com/rss/search?"
        f"q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        feed = feedparser.parse(rss_url)
        items = feed.get("entries", [])[:20]
        return [f"{item.title}. {item.get('summary','')}" for item in items]
    except Exception as e:
        print(f"[ERROR] Google News RSS for {ticker}: {e}")
        return []

def fetch_latest_news(sources, ticker):
    """
    sources: list of strings, e.g. ["newsapi","stocktwits","google"]
    ticker: stock symbol to search
    """
    api_key = os.getenv("NEWSAPI_KEY")
    texts = []

    if "newsapi" in sources and api_key:
        texts.extend(fetch_newsapi(ticker, api_key))
    if "stocktwits" in sources:
        texts.extend(fetch_stocktwits(ticker))
    if "google" in sources:
        texts.extend(fetch_google_news(ticker))

    return texts
