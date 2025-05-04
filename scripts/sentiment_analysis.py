# scripts/sentiment_analysis.py

import os
from transformers import pipeline
from utils.data_utils import fetch_latest_news

# Load the model once, enabling truncation of long inputs:
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if os.getenv("USE_CUDA") == "1" else -1,
)

def analyze_sentiment(texts):
    if not texts:
        return 0.0

    # Run in batches and truncate any inputs over 512 tokens
    results = sentiment_model(texts, truncation=True, max_length=512)
    sentiments = [1 if r["label"] == "POSITIVE" else -1 for r in results]
    scores     = [r["score"] for r in results]
    return sum(s * sc for s, sc in zip(sentiments, scores)) / len(scores)

def run_sentiment_analysis(config, stock_list):
    """
    config: dict with keys:
      - threshold: float
      - sources: list of ["newsapi","stocktwits","google"]
    stock_list: list of tickers
    """
    threshold = config.get("threshold", 0.1)
    sources   = config.get("sources", ["newsapi"])
    selected  = {}

    print(f"[INFO] Sentiment threshold = {threshold}, sources = {sources}")

    for symbol in stock_list:
        texts = fetch_latest_news(sources, symbol)
        if not texts:
            print(f"[WARN] No posts for {symbol}")
            continue

        score = analyze_sentiment(texts)
        print(f"{symbol} sentiment: {score:.3f}")
        if score >= threshold:
            selected[symbol] = score
            if len(selected) >= 4:
                break

    if not selected:
        print("[WARN] No stocks passed sentiment threshold this cycle.")
    return selected
