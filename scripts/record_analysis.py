import os
import csv
from datetime import datetime

def save_analysis_to_log(sentiment_scores, log_path="logs/sentiment_log.csv"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fieldnames = ["timestamp", "symbol", "sentiment_score"]
    timestamp = datetime.utcnow().isoformat()

    rows = [
        {
            "timestamp": timestamp,
            "symbol": symbol,
            "sentiment_score": score
        }
        for symbol, score in sentiment_scores.items()
    ]

    write_header = not os.path.exists(log_path)

    with open(log_path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Logged sentiment analysis for {len(rows)} stocks.")
