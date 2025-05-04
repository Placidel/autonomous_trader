import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def get_technical_indicators(df):
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    df = df.dropna()
    return df[['MA_10', 'MA_50', 'RSI']]

def compute_rsi(series, window):
    delta = series.diff(1).dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def train_stock_selector(stocks, sentiment_scores):
    feature_list, label_list = [], []

    for stock in stocks:
        df = yf.download(stock, period='1y', auto_adjust=True)
        if df.empty:
            print(f"[WARN] No data for {stock}, skipping.")
            continue

        # Keep only Close and rename
        df = df[['Close']].rename(columns={'Close': 'Close'})

        # Compute indicators
        indicators = get_technical_indicators(df)
        if indicators.empty:
            print(f"[WARN] Not enough data for indicators on {stock}, skipping.")
            continue

        # Re-insert Close into indicators so we can compare
        indicators['Close'] = df['Close'].loc[indicators.index]

        # Add sentiment and future price
        sentiment = sentiment_scores.get(stock, 0)
        indicators['Sentiment'] = sentiment
        indicators['Future_Close'] = indicators['Close'].shift(-5)

        # Drop rows with any NaN
        indicators = indicators.dropna()
        if indicators.empty:
            continue

        # Create training target
        indicators['Target'] = (indicators['Future_Close'] > indicators['Close']).astype(int)

        # Collect features and labels
        X = indicators[['MA_10', 'MA_50', 'RSI', 'Sentiment']]
        y = indicators['Target']

        feature_list.append(X)
        label_list.append(y)

    if not feature_list:
        raise ValueError("No valid training data found for any stock.")

    # Combine and train
    X_all = pd.concat(feature_list)
    y_all = pd.concat(label_list)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_scaled, y_all)

    return model, scaler

def predict_top_stocks(model, scaler, stocks, sentiment_scores):
    stock_scores = {}

    for stock in stocks:
        df = yf.download(stock, period='3mo', auto_adjust=True)
        if df.empty:
            print(f"[WARN] No recent data for {stock}, skipping.")
            continue

        df = df[['Close']].rename(columns={'Close': 'Close'})
        indicators = get_technical_indicators(df)
        if indicators.empty:
            print(f"[WARN] Insufficient data for indicators on {stock}, skipping.")
            continue

        # Extract Close into indicators if needed (predict only uses indicators + sentiment)
        last = indicators.iloc[-1]
        features = [last['MA_10'], last['MA_50'], last['RSI'], sentiment_scores.get(stock, 0)]

        prob = model.predict_proba(scaler.transform([features]))[0][1]
        stock_scores[stock] = prob

    if not stock_scores:
        raise ValueError("No stocks could be scored for prediction.")

    ranked = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
    print("Ranked Stocks (Highest potential first):")
    for sym, p in ranked:
        print(f"{sym}: {p:.2f}")
    return [sym for sym, _ in ranked]

def run_stock_selector(sentiment_scores, stocks):
    model, scaler = train_stock_selector(stocks, sentiment_scores)
    return predict_top_stocks(model, scaler, stocks, sentiment_scores)
