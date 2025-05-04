**Autonomous Trader AI**

An end-to-end autonomous trading system that uses AI agents to:

1. **Discover AI/tech stocks** via Alpaca
2. **Fetch multi-source news & social media** (NewsAPI, StockTwits, Google News RSS)
3. **Analyze sentiment** with DistilBERT
4. **Select top candidates** via a RandomForest + technical indicators (MA10, MA50, RSI)
5. **Train a PPO reinforcement-learning agent** to trade (Buy/Sell/Hold)
6. **Simulate or execute** daily trades via Alpaca Paper API
7. **Record all results** with timestamps for back-testing and evaluation

---

## 🚀 Features

* **Dynamic stock universe**: pulls all active US‐listed tickers, filters AI/tech by keyword
* **Multi-source sentiment**: NewsAPI, StockTwits, Google News RSS
* **Technical + sentiment ranking**: MA10, MA50, RSI + news sentiment
* **Reinforcement-learning agent**: Stable-Baselines3 PPO
* **Simulation & paper-trading**: toggle between virtual portfolio and Alpaca Paper API
* **Continuous pipeline**: hourly loop in Docker container, full logging

---

## 📋 Prerequisites

* **Python 3.11+**
* **pip**
* **Docker & Docker Compose** (for 24/7 deployment)
* Accounts & API keys for:

  * [Alpaca Paper Trading](https://alpaca.markets/)
  * [NewsAPI](https://newsapi.org/)

---

## 🔧 Installation

```bash
git clone https://github.com/your-org/autonomous_trader.git
cd autonomous_trader
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 Configuration

Create a `.env` file in the project root with your credentials:

```dotenv
# Alpaca Paper Trading API
ALPACA_API_KEY=YOUR_ALPACA_API KEY
ALPACA_SECRET_KEY=YOUR_ALPACA_SECRECT_KEY

# NewsAPI for news headlines
NEWSAPI_KEY=YOUR_NEWSAPI_KEY

# (Optional) Enable GPU
USE_CUDA=0
```

---

## 📁 Project Structure

```
.
├── .env
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── run_pipeline.py        # Continuous loop entrypoint
├── main.py                # One-off pipeline runner
├── config/
│   └── settings.json      # (optional extra config)
├── data/
│   └── ai_tech_stock_list.csv
├── logs/
│   ├── live_trading_log.csv
│   └── sentiment_log.csv
├── models/
│   └── trading_agent_<TICKER>.zip
├── scripts/
│   ├── get_ai_stocks.py
│   ├── sentiment_analysis.py
│   ├── stock_selector.py
│   ├── record_analysis.py
│   ├── train_trading_agent.py
│   ├── live_trading.py
│   └── trading_env.py
└── utils/
    ├── data_utils.py
    └── alpaca_connector.py
```

---

## ⚙️ Usage

### 1. One-off run

Run the full pipeline once (fetch stocks → sentiment → select → trade):

```bash
python main.py
```

### 2. Continuous continuous loop

Start the hourly loop locally:

```bash
python run_pipeline.py
```

* The script will:

  1. Fetch and update the AI/tech stock universe
  2. Fetch news & social posts per ticker
  3. Analyze sentiment
  4. Rank and select top tickers
  5. Append timestamped sentiment scores to `logs/sentiment_log.csv`
  6. Train/update the RL trading agent
  7. Simulate or place paper trades via Alpaca
  8. Append trade records to `logs/live_trading_log.csv`
  9. Sleep for 1 hour, then repeat

### 3. Docker deployment

Build and run the containerized pipeline:

```bash
docker-compose build
docker-compose up -d
```

* Container runs `run_pipeline.py`, with volumes:

  * `./data` persists stock lists
  * `./logs` persists logs
  * Environment variables loaded from `.env`

---

## 📊 Logs & Data

* **`logs/sentiment_log.csv`** — timestamped sentiment scores
* **`logs/live_trading_log.csv`** — daily trade records
* **`data/ai_tech_stock_list.csv`** — current ticker universe

---

## 🔍 Next Steps

* Add **risk management** (stop-loss, position sizing)
* Tune **RL hyperparameters** or try alternative algorithms
* Build a **dashboard** (Streamlit/React) for live monitoring
* Integrate **back-testing framework** (Backtrader)

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Commit & push
4. Open a pull request

Please adhere to the code style and add tests for new features.

---

## 📜 License

Released under the **MIT License**. See [LICENSE](LICENSE) for details.
