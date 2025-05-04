# Autonomous Trader AI

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

- **Dynamic stock universe**: pulls all active US‐listed tickers, filters AI/tech by keyword  
- **Multi-source sentiment**: NewsAPI, StockTwits, Google News RSS  
- **Technical + sentiment ranking**: MA10, MA50, RSI + news sentiment  
- **Reinforcement-learning agent**: Stable-Baselines3 PPO  
- **Simulation & paper-trading**: toggle between virtual portfolio and Alpaca Paper API  
- **Continuous pipeline**: hourly loop in Docker container, full logging  

---

## 📋 Prerequisites

- **Python 3.11+**  
- **pip**  
- **Docker & Docker Compose** (for 24/7 deployment)  
- Accounts & API keys for:  
  - [Alpaca Paper Trading](https://alpaca.markets/)  
  - [NewsAPI](https://newsapi.org/)  

---

## 🔧 Installation

```bash
git clone https://github.com/your-org/autonomous_trader.git
cd autonomous_trader
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
