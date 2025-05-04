import json
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

def load_alpaca_keys():
    with open('config/alpaca_keys.json') as f:
        return json.load(f)

def get_trading_client():
    keys = load_alpaca_keys()
    client = TradingClient(
        api_key=keys['APCA_API_KEY_ID'],
        secret_key=keys['APCA_API_SECRET_KEY'],
        paper=True  # Paper trading
    )
    return client

def submit_order(symbol, qty, side):
    client = get_trading_client()

    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )

    order = client.submit_order(order_data)
    print(f"Submitted {side.upper()} order for {qty} share(s) of {symbol}.")
    return order
