# debug backtester
from backtester import Backtester, read_trader_log
from parse_log import parse_log
from datamodel import Listing
import trader_v2

Trader = trader_v2.Trader
Product = trader_v2.Product

def calculate_kelp_fair(order_depth):
    # assumes order_depth has orders in it 
    best_ask = min(order_depth.sell_orders.keys())
    best_bid = max(order_depth.buy_orders.keys())
    filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
    filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
    mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
    mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid

    mmmid_price = (mm_ask + mm_bid) / 2
    return mmmid_price
    
def calculate_resin_fair(order_depth):
    return 10000

listings = {
    'RAINFOREST_RESIN': Listing(symbol='RAINFOREST_RESIN', product='RAINFOREST_RESIN', denomination='SEASHELLS'),
    'KELP': Listing(symbol='KELP', product='KELP', denomination='SEASHELLS')
}

position_limit = {
    'RAINFOREST_RESIN': 20,
    'KELP': 20
}

fair_calculations = {
    "RAINFOREST_RESIN": calculate_resin_fair,
    "KELP": calculate_kelp_fair
}

market_data, trade_history = read_trader_log('tutorial/logs/empty_submission.log')

backtest_dir = 'tutorial/backtests/debug_backtester.log'

trader = Trader()
trader.params = {Product.KELP: {'take_width': 2, 'clear_width': 1, 'prevent_adverse': True, 'adverse_volume': 15, 'reversion_beta': -0.18172393033850867, 'disregard_edge': 1, 'join_edge': 0, 'default_edge': 1}}

backtester = Backtester(trader, listings, position_limit, fair_calculations, market_data, trade_history, backtest_dir)
backtester.run()
backtester.pnl
sandbox, market, trades = parse_log(backtest_dir)
sizes = trades[trades['symbol'] == 'KELP'].groupby('timestamp').size()
print(sizes)