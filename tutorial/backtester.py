import pandas as pd
from collections import defaultdict
from datamodel import Listing, Observation, Trade, TradingState, OrderDepth
import json

class Backtester:
    def __init__(self, trader, listings: dict[str, Listing], position_limit: dict[str, int], fair_marks, 
                 market_data: pd.DataFrame, trade_history: pd.DataFrame, file_name: str = None):
        self.trader = trader
        self.listings = listings
        self.market_data = market_data
        self.position_limit = position_limit
        self.fair_marks = fair_marks
        self.trade_history = trade_history.sort_values(by=['timestamp', 'symbol'])
        self.file_name = file_name

        self.observations = [Observation({}, {}) for _ in range(len(market_data))]

        self.current_position = {product: 0 for product in self.listings.keys()}
        self.pnl_history = []
        self.pnl = {product: 0 for product in self.listings.keys()}
        self.cash = {product: 0 for product in self.listings.keys()}
        self.trades = []
        self.sandbox_logs = []
        
    def run(self):
        traderData = ""
        
        timestamp_group_md = self.market_data.groupby('timestamp')
        timestamp_group_th = self.trade_history.groupby('timestamp')
        
        own_trades = defaultdict(list)
        market_trades = defaultdict(list)
        pnl_product = defaultdict(float)
        
        trade_history_dict = {}
        
        for timestamp, group in timestamp_group_th:
            trades = []
            for _, row in group.iterrows():
                symbol = row['symbol']
                price = row['price']
                quantity = row['quantity']
                buyer = row['buyer'] if pd.notnull(row['buyer']) else ""
                seller = row['seller'] if pd.notnull(row['seller']) else ""

                
                trade = Trade(symbol, int(price), int(quantity), buyer, seller, timestamp)
                
                trades.append(trade)
            trade_history_dict[timestamp] = trades
        
        for timestamp, group in timestamp_group_md:
            order_depths = self._construct_order_depths(group)
            order_depths_matching = self._construct_order_depths(group)
            order_depths_pnl = self._construct_order_depths(group)
            state = self._construct_trading_state(traderData, timestamp, self.listings, order_depths, 
                                 dict(own_trades), dict(market_trades), self.current_position, self.observations)
            orders, conversions, traderData = self.trader.run(state)
            products = group['product'].tolist()
            sandboxLog = ""
            trades_at_timestamp = trade_history_dict.get(timestamp, [])

            for product in products:
                new_trades = []
                for order in orders.get(product, []):
                    trades_done, sandboxLog = self._execute_order(timestamp, order, order_depths_matching, self.current_position, self.cash, trade_history_dict, sandboxLog)
                    new_trades.extend(trades_done)
                own_trades[product] = new_trades
            self.sandbox_logs.append({"sandboxLog": sandboxLog, "lambdaLog": "", "timestamp": timestamp})

            trades_at_timestamp = trade_history_dict.get(timestamp, [])
            if trades_at_timestamp:
                for trade in trades_at_timestamp:
                    product = trade.symbol
                    market_trades[product].append(trade)
            else: 
                for product in products:
                    market_trades[product] = []

            
            for product in products:
                self._mark_pnl(self.cash, self.current_position, order_depths_pnl, self.pnl, product)
                self.pnl_history.append(self.pnl[product])
            self._add_trades(own_trades, market_trades)
        return self._log_trades(self.file_name)
    
    
    def _log_trades(self, filename: str = None):
        if filename is None:
            return 

        self.market_data['profit_and_loss'] = self.pnl_history

        output = ""
        output += "Sandbox logs:\n"
        for i in self.sandbox_logs:
            output += json.dumps(i, indent=2) + "\n"

        output += "\n\n\n\nActivities log:\n"
        market_data_csv = self.market_data.to_csv(index=False, sep=";")
        market_data_csv = market_data_csv.replace("\r\n", "\n")
        output += market_data_csv

        output += "\n\n\n\nTrade History:\n"
        output += json.dumps(self.trades, indent=2)

        with open(filename, 'w') as file:
            file.write(output)

            
    def _add_trades(self, own_trades: dict[str, list[Trade]], market_trades: dict[str, list[Trade]]):
        products = set(own_trades.keys()) | set(market_trades.keys())
        for product in products:
            self.trades.extend([self._trade_to_dict(trade) for trade in own_trades.get(product, [])])
        for product in products:
            self.trades.extend([self._trade_to_dict(trade) for trade in market_trades.get(product, [])])

    def _trade_to_dict(self, trade: Trade) -> dict[str, ]:
        return {
            "timestamp": trade.timestamp,
            "buyer": trade.buyer,
            "seller": trade.seller,
            "symbol": trade.symbol,
            "currency": "SEASHELLS",
            "price": trade.price,
            "quantity": trade.quantity,
        }
        
    def _construct_trading_state(self, traderData, timestamp, listings, order_depths, 
                                 own_trades, market_trades, position, observations):
        state = TradingState(traderData, timestamp, listings, order_depths, 
                             own_trades, market_trades, position, observations)
        return state
    
        
    def _construct_order_depths(self, group):
        order_depths = {}
        for idx, row in group.iterrows():
            product = row['product']
            order_depth = OrderDepth()
            for i in range(1, 4):
                if f'bid_price_{i}' in row and f'bid_volume_{i}' in row:
                    bid_price = row[f'bid_price_{i}']
                    bid_volume = row[f'bid_volume_{i}']
                    if not pd.isna(bid_price) and not pd.isna(bid_volume):
                        order_depth.buy_orders[int(bid_price)] = int(bid_volume)
                if f'ask_price_{i}' in row and f'ask_volume_{i}' in row:
                    ask_price = row[f'ask_price_{i}']
                    ask_volume = row[f'ask_volume_{i}']
                    if not pd.isna(ask_price) and not pd.isna(ask_volume):
                        order_depth.sell_orders[int(ask_price)] = -int(ask_volume)
            order_depths[product] = order_depth
        return order_depths
    
        
        
    def _execute_buy_order(self, timestamp, order, order_depths, position, cash, trade_history_dict, sandboxLog):
        trades = []
        order_depth = order_depths[order.symbol]

        for price, volume in list(order_depth.sell_orders.items()):
            if price > order.price or order.quantity == 0:
                break

            trade_volume = min(abs(order.quantity), abs(volume))
            if abs(trade_volume + position[order.symbol]) <= int(self.position_limit[order.symbol]):
                trades.append(Trade(order.symbol, price, trade_volume, "SUBMISSION", "", timestamp))
                position[order.symbol] += trade_volume
                self.cash[order.symbol] -= price * trade_volume
                order_depth.sell_orders[price] += trade_volume
                order.quantity -= trade_volume
            else:
                sandboxLog += f"\nOrders for product {order.symbol} exceeded limit of {self.position_limit[order.symbol]} set"
            

            if order_depth.sell_orders[price] == 0:
                del order_depth.sell_orders[price]
        
        trades_at_timestamp = trade_history_dict.get(timestamp, [])
        new_trades_at_timestamp = []
        for trade in trades_at_timestamp:
            if trade.symbol == order.symbol:
                if trade.price < order.price:
                    trade_volume = min(abs(order.quantity), abs(trade.quantity))
                    trades.append(Trade(order.symbol, order.price, trade_volume, "SUBMISSION", "", timestamp))
                    order.quantity -= trade_volume
                    position[order.symbol] += trade_volume
                    self.cash[order.symbol] -= order.price * trade_volume
                    if trade_volume == abs(trade.quantity):
                        continue
                    else:
                        new_quantity = trade.quantity - trade_volume
                        new_trades_at_timestamp.append(Trade(order.symbol, order.price, new_quantity, "", "", timestamp))
                        continue
            new_trades_at_timestamp.append(trade)  

        if len(new_trades_at_timestamp) > 0:
            trade_history_dict[timestamp] = new_trades_at_timestamp

        return trades, sandboxLog
        
        
        
    def _execute_sell_order(self, timestamp, order, order_depths, position, cash, trade_history_dict, sandboxLog):
        trades = []
        order_depth = order_depths[order.symbol]
        
        for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if price < order.price or order.quantity == 0:
                break

            trade_volume = min(abs(order.quantity), abs(volume))
            if abs(position[order.symbol] - trade_volume) <= int(self.position_limit[order.symbol]):
                trades.append(Trade(order.symbol, price, trade_volume, "", "SUBMISSION", timestamp))
                position[order.symbol] -= trade_volume
                self.cash[order.symbol] += price * abs(trade_volume)
                order_depth.buy_orders[price] -= abs(trade_volume)
                order.quantity += trade_volume
            else:
                sandboxLog += f"\nOrders for product {order.symbol} exceeded limit of {self.position_limit[order.symbol]} set"

            if order_depth.buy_orders[price] == 0:
                del order_depth.buy_orders[price]

        trades_at_timestamp = trade_history_dict.get(timestamp, [])
        new_trades_at_timestamp = []
        for trade in trades_at_timestamp:
            if trade.symbol == order.symbol:
                if trade.price > order.price:
                    trade_volume = min(abs(order.quantity), abs(trade.quantity))
                    trades.append(Trade(order.symbol, order.price, trade_volume, "", "SUBMISSION", timestamp))
                    order.quantity += trade_volume
                    position[order.symbol] -= trade_volume
                    self.cash[order.symbol] += order.price * trade_volume
                    if trade_volume == abs(trade.quantity):
                        continue
                    else:
                        new_quantity = trade.quantity - trade_volume
                        new_trades_at_timestamp.append(Trade(order.symbol, order.price, new_quantity, "", "", timestamp))
                        continue
            new_trades_at_timestamp.append(trade)  

        if len(new_trades_at_timestamp) > 0:
            trade_history_dict[timestamp] = new_trades_at_timestamp
                
        return trades, sandboxLog
        
        
        
    def _execute_order(self, timestamp, order, order_depths, position, cash, trades_at_timestamp, sandboxLog):
        if order.quantity == 0:
            return []
        
        order_depth = order_depths[order.symbol]
        if order.quantity > 0:
            return self._execute_buy_order(timestamp, order, order_depths, position, cash, trades_at_timestamp, sandboxLog)
        else:
            return self._execute_sell_order(timestamp, order, order_depths, position, cash, trades_at_timestamp, sandboxLog)
    
    def _mark_pnl(self, cash, position, order_depths, pnl, product):
        order_depth = order_depths[product]
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid = (best_ask + best_bid)/2
        fair = mid
        if product in self.fair_marks:
            get_fair = self.fair_marks[product]
            fair = get_fair(order_depth)
        
        pnl[product] = cash[product] + fair * position[product]

import io


def read_trader_log(file):
    with open(file, 'r') as file:
        log_content = file.read()
    sections = log_content.split('Sandbox logs:')[1].split('Activities log:')
    sandbox_log =  sections[0].strip()
    activities_log = sections[1].split('Trade History:')[0]
    # sandbox_log_list = [json.loads(line) for line in sandbox_log.split('\n')]
    trade_history =  json.loads(sections[1].split('Trade History:')[1])
    # sandbox_log_df = pd.DataFrame(sandbox_log_list)
    market_data_df = pd.read_csv(io.StringIO(activities_log), sep=";", header=0)
    trade_history_df = pd.json_normalize(trade_history)
    return market_data_df, trade_history_df

from io import StringIO
import re
import orjson

def read_submission_log(filename: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(filename, 'r') as f:
        log_content = f.read()

    sandbox, market = log_content.split("Sandbox logs:")[1].split("Activities log:")
    market, trade_history = market.split("Trade History:")

    market = pd.read_csv(StringIO(market.strip()), sep=';')
    trade_history = orjson.loads(trade_history.strip())
    trade_history = pd.DataFrame(trade_history)
    sandbox = [orjson.loads(s) for s in re.findall(r'\{.*?\}', sandbox, re.DOTALL)]
    sandbox = pd.DataFrame(sandbox)
    return market, trade_history, sandbox
    
market_data, trade_history, sandbox = read_submission_log('logs/empty_submission.log')

import itertools

def generate_param_combinations(param_grid):
    param_names = param_grid.keys()
    param_values = param_grid.values()
    combinations = list(itertools.product(*param_values))
    return [dict(zip(param_names, combination)) for combination in combinations]


import os
from tqdm import tqdm

def run_backtests(trader, listings, position_limit, fair_calcs, market_data, trade_history, backtest_dir, param_grid, symbol):
    if not os.path.exists(backtest_dir):
        os.makedirs(backtest_dir)

    param_combinations = generate_param_combinations(param_grid[symbol])

    results = []
    for params in tqdm(param_combinations, desc=f"Running backtests for {symbol}", unit="backtest"):
        trader.params = {symbol: params}
        backtester = Backtester(trader, listings, position_limit, fair_calcs, market_data, trade_history)
        backtester.run()

        param_str = ";".join([f"{key}={value}" for key, value in params.items()])
        log_filename = f"{backtest_dir}/{symbol}_{param_str}.log"
        backtester._log_trades(log_filename)

        results.append((params, backtester.pnl[symbol]))

    return results


def analyze_log_files(backtest_dir):
    log_files = [f for f in os.listdir(backtest_dir) if f.endswith('.log')]
    
    results = []
    for log_file in log_files:
        file_path = os.path.join(backtest_dir, log_file)
        
        # Extract symbol and parameters from the file name
        file_name = os.path.splitext(log_file)[0]
        print(file_name)
        symbol, params_str = file_name.split(';', 1)
        params = dict(param.split('=') for param in params_str.split(';'))
        
        # Read the contents of the log file
        with open(file_path, 'r') as file:
            log_content = file.read()
        
        # Store the symbol, parameters, and log content in the results
        results.append({
            'symbol': symbol,
            'params': params,
            'log_content': log_content
        })
    
    return results
