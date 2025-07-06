from datamodel import OrderDepth, UserId, TradingState, Order, Trade
import string
import jsonpickle
from copy import deepcopy
import numpy as np
import pandas as pd

class Product:
    RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PB1 = "PICNIC_BASKET1"
    PB2 = "PICNIC_BASKET2"
    MAC = "MAGNIFICENT_MACARONS"
    VOLC_ROCK = "VOLCANIC_ROCK"
    VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"


POSITION_LIMIT = {
    Product.RESIN: 50,
    Product.KELP: 50,
    Product.INK: 50,
    Product.CROISSANTS: 250,
    Product.JAMS: 350,
    Product.DJEMBES: 60,
    Product.PB1: 60,
    Product.PB2: 100,
}

DEFAULT_PARAMS = {}


class Trader:
    def __init__(self, PARAMS=None):
        self.PARAMS = PARAMS if PARAMS is not None else DEFAULT_PARAMS

    def limit_buy(
        self,
        product: str,
        order_depth: OrderDepth,
        limit_price: int | float,
        position: int,
        position_limit: int,
    ) -> tuple[list[Order], int, int]:
        to_buy = position_limit - position
        market_sell_orders = sorted(order_depth.sell_orders.items())

        own_orders = []
        buy_order_volume = 0

        max_buy_price = limit_price

        for price, volume in market_sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                own_orders.append(Order(product, price, quantity))
                to_buy -= quantity
                order_depth.sell_orders[price] += quantity
                if order_depth.sell_orders[price] == 0:
                    order_depth.sell_orders.pop(price)
                buy_order_volume += quantity

        return own_orders

    def limit_sell(
        self,
        product: str,
        order_depth: OrderDepth,
        limit_price: int | float,
        position: int,
        position_limit: int,
    ) -> tuple[list[Order], int, int]:
        to_sell = position - -position_limit

        market_buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        own_orders = []
        sell_order_volume = 0

        min_sell_price = limit_price

        for price, volume in market_buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                own_orders.append(Order(product, price, -quantity))
                to_sell -= quantity
                order_depth.buy_orders[price] -= quantity
                if order_depth.buy_orders[price] == 0:
                    order_depth.buy_orders.pop(price)
                sell_order_volume += quantity

        return own_orders
    

    def get_trader_var(self, var_name: str, trader_data: dict) -> float:
        trader_var = trader_data.get(var_name)
        return trader_var
    
    def set_trader_var(self, var_name: str, value, trader_data: dict):
        trader_data[var_name] = value
        return trader_data

    def update_timestamped_signals(self, signals, timestamp: int, trader_data: dict) -> list[tuple[str, str, float, int]]:
        timestamped_signals = trader_data.get('timestamped_signals', [])
        for signal, prod, price in signals:
            timestamped_signals.append((signal, prod, price, timestamp))
        trader_data['timestamped_signals'] = timestamped_signals
        return timestamped_signals

    def get_timestamped_signals(self, timestamp: int, tte: int, trader_data: dict) -> list[tuple[str, str, float, int]]:
        timestamped_signals = trader_data.get('timestamped_signals', [])
        timestamped_signals = [(signal, prod, price) for signal, prod, price, ts in timestamped_signals if timestamp - ts <= tte]
        return timestamped_signals

    def person_trades(self, state: TradingState, result: dict, trader_data: dict):

        signals = []

        OLIVIA = 'Olivia'

        combined_trades: dict[str, list[Trade]] = {}

        for prod, trades in state.market_trades.items():
            combined_trades[prod] = combined_trades.get(prod, []) + trades

        for prod, trades in state.own_trades.items():
            combined_trades[prod] = combined_trades.get(prod, []) + trades

        for prod, trades in combined_trades.items():
            olivia_trades = []
            for t in trades:
                if t.buyer == t.seller:
                    continue
                elif t.buyer == OLIVIA:
                    olivia_trades.append(t)
                elif t.seller == OLIVIA:
                    t.quantity *= -1
                    olivia_trades.append(t)
                    
            if len(olivia_trades) == 0:
                continue

            trade = sorted(olivia_trades, key=lambda t: abs(t.quantity))[0]
            quantity = trade.quantity
            price = trade.price
            if quantity > 0:
                signals.append(('BUY', prod, price * 1.2))
            elif quantity < 0:
                signals.append(('SELL', prod, price * 0.8))

        unexpired_signals = []
        unexpired_signals += self.get_timestamped_signals(state.timestamp, 500, trader_data)

        all_signals = signals + unexpired_signals

        # if unexpired_signals:
        #     print(f'Unexpired Signals: {unexpired_signals}')

        for signal, prod, price in all_signals:
            od = deepcopy(state.order_depths.get(prod, {}))
            pos = state.position.get(prod, 0)
            prod_orders = []
            if signal == 'BUY':
                prod_orders += self.limit_buy(prod, od, price, pos, POSITION_LIMIT[prod])
                # prod_orders += self.limit_buy(prod, od, 999999, pos, POSITION_LIMIT[prod])
            elif signal == 'SELL':
                prod_orders += self.limit_sell(prod, od, price, pos, POSITION_LIMIT[prod])
                # prod_orders += self.limit_sell(prod, od, 0, pos, POSITION_LIMIT[prod])
            
            result[prod] = result.get(prod, []) + prod_orders

        self.update_timestamped_signals(signals, state.timestamp, trader_data)

    def run(self, state: TradingState) -> dict[str, list[Order]]:
        self.timestamp = state.timestamp
        trader_data = {}
        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)

        result = {}

        self.person_trades(state, result, trader_data)

        # if result:
        #     print(f'{state.position=}')
        #     print(f'{result=}')

        

        traderData = jsonpickle.encode(trader_data)
        conversions = 0

        return result, conversions, traderData