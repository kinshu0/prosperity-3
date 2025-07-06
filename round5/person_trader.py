from datamodel import OrderDepth, UserId, TradingState, Order, Trade
import string
import jsonpickle
from copy import deepcopy
import numpy as np
import pandas as pd


# import json
# from typing import Any

# from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


# class Logger:
#     def __init__(self) -> None:
#         self.logs = ""
#         self.max_log_length = 3750

#     def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
#         self.logs += sep.join(map(str, objects)) + end

#     def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
#         base_length = len(
#             self.to_json(
#                 [
#                     self.compress_state(state, ""),
#                     self.compress_orders(orders),
#                     conversions,
#                     "",
#                     "",
#                 ]
#             )
#         )

#         # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
#         max_item_length = (self.max_log_length - base_length) // 3

#         print(
#             self.to_json(
#                 [
#                     self.compress_state(state, self.truncate(state.traderData, max_item_length)),
#                     self.compress_orders(orders),
#                     conversions,
#                     self.truncate(trader_data, max_item_length),
#                     self.truncate(self.logs, max_item_length),
#                 ]
#             )
#         )

#         self.logs = ""

#     def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
#         return [
#             state.timestamp,
#             trader_data,
#             self.compress_listings(state.listings),
#             self.compress_order_depths(state.order_depths),
#             self.compress_trades(state.own_trades),
#             self.compress_trades(state.market_trades),
#             state.position,
#             self.compress_observations(state.observations),
#         ]

#     def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
#         compressed = []
#         for listing in listings.values():
#             compressed.append([listing.symbol, listing.product, listing.denomination])

#         return compressed

#     def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
#         compressed = {}
#         for symbol, order_depth in order_depths.items():
#             compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

#         return compressed

#     def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
#         compressed = []
#         for arr in trades.values():
#             for trade in arr:
#                 compressed.append(
#                     [
#                         trade.symbol,
#                         trade.price,
#                         trade.quantity,
#                         trade.buyer,
#                         trade.seller,
#                         trade.timestamp,
#                     ]
#                 )

#         return compressed

#     def compress_observations(self, observations: Observation) -> list[Any]:
#         conversion_observations = {}
#         for product, observation in observations.conversionObservations.items():
#             conversion_observations[product] = [
#                 observation.bidPrice,
#                 observation.askPrice,
#                 observation.transportFees,
#                 observation.exportTariff,
#                 observation.importTariff,
#                 observation.sugarPrice,
#                 observation.sunlightIndex,
#             ]

#         return [observations.plainValueObservations, conversion_observations]

#     def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
#         compressed = []
#         for arr in orders.values():
#             for order in arr:
#                 compressed.append([order.symbol, order.price, order.quantity])

#         return compressed

#     def to_json(self, value: Any) -> str:
#         return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

#     def truncate(self, value: str, max_length: int) -> str:
#         lo, hi = 0, min(len(value), max_length)
#         out = ""

#         while lo <= hi:
#             mid = (lo + hi) // 2

#             candidate = value[:mid]
#             if len(candidate) < len(value):
#                 candidate += "..."

#             encoded_candidate = json.dumps(candidate)

#             if len(encoded_candidate) <= max_length:
#                 out = candidate
#                 lo = mid + 1
#             else:
#                 hi = mid - 1

#         return out


# logger = Logger()


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
    

    # def get_trader_var(self, var_name: str, trader_data: dict) -> float:
    #     trader_var = trader_data.get(var_name)
    #     return trader_var
    
    # def set_trader_var(self, var_name: str, value, trader_data: dict):
    #     trader_data[var_name] = value
    #     return trader_data

    # def update_timestamped_signals(self, signals, timestamp: int, trader_data: dict) -> list[tuple[str, str, float, int]]:
    #     timestamped_signals = trader_data.get('timestamped_signals', [])
    #     for signal, prod, price in signals:
    #         timestamped_signals.append((signal, prod, price, timestamp))
    #     trader_data['timestamped_signals'] = timestamped_signals
    #     return timestamped_signals

    # def get_timestamped_signals(self, timestamp: int, tte: int, trader_data: dict) -> list[tuple[str, str, float, int]]:
    #     timestamped_signals = trader_data.get('timestamped_signals', [])
    #     timestamped_signals = [(signal, prod, price) for signal, prod, price, ts in timestamped_signals if timestamp - ts <= tte]
    #     return timestamped_signals
    
    def update_past_trades(self, new_trades: list[Trade], timestamp: int, trader_data: dict):
        tte = 500
        past_trades = trader_data.get('past_trades', [])
        past_trades = [trade for trade in past_trades if timestamp - trade[5] <= tte]
        for trade in new_trades:
            trade_tuple = (trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp)
            if trade_tuple not in past_trades:
                past_trades.append(trade_tuple)
        trader_data['past_trades'] = past_trades

    def query_past_trades(self, person: str, timestamp: int, tte: int, trader_data: dict) -> list[Trade]:
        past_trades = trader_data.get('past_trades', [])
        past_trades = [Trade(symbol, price, quantity, buyer, seller, timestamp) for symbol, price, quantity, buyer, seller, timestamp in past_trades]
        past_trades = [trade for trade in past_trades if timestamp - trade.timestamp <= tte and (trade.buyer == person or trade.seller == person)]
        return past_trades

    def person_trades(self, state: TradingState, result: dict, trader_data: dict):

        signals = []

        OLIVIA = 'Olivia'

        olivia_past_trades = self.query_past_trades(OLIVIA, state.timestamp, 500, trader_data)

        if olivia_past_trades:
            print(olivia_past_trades)
            print(f'{state.position=}')

        for trade in olivia_past_trades:
            quantity = trade.quantity
            price = trade.price
            if trade.seller == OLIVIA:
                quantity *= -1
            if quantity > 0:
                signals.append(('BUY', trade.symbol, price * 1.2))
            elif quantity < 0:
                signals.append(('SELL', trade.symbol, price * 0.8))

        if len(signals) == 0:
            return

        # unexpired_signals = []
        # unexpired_signals += self.get_timestamped_signals(state.timestamp, 500, trader_data)

        # all_signals = signals + unexpired_signals

        for signal, prod, price in signals:
            if prod == Product.KELP:
                continue
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

        # self.update_timestamped_signals(signals, state.timestamp, trader_data)

    def run(self, state: TradingState) -> dict[str, list[Order]]:
        self.timestamp = state.timestamp
        trader_data = {}
        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)

        result = {}

        all_trades = []
        for values in state.market_trades.values():
            all_trades += values
        for values in state.own_trades.values():
            all_trades += values

        self.update_past_trades(all_trades, state.timestamp, trader_data)
        # self.update_past_trades(all_trades, trader_data)

        self.person_trades(state, result, trader_data)

        # if result:
        #     print(f'{state.position=}')
        #     print(f'{result=}')


        traderData = jsonpickle.encode(trader_data)
        conversions = 0

        # logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData