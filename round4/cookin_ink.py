from datamodel import OrderDepth, UserId, TradingState, Order
import string
import jsonpickle
from copy import deepcopy
import numpy as np


class Product:
    INK = 'SQUID_INK'

POSITION_LIMIT = {
    Product.INK: 50
}

# DEFAULT_PARAMS = {
#     'ink_window_size': 20,
#     'ink_momentum_lookback': 10,
#     'ink_momentum_threshold': 0.002,  # 0.2% momentum
# }

# from datamodel import OrderDepth, UserId, TradingState, Order, Trade, Symbol, Listing, Observation, ProsperityEncoder
# from typing import Any
# import json

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
#         if len(value) <= max_length:
#             return value

#         return value[: max_length - 3] + "..."


# logger = Logger()


# class Trader:
#     def __init__(self, PARAMS=None):
#         self.PARAMS = PARAMS
#         if self.PARAMS is None:
#             self.PARAMS = DEFAULT_PARAMS

#     def run(self, state: TradingState) -> dict[str, list[Order]]:
#         self.timestamp = state.timestamp
#         trader_data = {}
#         if state.traderData is not None and len(state.traderData) > 0:
#             trader_data = jsonpickle.decode(state.traderData)

#         result = {}

#         order_depth = state.order_depths[Product.INK]

#         ink_orders = []

#         position = state.position.get(Product.INK, 0)

#         market_buy_orders = order_depth.buy_orders
#         market_sell_orders = order_depth.sell_orders

#         best_bid, best_bid_vol = sorted(market_buy_orders.items(), reverse=True)[0]
#         best_ask, best_ask_vol = sorted(market_sell_orders.items(), reverse=False)[0]

#         mid = (best_bid + best_ask) / 2

#         window: list = trader_data.get('ink_window', [])

#         mean = np.mean(window) if window else mid
#         std = np.std(window) if window else 0

#         z = (mid - mean) / std if std > 0 else 0

#         take_threshold = self.PARAMS['ink_take_threshold']
#         clear_threshold = self.PARAMS['ink_clear_threshold']

#         # pct_change = mid / window[-1] - 1 if window else 0

#         if z >= take_threshold:
#             # short
#             # print('SELL')
#             to_sell = min(best_bid_vol, position - -POSITION_LIMIT[Product.INK])
#             if to_sell > 0:
#                 ink_orders.append(Order(Product.INK, best_bid, -to_sell))

#         elif z <= -take_threshold:
#             # long
#             # print('BUY')
#             to_buy = min(-best_ask_vol, POSITION_LIMIT[Product.INK] - position)
#             if to_buy > 0:
#                 ink_orders.append(Order(Product.INK, best_ask, to_buy))

#         # clear
#         elif z <= clear_threshold and z >= -clear_threshold and position > 0:
#             # clear
#             to_sell = position
#             if to_sell > 0:
#                 ink_orders.append(Order(Product.INK, best_bid, -to_sell))
#         elif z <= clear_threshold and z >= -clear_threshold and position < 0:
#             # clear
#             to_buy = -position
#             if to_buy > 0:
#                 ink_orders.append(Order(Product.INK, best_ask, to_buy))
        
#         # update window
#         window.append(mid)
#         if len(window) > self.PARAMS['ink_window_size']:
#             window.pop(0)

#         trader_data['ink_window'] = window
        
#         result[Product.INK] = ink_orders
        
#         traderData = jsonpickle.encode(trader_data)
#         conversions = 0


#         # logger.flush(
#         #     state,
#         #     result,
#         #     conversions,
#         #     traderData
#         # )

#         return result, conversions, traderData

DEFAULT_PARAMS = {
    'ink_window_size': 10,              # Size of the rolling window
    'ink_momentum_threshold': 0.0003,       # Change in price vs. mean that triggers a trade
    'ink_clear_threshold': 0.0002     # Range around mean where you flatten
}

class Trader:
    def __init__(self, PARAMS=None):
        self.PARAMS = PARAMS if PARAMS is not None else DEFAULT_PARAMS

    def run(self, state: TradingState) -> dict[str, list[Order]]:
        self.timestamp = state.timestamp
        trader_data = {}
        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)

        result = {}
        order_depth = state.order_depths[Product.INK]
        ink_orders = []

        position = state.position.get(Product.INK, 0)
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        best_bid, best_bid_vol = sorted(buy_orders.items(), reverse=True)[0]
        best_ask, best_ask_vol = sorted(sell_orders.items(), reverse=False)[0]
        mid = (best_bid + best_ask) / 2

        # Load historical mid prices
        price_history = trader_data.get('price_history', [])
        price_history.append(mid)
        if len(price_history) > 20:
            price_history.pop(0)

        signal = None
        if len(price_history) >= 11:
            # Calculate returns (10-period lagged momentum)
            return_now = (price_history[-1] / price_history[-11]) - 1
            return_past = (price_history[-2] / price_history[-12]) - 1 if len(price_history) >= 12 else 0

            # Detect momentum flip with magnitude threshold
            if return_past < 0 and return_now > 0 and (return_now - return_past) >= 0.008:
                signal = 'BUY'
            elif return_past > 0 and return_now < 0 and (return_past - return_now) >= 0.008:
                signal = 'SELL'
            elif abs(return_now) < 0.002:
                signal = 'CLEAR'

        # Act on signal
        if signal == 'BUY':
            to_buy = min(-best_ask_vol, POSITION_LIMIT[Product.INK] - position)
            if to_buy > 0:
                ink_orders.append(Order(Product.INK, best_ask, to_buy))

        elif signal == 'SELL':
            to_sell = min(best_bid_vol, position - -POSITION_LIMIT[Product.INK])
            if to_sell > 0:
                ink_orders.append(Order(Product.INK, best_bid, -to_sell))

        elif signal == 'CLEAR':
            if position > 0:
                ink_orders.append(Order(Product.INK, best_bid, -position))
            elif position < 0:
                ink_orders.append(Order(Product.INK, best_ask, -position))

        trader_data['price_history'] = price_history
        result[Product.INK] = ink_orders
        traderData = jsonpickle.encode(trader_data)
        conversions = 0

        return result, conversions, traderData
