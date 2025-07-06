from datamodel import OrderDepth, UserId, TradingState, Order
import string
import jsonpickle
from copy import deepcopy
import numpy as np
import pandas as pd

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

class Product:
    INK = 'SQUID_INK'


POSITION_LIMIT = {
    Product.INK: 50
}

DEFAULT_PARAMS = {

}

class Trader:
    def __init__(self, PARAMS=None):
        self.PARAMS = PARAMS if PARAMS is not None else DEFAULT_PARAMS

    def hysteresis_inflections(self, series, high_thresh=0.001, low_thresh=0.0005):
        dot_dot = series.dropna()
        idxs = dot_dot.index
        values = dot_dot.values

        inflections = []
        can_signal_pos = True
        can_signal_neg = True

        for i in range(len(values)):
            val = values[i]
            idx = idxs[i]

            if can_signal_pos and val > high_thresh:
                inflections.append(('positive', idx))
                can_signal_pos = False  # turn off until we fall below low_thresh
            elif val < low_thresh:
                can_signal_pos = True

            if can_signal_neg and val < -high_thresh:
                inflections.append(('negative', idx))
                can_signal_neg = False
            elif val > -low_thresh:
                can_signal_neg = True

        return inflections

    def run(self, state: TradingState) -> dict[str, list[Order]]:
        self.timestamp = state.timestamp
        trader_data = {}
        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)

        result = {}
        order_depth = state.order_depths[Product.INK]
        macaron_orders = []

        position = state.position.get(Product.INK, 0)
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        best_bid, best_bid_vol = sorted(buy_orders.items(), reverse=True)[0]
        best_ask, best_ask_vol = sorted(sell_orders.items(), reverse=False)[0]
        mid = (best_bid + best_ask) / 2

        # conversion observation
        observation = state.observations.conversionObservations.get(Product.INK)

        conversionBid = observation.bidPrice
        conversionAsk = observation.askPrice
        transportFees = observation.transportFees
        exportTariff = observation.exportTariff
        importTariff = observation.importTariff
        sugarPrice = observation.sugarPrice
        sunlightIndex = observation.sunlightIndex


        # fixed parameters
        threshold = 0.02
        required_sunlight_window_size = 25


        sunlight_history: list = trader_data.get('sunlight_history', [])
        sunlight_history.append(sunlightIndex)
        if len(sunlight_history) > required_sunlight_window_size:
            sunlight_history.pop(0)
        trader_data['sunlight_history'] = sunlight_history

        sunlight: pd.Series = pd.Series(sunlight_history)
        dot_sunlight = sunlight.diff().rolling(10).mean().rolling(10).mean()
        dot_dot_sunlight = dot_sunlight.diff(2)

        dot_inflections = self.hysteresis_inflections(dot_dot_sunlight)

        signal = None
        regime = None

        # the minus two indicates how long the signal for the regime change will run
        if dot_inflections and dot_inflections[-1][1] >= required_sunlight_window_size - 2:
            inflection_sign, index = dot_inflections[-1]
            last_dot_sunlight = dot_sunlight.iloc[-1]
            if last_dot_sunlight >= 0.02:
                regime = 'RISING'
            elif last_dot_sunlight <= -0.02:
                regime = 'FALLING'
            elif -0.005 <= last_dot_sunlight <= 0.005:
                regime = 'NEUTRAL'
            
        
        # regime to signal
        if regime == 'RISING':
            signal = 'SELL'
        elif regime == 'FALLING':
            signal = 'BUY'
        elif regime == 'NEUTRAL':
            # TODO implement normal market take and make here with clear
            signal = 'CLEAR'

        if regime:
            print(regime)
        
        # signal to execution
        if signal == 'BUY':
            to_buy = min(-best_ask_vol, POSITION_LIMIT[Product.INK] - position)
            if to_buy > 0:
                macaron_orders.append(Order(Product.INK, best_ask, to_buy))

        elif signal == 'SELL':
            to_sell = min(best_bid_vol, position - -POSITION_LIMIT[Product.INK])
            if to_sell > 0:
                macaron_orders.append(Order(Product.INK, best_bid, -to_sell))

        elif signal == 'CLEAR':
            if position > 0:
                macaron_orders.append(Order(Product.INK, best_bid, -position))
            elif position < 0:
                macaron_orders.append(Order(Product.INK, best_ask, -position))


        result[Product.INK] = macaron_orders

        traderData = jsonpickle.encode(trader_data)
        conversions = 0

        # logger.flush(
        #     state,
        #     result,
        #     conversions,
        #     traderData
        # )

        return result, conversions, traderData
