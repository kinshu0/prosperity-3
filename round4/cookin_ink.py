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

from datamodel import OrderDepth, UserId, TradingState, Order, Trade, Symbol, Listing, Observation, ProsperityEncoder
from typing import Any
import json

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


'''
	ink_lookback_window	ink_momentum_threshold	ink_past_lag	ink_clear_threshold	profit_day_1	tot_profit
182	20	0.007	3	0.0001	4826.0	4826.0
29	5	0.012	5	0.0001	4357.0	4357.0
269	30	0.010	5	0.0001	4064.0	4064.0
34	5	0.015	5	0.0001	3798.0	3798.0
32	5	0.015	3	0.0001	3676.0	3676.0
'''



# DEFAULT_PARAMS = {
#     'ink_lookback_window': 10,            # Number of periods for momentum lookback
#     'ink_momentum_threshold': 0.008,      # Change in return required to trigger a trade
#     'ink_past_lag': 1,                     # Number of periods to look back for past return
#     'ink_clear_threshold': 0.0001,         # Threshold below which we clear the position,
# }

'''
TODO
Implement clearing strategy, has to be different from normal mean reversion clearing
We have to hold for a certain period? Or wait for smaller local inflection perhaps?
Ideas
- some fixed / dynamic hold period based on certain parameters
- some threshold for uptick / downtick to trigger clearing based on positve / negative position

'''


DEFAULT_PARAMS = {
    'ink_lookback_window': 20,            # Number of periods for momentum lookback
    'ink_momentum_threshold': 0.005,      # Change in return required to trigger a trade
    'ink_past_lag': 1,                     # Number of periods to look back for past return
    'ink_clear_threshold': 0.0001,         # Threshold below which we clear the position,
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

        signal = None
        lookback = self.PARAMS['ink_lookback_window']
        threshold = self.PARAMS['ink_momentum_threshold']
        past_lag = self.PARAMS['ink_past_lag']
        required_history = lookback + past_lag + 1

        price_history = trader_data.get('price_history', [])
        price_history.append(mid)
        if len(price_history) > required_history:
            price_history.pop(0)

        if len(price_history) >= required_history:
            return_now = (price_history[-1] / price_history[-1 - lookback]) - 1
            return_past = (price_history[-1-past_lag] / price_history[-1-past_lag - lookback]) - 1

            # Detect momentum flip with magnitude threshold
            if return_past < 0 and return_now > 0 and (return_now - return_past) >= threshold:
                signal = 'BUY'
            elif return_past > 0 and return_now < 0 and (return_past - return_now) >= threshold:
                signal = 'SELL'
            # elif abs(return_now) < self.PARAMS['ink_clear_threshold']:
            #     signal = 'CLEAR'

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



        logger.flush(
            state,
            result,
            conversions,
            traderData
        )

        return result, conversions, traderData
