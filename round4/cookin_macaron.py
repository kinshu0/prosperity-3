from datamodel import OrderDepth, UserId, TradingState, Order
import string
import jsonpickle
from copy import deepcopy
import numpy as np
import pandas as pd

# from datamodel import OrderDepth, UserId, TradingState, Order, Trade, Symbol, listing, Observation, ProsperityEncoder
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

#     def compress_listings(self, listings: dict[Symbol, listing]) -> list[list[Any]]:
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
    MACARON = 'MAGNIFICENT_MACARONS'

POSITION_LIMIT = {
    Product.MACARON: 75
}

DEFAULT_PARAMS = {

}

class Trader:


    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: list[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: list[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: list[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> list[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (list[Order], int, int):
        orders: list[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (list[Order], int, int):
        orders: list[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: list[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume
    


    def __init__(self, PARAMS=None):
        self.PARAMS = PARAMS if PARAMS is not None else DEFAULT_PARAMS
        self.LIMIT = POSITION_LIMIT

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
        order_depth = state.order_depths[Product.MACARON]
        macaron_orders: list[Order] = []

        position = state.position.get(Product.MACARON, 0)
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        best_bid, best_bid_vol = sorted(buy_orders.items(), reverse=True)[0]
        best_ask, best_ask_vol = sorted(sell_orders.items(), reverse=False)[0]
        mid = (best_bid + best_ask) / 2

        # conversion observation
        observation = state.observations.conversionObservations.get(Product.MACARON)

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
            
        TAKE = True
        CLEAR = True
        MAKE = True
        
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
            to_buy = min(-best_ask_vol, POSITION_LIMIT[Product.MACARON] - position)
            if to_buy > 0:
                macaron_orders.append(Order(Product.MACARON, best_ask, to_buy))

        elif signal == 'SELL':
            to_sell = min(best_bid_vol, position - -POSITION_LIMIT[Product.MACARON])
            if to_sell > 0:
                macaron_orders.append(Order(Product.MACARON, best_bid, -to_sell))

        position += (sum(o.quantity for o in macaron_orders) if macaron_orders else 0)

        if TAKE:
            fair = sunlight.mean()
            take_orders, buy_vol, sell_vol = self.take_orders(Product.MACARON, order_depth, fair, 2, position)
            # print(take_orders)
            position += buy_vol - sell_vol
            macaron_orders.extend(take_orders)

        if CLEAR:
            if position > 0:
                macaron_orders.append(Order(Product.MACARON, best_bid, -position))
            elif position < 0:
                macaron_orders.append(Order(Product.MACARON, best_ask, -position))


        result[Product.MACARON] = macaron_orders

        traderData = jsonpickle.encode(trader_data)
        conversions = 0

        # logger.flush(
        #     state,
        #     result,
        #     conversions,
        #     traderData
        # )

        return result, conversions, traderData
