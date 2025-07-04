from datamodel import OrderDepth, UserId, TradingState, Order
import string
import jsonpickle
import math
from copy import deepcopy

class Product:
    KELP = 'KELP'
    RESIN = 'RAINFOREST_RESIN'
    INK = 'SQUID_INK'


# PARAMS = {'ink_change_threshold_pct': 0.015,
#  'ink_window_size': 25,
#  'ink_position_limit': 50,
#  'clear_price_thresh': 0.0
# }
PARAMS = {'ink_change_threshold_pct': 0.012,
 'ink_window_size': 20,
 'ink_position_limit': 50,
 'clear_price_thresh': 0.0
}


import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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


class Trader:
    def __init__(self, params: dict = None):
        self.params = params
        if params is None:
            self.params = PARAMS

    def market_take(self, product: str, order_depth: OrderDepth, fair: int | float, width: int | float, position: int, position_limit: int) -> tuple[list[Order], int, int]:
        to_buy = position_limit - position
        to_sell = position - -position_limit

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        own_orders = []
        buy_order_volume = 0
        sell_order_volume = 0

        max_buy_price = fair - width
        min_sell_price = fair + width


        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                own_orders.append(Order(product, price, quantity))
                to_buy -= quantity
                order_depth.sell_orders[price] += quantity
                if order_depth.sell_orders[price] == 0:
                    order_depth.sell_orders.pop(price)
                buy_order_volume += quantity
                
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                own_orders.append(Order(product, price, -quantity))
                to_sell -= quantity
                order_depth.buy_orders[price] -= quantity
                if order_depth.buy_orders[price] == 0:
                    order_depth.buy_orders.pop(price)
                sell_order_volume += quantity

        return own_orders, buy_order_volume, sell_order_volume

    def clear_position(self, product: str, order_depth: OrderDepth, fair: float | int, position: int, buy_order_volume: int, sell_order_volume: int, position_limit: int) -> list[Order]:
        pos_after_take = position + buy_order_volume - sell_order_volume
        own_orders = []

        clear_width = 0

        market_bids = list(order_depth.buy_orders.items())
        market_bids.sort(reverse=True)

        for bid_price, bid_volume in market_bids:
            if pos_after_take > 0 and bid_price >= fair + clear_width:
                sell_vol = min(bid_volume, pos_after_take, position - sell_order_volume - -position_limit)
                sell_order = Order(product, bid_price, -sell_vol)

                own_orders.append(sell_order)
                pos_after_take -= sell_vol
                sell_order_volume += sell_vol

                order_depth.buy_orders[bid_price] -= sell_vol
                if order_depth.buy_orders[bid_price] == 0:
                    order_depth.buy_orders.pop(bid_price)

        
        market_asks = list(order_depth.sell_orders.items())
        market_asks.sort()

        for ask_price, ask_volume in market_asks:
            if pos_after_take < 0 and ask_price <= fair - clear_width:
                buy_vol = min(-ask_volume, -pos_after_take, position_limit - (position + buy_order_volume))
                buy_order = Order(product, ask_price, buy_vol)

                own_orders.append(buy_order)
                pos_after_take += buy_vol
                buy_order_volume += buy_vol

                order_depth.sell_orders[ask_price] += buy_vol
                if order_depth.sell_orders[ask_price] == 0:
                    order_depth.sell_orders.pop(ask_price)

        return own_orders, buy_order_volume, sell_order_volume

    def market_make(self, product: str, order_depth: OrderDepth, fair: float | int, make_width: int, position: int, buy_order_volume: int, sell_order_volume: int, position_limit: int) -> list[Order]:
        orders = []

        market_bids = list(order_depth.buy_orders.items())
        market_bids.sort(reverse=True)

        best_bid, _ = max(market_bids)
        bid_vol = position_limit - (position + buy_order_volume)

        market_asks = list(order_depth.sell_orders.items())
        market_asks.sort()

        best_ask, _ = min(market_asks)
        ask_vol = position - sell_order_volume - -position_limit
        ask_vol = -ask_vol

        penny_bid = min(best_bid + 1, math.floor(fair - make_width))
        penny_ask = max(best_ask - 1, math.ceil(fair + make_width))

        orders.append(Order(product, penny_bid, bid_vol))
        orders.append(Order(product, penny_ask, ask_vol))     
        buy_order_volume += bid_vol
        sell_order_volume += ask_vol   
        
        return orders, buy_order_volume, sell_order_volume
    
    def mm_mid(self, product: str, order_depth: OrderDepth, trader_data: dict) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            adverse_volume = 15
            # reversion_beta = -0.18172393033850867
            reversion_beta = 0

            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= adverse_volume
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= adverse_volume
            ]

            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if trader_data.get(f"{product}_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = trader_data[f"{product}_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if trader_data.get(f"{product}_last_price", None) != None:
                last_price = trader_data[f"{product}_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * reversion_beta
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            trader_data[f"{product}_last_price"] = mmmid_price
            return fair
        return None

    def mid_price(self, order_depth: OrderDepth) -> float:
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        mid = (best_ask + best_bid) / 2
        return mid
    
    def rolling_mean(self, product: str, order_depth: OrderDepth, trader_data: dict) -> float:
        ink_history = trader_data.get(f'{product}_history', [])

        curr_mid = self.mid_price(order_depth)
            
        # todo: shouldn't be using currrent mid for comparison against rolling mean past
        ink_history.append(curr_mid)

        if len(ink_history) > 10:
            ink_history.pop(0)
        trader_data[f'{product}_history'] = ink_history

        return sum(ink_history) / len(ink_history)
    
    def limit_buy(self, product: str, order_depth: OrderDepth, limit_price: int | float, position: int, position_limit: int) -> tuple[list[Order], int, int]:
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
    
    def limit_sell(self, product: str, order_depth: OrderDepth, limit_price: int | float, position: int, position_limit: int) -> tuple[list[Order], int, int]:
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

    def ink(self, order_depth: OrderDepth, position: int, trader_data: dict) -> list[Order]:
        od = deepcopy(order_depth)
        best_market_ask = min(order_depth.sell_orders.keys())
        best_market_bid = max(order_depth.buy_orders.keys())

        ink_window_size = self.params['ink_window_size']
        change_threshold_pct = self.params['ink_change_threshold_pct']
        position_limit = self.params['ink_position_limit']
        clear_price_thresh = self.params['clear_price_thresh']

        curr_mid = self.mid_price(order_depth)
        
        ink_price_hist: list = trader_data.get('ink_price_hist', [])
        mean_price = sum(ink_price_hist) / len(ink_price_hist) if ink_price_hist else curr_mid
        std = (sum((x - mean_price) ** 2 for x in ink_price_hist) / len(ink_price_hist)) ** (1/2) if ink_price_hist else 1

        ink_trigger_hist: list = trader_data.get('ink_trigger_hist', [0] * ink_window_size)

        position_hist: list = trader_data.get('ink_position_hist', [])
        position_hist.append(position)
        if len(position_hist) > ink_window_size:
            position_hist.pop(0)

        orders = []

        delta_pct = curr_mid / mean_price - 1

        threshold_triggered = False

        # big change up
        if delta_pct >= change_threshold_pct:
            # sell
            orders = self.limit_sell(Product.INK, od, best_market_bid, position, position_limit)
            threshold_triggered = True

        # big change down
        elif delta_pct <= -change_threshold_pct:
            # buy
            orders = self.limit_buy(Product.INK, od, best_market_ask, position, position_limit)
            threshold_triggered = True

        
        # neutralize position on flats
        elif position > 0:
            # sell
            orders = self.limit_sell(Product.INK, od, mean_price - clear_price_thresh, position, 0)

        elif position < 0:
            # buy
            orders = self.limit_buy(Product.INK, od, mean_price + clear_price_thresh, position, 0)

        
        # if ink_trigger_hist[-1] == 1 and not threshold_triggered:
        #     # neutralize position
        #     if position > 0:
        #         orders = self.limit_sell(Product.INK, od, mean_price - clear_price_thresh, position, 0)
        #         # orders = self.limit_sell(Product.INK, od, best_market_bid, position, 0)
        #     elif position < 0:
        #         orders = self.limit_buy(Product.INK, od, mean_price + clear_price_thresh, position, 0)
        #         # orders = self.limit_buy(Product.INK, od, best_market_ask, position, 0)
        
        '''Update ink price history'''
        if not threshold_triggered:
            ink_price_hist.append(curr_mid)

        ink_trigger_hist.append(1 if threshold_triggered else 0)

        if len(ink_price_hist) > ink_window_size:
            ink_price_hist.pop(0)

        if len(ink_trigger_hist) > ink_window_size:
            ink_trigger_hist.pop(0)

        trader_data['ink_price_hist'] = ink_price_hist
        trader_data['ink_trigger_hist'] = ink_trigger_hist
        trader_data['ink_position_hist'] = position_hist

        return orders


    def run(self, state: TradingState) -> dict[str, list[Order]]:
        self.timestamp = state.timestamp
        order_depth = state.order_depths
        traderData = state.traderData

        result = {}

        if traderData is None or traderData == "":
            trader_data = {}
        else:
            trader_data = jsonpickle.decode(traderData)

        for product in [Product.INK,]:
            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]

            orders = []

            if order_depth is None:
                continue

            orders = self.ink(order_depth, position, trader_data)

            result[product] = orders

        trader_data = jsonpickle.encode(trader_data)
        
        conversions = 1

        logger.flush(state, result, conversions, traderData)

        return result, conversions, trader_data