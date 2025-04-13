from datamodel import OrderDepth, UserId, TradingState, Order
import string
import jsonpickle
import math
from copy import deepcopy
from collections import deque


class Product:
    BASKET1 = "PICNIC_BASKET1"
    JAMS = "JAMS"
    CROISSANTS = "CROISSANTS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"

BASKET1_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}


# PARAMS = {'ink_change_threshold_pct': 0.015,
#  'ink_window_size': 25,
#  'ink_position_limit': 50,
#  'clear_price_thresh': 0.0
# }


class OrderType:
    BUY = 'BUY'
    SELL = 'SELL'

PARAMS = {
    "ink_change_threshold_pct": 0.012,
    "ink_window_size": 20,
    "ink_position_limit": 50,
    "clear_price_thresh": 0.0,
}


import json
from typing import Any

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
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
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
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

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
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

# logger = Logger()

class Trader:
    def __init__(self, params: dict = None):
        self.params = params
        if params is None:
            self.params = PARAMS

    def market_take(
        self,
        product: str,
        order_depth: OrderDepth,
        fair: int | float,
        width: int | float,
        position: int,
        position_limit: int,
    ) -> tuple[list[Order], int, int]:
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

    def clear_position(
        self,
        product: str,
        order_depth: OrderDepth,
        fair: float | int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        position_limit: int,
    ) -> list[Order]:
        pos_after_take = position + buy_order_volume - sell_order_volume
        own_orders = []

        clear_width = 0

        market_bids = list(order_depth.buy_orders.items())
        market_bids.sort(reverse=True)

        for bid_price, bid_volume in market_bids:
            if pos_after_take > 0 and bid_price >= fair + clear_width:
                sell_vol = min(
                    bid_volume,
                    pos_after_take,
                    position - sell_order_volume - -position_limit,
                )
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
                buy_vol = min(
                    -ask_volume,
                    -pos_after_take,
                    position_limit - (position + buy_order_volume),
                )
                buy_order = Order(product, ask_price, buy_vol)

                own_orders.append(buy_order)
                pos_after_take += buy_vol
                buy_order_volume += buy_vol

                order_depth.sell_orders[ask_price] += buy_vol
                if order_depth.sell_orders[ask_price] == 0:
                    order_depth.sell_orders.pop(ask_price)

        return own_orders, buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        order_depth: OrderDepth,
        fair: float | int,
        make_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        position_limit: int,
    ) -> list[Order]:
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
                if abs(order_depth.sell_orders[price]) >= adverse_volume
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= adverse_volume
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
                pred_returns = last_returns * reversion_beta
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

    def swmid(self, order_depth: OrderDepth) -> float:
        best_ask, ask_vol = min(order_depth.sell_orders.items())
        best_bid, bid_vol = max(order_depth.buy_orders.items())

        return (best_ask * bid_vol + best_bid * -ask_vol) / (bid_vol + -ask_vol)

    def rolling_mean(
        self, product: str, order_depth: OrderDepth, trader_data: dict
    ) -> float:
        ink_history = trader_data.get(f"{product}_history", [])

        curr_mid = self.mid_price(order_depth)

        # todo: shouldn't be using currrent mid for comparison against rolling mean past
        ink_history.append(curr_mid)

        if len(ink_history) > 10:
            ink_history.pop(0)
        trader_data[f"{product}_history"] = ink_history

        return sum(ink_history) / len(ink_history)

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

    def chunk_orders(self, orders: list[tuple], chunk_size: int, order_type=OrderType.BUY) -> dict[int, int]:
        if order_type == OrderType.BUY:
            orders = sorted(orders, reverse=True)
        else:
            orders = sorted([(p, -q) for p, q in orders])

        chunk_depth = {}
        i = 0
        while i < len(orders):
            price, quantity = orders[i]
            num_chunks = quantity // chunk_size
            leftover_pieces = quantity % chunk_size
            chunk_price = price * chunk_size
            if num_chunks > 0:
                chunk_depth[chunk_price] = chunk_depth.get(chunk_price, 0) + num_chunks
            if leftover_pieces > 0:
                hybrid_price = leftover_pieces * price
                i += 1
                while i < len(orders):
                    next_price, next_quantity = orders[i]
                    to_fill = chunk_size - leftover_pieces
                    filled = min(next_quantity, to_fill)
                    leftover_pieces += filled
                    hybrid_price += filled * next_price
                    next_quantity -= filled
                    orders[i] = next_price, next_quantity
                    if leftover_pieces == chunk_size:
                        break
                    i += 1
                if leftover_pieces == chunk_size:
                    chunk_depth[hybrid_price] = chunk_depth.get(hybrid_price, 0) + 1
            else:
                i += 1
        
        if order_type == OrderType.SELL:
            for price in chunk_depth:
                q = chunk_depth[price]
                chunk_depth[price] = -q

        return chunk_depth

    def construct_synth_buy_orders(self, order_depths: dict[str, OrderDepth], weights: dict[str, int]) -> dict[int, int]:

        synth_buy_orders = {}
        
        prod_chunked_buy = {}

        for prod, weight in weights.items():
            if prod not in order_depths:
                return {}

            # if weight is positve, market buy orders
            if weight > 0:
                buy_orders = order_depths[prod].buy_orders.items()
                chunked_buys = self.chunk_orders(buy_orders, weight)
                chunked_buys = sorted(chunked_buys.items(), reverse=True)
            # if weight is negative, market sell orders
            elif weight < 0:
                buy_orders = order_depths[prod].sell_orders.items()
                chunked_buys = self.chunk_orders(buy_orders, abs(weight), order_type=OrderType.SELL)
                chunked_buys = sorted(chunked_buys.items())

            if len(buy_orders) == 0:
                return {}

            prod_chunked_buy[prod] = chunked_buys

        exhausted = False

        while not exhausted:
            quantity = 999999999
            for prod, chunked_buys in prod_chunked_buy.items():
                best_price, vol = chunked_buys[0]
                quantity = min(quantity, abs(vol))

            synth_bid = 0

            for prod, chunked_buys in prod_chunked_buy.items():
                best_price, vol = chunked_buys[0]
                # dealing with negative weighted products
                if vol < 0:
                    synth_bid -= best_price
                    newvol = vol + quantity
                else:
                    synth_bid += best_price
                    newvol = vol - quantity
                chunked_buys[0] = best_price, newvol
                if newvol == 0:
                    chunked_buys.pop(0)
                if len(chunked_buys) == 0:
                    exhausted = True
                prod_chunked_buy[prod] = chunked_buys
                
            synth_buy_orders[synth_bid] = quantity

        return synth_buy_orders

    def construct_synth_od(self, order_depths: dict[str, OrderDepth], weights: dict[str, int]) -> OrderDepth:
            
        order_depth = OrderDepth()
        order_depth.buy_orders = self.construct_synth_buy_orders(order_depths, weights)

        neg_weights = {prod: -weights[prod] for prod in weights}
        sell_orders = self.construct_synth_buy_orders(order_depths, neg_weights)
        sell_orders_pos = {}
        for price, q in sell_orders.items():
            sell_orders_pos[-price] = -q

        order_depth.sell_orders = sell_orders_pos

        return order_depth

    def buy_synth(self, limit_price: int, position_limits: dict[str, int], positions: dict[str, int], synth_od: OrderDepth, order_depths: dict[str, OrderDepth], weights: dict[str, int]):
        
        can_buy = 999999

        for prod, weight in weights.items():
            # if weight is greater than 0 we want to look at market sells to match
            # don't want to exceed upper position limit
            if weight > 0:
                chunks = position_limits[prod] - positions[prod]
            # if weight is less than 0 we want to look at market buys and not exceed lower position limit
            else:
                chunks = positions[prod] - -position_limits[prod]
            can_buy = min(chunks // abs(weight), can_buy)
        

        # get synth volume to buy
        # we match with synth market sells to buy synth
        # only part different between buy and sell?
        synth_market_sells = synth_od.sell_orders
        acc_synth_market_sells = [(p, q) for p, q in synth_market_sells.items() if p <= limit_price]

        acc_market_sell_vol = sum(x[1] for x in acc_synth_market_sells) if acc_synth_market_sells else 0

        to_buy = min(-acc_market_sell_vol, can_buy)

        if to_buy == 0:
            return []
        
        orders = []

        for prod, weight in weights.items():
            # here q would automatically be negative if it's negative weight
            q = weight * to_buy
            match_price = 0 if weight < 0 else 9999999
            orders.append(Order(prod, match_price, q))

        return orders

    def sell_synth(self, limit_price: int, position_limits: dict[str, int], positions: dict[str, int], synth_od: OrderDepth, order_depths: dict[str, OrderDepth], weights: dict[str, int]):

        weights = {prod: -weights[prod] for prod in weights}
        
        can_sell = 999999

        for prod, weight in weights.items():
            if weight > 0:
                chunks = position_limits[prod] - positions[prod]
            else:
                chunks = positions[prod] - -position_limits[prod]
            can_sell = min(chunks // abs(weight), can_sell)

        synth_market_buys = synth_od.buy_orders
        acc_synth_market_buys = [(p, q) for p, q in synth_market_buys.items() if p >= limit_price]

        acc_market_buy_vol = sum(x[1] for x in acc_synth_market_buys) if acc_synth_market_buys else 0

        to_buy = min(acc_market_buy_vol, can_sell)

        if to_buy == 0:
            return []
        
        orders = []

        for prod, weight in weights.items():
            q = weight * to_buy
            match_price = 0 if weight < 0 else 9999999
            orders.append(Order(prod, match_price, q))

        return orders

    '''
    spread = basket1-synth spread
    long basket1, short synth
    long spread we'd want basket1+ and synth-

    spread being positive = basket1 overvalued synthetic undervalued

    spread z-score being above 5 = basket1 overvalued by much more than it should be and synth being much more undervalued
    we expect this to revert to current window mean
    so short spread by selling basket 1 and buying synth1

    should we use best bid/ask or swmid?
    case for using best bid/ask:
    - to long spread we'd match with market ask for basket1 and market bid for synth
    - use best bid/ask values of this to calculate price of spread
    

    window of mean prices for spread
    same for standard deviation
    calculate z-score of current swmid for spread
    if z-score > threshold: short spread
    if z-score < -threshold: long spread

    value synth buy buying individual components

    we want to either buy or sell synth / 


    dealing with not enough volume to buy/sell synth?

    modify order depth to add synthetic order volume and later parse into components in post-processing


    how do we buy synthetic and display synthetic volumes, we obv never buy components individually

    creating order depth for synthetic

    if there are 20 strawberries, 5 jams, 2 djembes -> for 6, 3, 1 it translates to volume of 1 synthetic at whatever price is being displayed

    for 2 synthetics, we'd pick off volume from other price levels which prices our components differently

    [(254, 21), (247, 8), (231, 12)]
    [(254, 91), (247, 8), (231, 12)]
    [(254, 11), (247, 8), (231, 12)]

    parameters for z-score trading spreads
    thresh: 8, std_window: 25, sma_window: 125, pnl: 12486.761331593298
    thresh: 10, std_window: 25, sma_window: 150, pnl: 11953.98168192296
    thresh: 5, std_window: 20, sma_window: 35, pnl: 10530.0

    '''

    def basket(
        self, order_depths: dict[str, OrderDepth], positions: dict[str, int], trader_data: dict
    ) -> list[Order]:
        
        synth_weights = {
            Product.BASKET1: 1,
            Product.CROISSANTS: -6,
            Product.JAMS: -3,
            Product.DJEMBES: -1
        }

        for prod in synth_weights:
            positions[prod] = positions.get(prod, 0)

        synth_od = self.construct_synth_od(order_depths, synth_weights)
        
        best_ask = min(synth_od.sell_orders.keys())
        best_bid = max(synth_od.buy_orders.keys())
        
        sma_window_size = 35
        std_window_size = 20
        thresh_z = 5

        swmid = self.swmid(synth_od)

        spread_price_hist = trader_data.get("spread_price_hist", [])
        spread_price_hist = deque(spread_price_hist)
        swmid_mean = (
            sum(spread_price_hist) / len(spread_price_hist) if spread_price_hist else swmid
        )
        
        std = 0
        for i in reversed(range(len(spread_price_hist))):
            std += (spread_price_hist[i] - swmid_mean) ** 2
        std = (std / len(spread_price_hist)) ** (1 / 2) if spread_price_hist else 1

        std = max(std, 1)

        orders = []
        z = (swmid - swmid_mean) / std

        threshold_triggered = False

        position_limits = {
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.BASKET1: 60,
        }

        orders = []

        print(z)

        # big change up
        if z >= thresh_z:
            # sell
            # TODO
            print('SELL THRESHOLD TRIGGERED')
            sell_synth_orders = self.sell_synth(best_ask, position_limits, positions, synth_od, order_depths, synth_weights)
            orders.extend(sell_synth_orders)
            threshold_triggered = True

        # big change down
        elif z <= -thresh_z:
            # buy
            # TODO
            print('BUY THRESHOLD TRIGGERED')
            buy_synth_orders = self.buy_synth(best_ask, position_limits, positions, synth_od, order_depths, synth_weights)
            orders.extend(buy_synth_orders)
            threshold_triggered = True
        
        """Update price history"""
        spread_price_hist.append(swmid)

        if len(spread_price_hist) > sma_window_size:
            spread_price_hist.popleft()

        trader_data["spread_price_hist"] = list(spread_price_hist)

        return orders

    def run(self, state: TradingState) -> dict[str, list[Order]]:
        self.timestamp = state.timestamp
        order_depths = deepcopy(state.order_depths)
        positions = deepcopy(state.position)
        traderData = state.traderData

        result = {}

        if traderData is None or traderData == "":
            trader_data = {}
        else:
            trader_data = jsonpickle.decode(traderData)


        basket_orders = self.basket(order_depths, positions, trader_data)

        for product in basket_orders:
            result[product] = result.get(product, []) + basket_orders[product]

        trader_data = jsonpickle.encode(trader_data)

        conversions = 1

        # logger.flush(state, result, conversions, traderData)

        return result, conversions, trader_data
