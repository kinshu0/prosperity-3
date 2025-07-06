'''
option volatility mean reversion

implied vol is mean reverting, and mean implied vol should be same across all days

when implied vol exceeds mean vol by some threshold (n standard deviations), short option and delta hedge
when implied vol is below mean vol by some threshold, long option and delta hedge


assumption here is implied vol is constant across days

maybe assumption is incorrect since theta decay not taken into account?

we once again want to do black scholes over a windowo o ftime

do this with all strikes

is delta about the same over time?
what is the multiplier of vouchers?

there is probably way to trade arb between vouchers of different strikes as well, not gonna focus on that now


what about gamma scalping?


same as just trading around expected vs actual bs call where for each trade we hedge with delta


'''
from datamodel import OrderDepth, UserId, TradingState, Order
import string
import jsonpickle
import math
from copy import deepcopy
from collections import deque


class Product:
    ROCK = 'VOLCANIC_ROCK'
    VOUCHER_9500 = 'VOLCANIC_ROCK_VOUCHER_9500'
    VOUCHER_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
    VOUCHER_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
    VOUCHER_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
    VOUCHER_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'

position_limits = {
    Product.ROCK: 400,
    Product.VOUCHER_9500: 200,
    Product.VOUCHER_9750: 200,
    Product.VOUCHER_10000: 200,
    Product.VOUCHER_10250: 200,
    Product.VOUCHER_10500: 200
}


class OrderType:
    BUY = 'BUY'
    SELL = 'SELL'

PARAMS = {
    "ink_change_threshold_pct": 0.012,
    "ink_window_size": 20,
    "ink_position_limit": 50,
    "clear_price_thresh": 0.0,

    "sma_window_size": 125,
    "std_window_size": 25,
    "thresh_z": 8
    # "sma_window_size": 300,
    # "std_window_size": 80,
    # "thresh_z": 8
}


class


class Trader:
    def __init__(self, params: dict = None):
        self.params = params
        if params is None:
            self.params = PARAMS

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


    def basket(
        self, order_depths: dict[str, OrderDepth], positions: dict[str, int], trader_data: dict
    ) -> list[Order]:

        synthetics = [
            {
                Product.BASKET1: 1,
                Product.CROISSANTS: -6,
                Product.JAMS: -3,
                Product.DJEMBES: -1
            },
            {
                Product.BASKET2: 1,
                Product.CROISSANTS: -4,
                Product.JAMS: -2
            },
            {
                Product.BASKET1: 1,
                Product.BASKET2: -1,
                Product.CROISSANTS: -2,
                Product.JAMS: -1,
                Product.DJEMBES: -1
            },
            {
                Product.BASKET1: 2,
                Product.BASKET2: -3,
                Product.DJEMBES: -2
            },
        ]

        synth_weights = synthetics[2]

        for prod in synth_weights:
            positions[prod] = positions.get(prod, 0)

        synth_od = self.construct_synth_od(order_depths, synth_weights)
        
        best_ask = min(synth_od.sell_orders.keys())
        best_bid = max(synth_od.buy_orders.keys())
        
        # thresh: 8, std_window: 25, sma_window: 125

        # this works TODO:

        sma_window_size = self.params['sma_window_size']
        std_window_size = self.params['std_window_size']
        thresh_z = self.params['thresh_z']
        
        sma_window_size = 125
        std_window_size = 25
        thresh_z = 8

        swmid = self.swmid(synth_od)

        spread_price_hist = trader_data.get("spread_price_hist", [])
        spread_price_hist = deque(spread_price_hist)
        swmid_mean = (
            sum(spread_price_hist) / len(spread_price_hist) if spread_price_hist else swmid
        )
        
        std = 0
        std_window_size = min(std_window_size, len(spread_price_hist))
        for i in reversed(range(std_window_size)):
            std += (spread_price_hist[i] - swmid_mean) ** 2
        std = (std / std_window_size) ** (1 / 2) if spread_price_hist else 1

        std = max(std, 1)

        orders = []
        z = (swmid - swmid_mean) / std

        threshold_triggered = False

        position_limits = {
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.BASKET1: 60,
            Product.BASKET2: 100
        }

        orders = []

        # # big change up
        # if z >= thresh_z:
        #     # print('SELL THRESHOLD TRIGGERED')
        #     sell_synth_orders = self.sell_synth(best_bid - aggressiveness, position_limits, positions, synth_od, order_depths, synth_weights)
        #     orders.extend(sell_synth_orders)
        #     threshold_triggered = True

        # # big change down
        # elif z <= -thresh_z:
        #     # print('BUY THRESHOLD TRIGGERED')
        #     buy_synth_orders = self.buy_synth(best_ask + aggressiveness, position_limits, positions, synth_od, order_depths, synth_weights)
        #     orders.extend(buy_synth_orders)
        #     threshold_triggered = True

        synth_ods = [self.construct_synth_od(order_depths, synth) for synth in synthetics]

        synth_swmids = [self.swmid(synth_od) for synth_od in synth_ods]

        # swmid with max abolute value and its index
        max_smid_index = 0
        for i, smid in enumerate(synth_swmids):
            if abs(smid) > abs(synth_swmids[max_smid_index]):
                max_smid_index = i
        
        synth_weights = synthetics[max_smid_index]

        best_ask = min(synth_ods[max_smid_index].sell_orders.keys())
        best_bid = max(synth_ods[max_smid_index].buy_orders.keys())

        swmid = synth_swmids[max_smid_index]

        abs_threshold = 50

        aggresiveness = 5

        if swmid <= -abs_threshold:
            # print('BUY THRESHOLD TRIGGERED')
            buy_synth_orders = self.buy_synth(best_ask + aggresiveness, position_limits, positions, synth_od, order_depths, synth_weights)
            orders.extend(buy_synth_orders)
            threshold_triggered = True
        elif swmid >= abs_threshold:
            # print('SELL THRESHOLD TRIGGERED')
            sell_synth_orders = self.sell_synth(best_bid - aggresiveness, position_limits, positions, synth_od, order_depths, synth_weights)
            orders.extend(sell_synth_orders)
            threshold_triggered = True

        # if swmid > swmid
        
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

        rock_and_call_orders

        basket_orders = self.basket(order_depths, positions, trader_data)

        for order in basket_orders:
            result[order.symbol] = result.get(order.symbol, []) + [order]

        trader_data = jsonpickle.encode(trader_data)

        conversions = 1

        # logger.flush(state, result, conversions, traderData)

        return result, conversions, trader_data
