from datamodel import OrderDepth, UserId, TradingState, Order
import string
import jsonpickle
import math
from copy import deepcopy

class Product:
    KELP = 'KELP'
    RESIN = 'RAINFOREST_RESIN'
    INK = 'SQUID_INK'

class Trader:

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
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            mid = (best_ask + best_bid) / 2
            return mid
        return None
    
    def rolling_mean(self, product: str, order_depth: OrderDepth, trader_data: dict) -> float:
        ink_history = trader_data.get(f'{product}_history', [])

        curr_mid = self.mid_price(order_depth)
            
        # todo: shouldn't be using currrent mid for comparison against rolling mean past
        ink_history.append(curr_mid)

        if len(ink_history) > 10:
            ink_history.pop(0)
        trader_data[f'{product}_history'] = ink_history

        return sum(ink_history) / len(ink_history)
    
    def take_buy(self, product: str, order_depth: OrderDepth, fair: int | float, width: int | float, position: int, position_limit: int) -> tuple[list[Order], int, int]:
        to_buy = position_limit - position
        market_sell_orders = sorted(order_depth.sell_orders.items())
        fair, _ = market_sell_orders[0]

        own_orders = []
        buy_order_volume = 0

        max_buy_price = fair - width

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
    
    def take_sell(self, product: str, order_depth: OrderDepth, fair: int | float, width: int | float, position: int, position_limit: int) -> tuple[list[Order], int, int]:
        to_sell = position - -position_limit

        market_buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        fair, _ = market_buy_orders[0]

        own_orders = []
        sell_order_volume = 0

        min_sell_price = fair + width

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
        
        # todo: find the best parameters here
        # todo: instead of using fixed momentum_thresh, use standard deviation of ink_price_hist x constank factor as momentum_thresh

        momentum_thresh = 5
        ink_price_hist_window = 10

        curr_mid = self.mid_price(order_depth)
        
        ink_price_hist: list = trader_data.get('ink_price_hist', [])

        last_mean = sum(ink_price_hist) / len(ink_price_hist) if ink_price_hist else curr_mid

        orders = []

        # breaking ub
        if curr_mid >= last_mean + momentum_thresh:
            # buy
            orders = self.take_buy(Product.INK, order_depth, 999999, 0, position, 50)

        # breaking lb
        if curr_mid <= last_mean + momentum_thresh:
            # sell
            orders = self.take_sell(Product.INK, order_depth, 999999, 0, position, 50)
        
        # within bounds and we don't expect to break ub or lb
        # todo: just market make / take on both directions
        
        # update ink_price_hist
        ink_price_hist.append(curr_mid)

        if len(ink_price_hist) > ink_price_hist_window:
            ink_price_hist.pop(0)

        trader_data['ink_price_hist'] = ink_price_hist

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
        return result, conversions, trader_data