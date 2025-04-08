from datamodel import OrderDepth, UserId, TradingState, Order
import string
import jsonpickle
from copy import deepcopy

class Product:
    KELP = 'KELP'
    RESIN = 'RAINFOREST_RESIN'

class Trader:

    def market_take(self, product: str, order_depth: OrderDepth, fair: int | float, width: int | float, position: int, position_limit: int) -> tuple[list[Order], int, int]:
        to_buy = position_limit - position
        to_sell = position + position_limit

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
                # todo: do we need this
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

    def clear_position(self, product: str, order_depth: OrderDepth, fair: float | int, position: int, buy_order_volume: int, sell_order_volume: int, position_limit: 20) -> list[Order]:
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

    def market_make(self, product: str, order_depth: OrderDepth, fair: float | int, position: int, buy_order_volume: int, sell_order_volume: int, position_limit: 20) -> list[Order]:
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

        penny_bid = min(best_bid + 1, 9999)
        penny_ask = max(best_ask - 1, 10001)

        orders.append(Order(product, penny_bid, bid_vol))
        orders.append(Order(product, penny_ask, ask_vol))     
        buy_order_volume += bid_vol
        sell_order_volume += ask_vol   
        
        return orders, buy_order_volume, sell_order_volume

    def calc_fair_kelp(self):
        pass

    def kelp(self, order_depth: OrderDepth, position: int) -> list[Order]:
        return []

    def resin(self, order_depth: OrderDepth, position: int) -> list[Order]:
        take_orders, buy_order_volume, sell_order_volume = self.market_take(Product.RESIN, order_depth, 10000, 1, position, 50)
        clear_orders, buy_order_volume, sell_order_volume = self.clear_position(Product.RESIN, order_depth, 10000, position, buy_order_volume, sell_order_volume, 50)
        make_orders, buy_order_volume, sell_order_volume = self.market_make(Product.RESIN, order_depth, 10000, position, buy_order_volume, sell_order_volume, 50)

        return take_orders + clear_orders + make_orders
        # return take_orders + make_orders

    def run(self, state: TradingState) -> dict[str, list[Order]]:
        self.timestamp = state.timestamp
        order_depth = state.order_depths
        traderData = state.traderData

        result = {}

        for product in order_depth:
            position = state.position.get(product, 0)
            order_depth = deepcopy(state.order_depths[product])

            if order_depth is None:
                continue

            if product == Product.RESIN:
                # print(f'Timestamp: {self.timestamp}')
                # print(f'Market bids: {order_depth.buy_orders}')
                # print(f'Market asks: {order_depth.sell_orders}')
                orders = self.resin(order_depth, position)
                # print(f'Resin orders: {orders}')

            elif product == Product.KELP:
                orders = self.kelp(order_depth, position)

            result[product] = orders
        
        conversions = 1
        return result, conversions, traderData