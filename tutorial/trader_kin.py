from datamodel import OrderDepth, UserId, TradingState, Order
import string
import jsonpickle


class Product:
    KELP = 'KELP'
    RESIN = 'RAINFOREST_RESIN'

class Trader:

    def market_take(self, product: str, order_depth: OrderDepth, fair: int | float, width: int | float, position: int, position_limit: int) -> tuple[list[Order], int, int]:
        sell_orders = []
        market_bids = list(order_depth.buy_orders.items())
        market_bids.sort(reverse=True)
        sell_order_volume = 0
        for bid_price, bid_volume in market_bids:
            if position <= -position_limit:
                break
            if bid_price >= fair + width:
                sell_price = bid_price
                sell_vol = min(bid_volume, position - -position_limit)
                sell_order = Order(product, sell_price, -sell_vol)
                sell_orders.append(sell_order)
                sell_order_volume += sell_vol
                position -= sell_vol
                order_depth.buy_orders[bid_price] -= sell_vol
                if order_depth.buy_orders[bid_price] == 0:
                    order_depth.buy_orders.pop(bid_price)
        
        buy_orders = []
        market_asks = list(order_depth.sell_orders.items())
        market_asks.sort()
        buy_order_volume = 0
        for ask_price, ask_volume in market_asks:
            if position >= position_limit:
                break
            if ask_price <= fair - width:
                buy_price = ask_price
                buy_vol = min(-ask_volume, position_limit - position)
                buy_order = Order(product, buy_price, buy_vol)
                buy_orders.append(buy_order)
                buy_order_volume += buy_vol
                position += buy_vol
                order_depth.sell_orders[ask_price] += buy_vol
                if order_depth.sell_orders[ask_price] == 0:
                    order_depth.sell_orders.pop(ask_price)

        return buy_orders + sell_orders, buy_order_volume, sell_order_volume

    def clear_position(self, product: str, order_depth: OrderDepth, fair: float | int, position: int, buy_order_volume: int, sell_order_volume: int) -> list[Order]:
        pos_after_take = position + buy_order_volume - sell_order_volume
        sell_orders = []
        buy_orders = []
        sell_order_volume = 0
        buy_order_volume = 0

        clear_width = 0
        
        if pos_after_take > 0:
            market_bids = list(order_depth.buy_orders.items())
            market_bids.sort(reverse=True)
            for bid_price, bid_volume in market_bids:
                if pos_after_take <= 0:
                    break
                if bid_price >= fair + clear_width:
                    sell_vol = min(bid_volume, pos_after_take)
                    sell_order = Order(product, bid_price, -sell_vol)
                    sell_orders.append(sell_order)
                    pos_after_take -= sell_vol
                    order_depth.buy_orders[bid_price] -= sell_vol
                    if order_depth.buy_orders[bid_price] == 0:
                        order_depth.buy_orders.pop(bid_price)
                    sell_order_volume += sell_vol

        elif pos_after_take < 0:
            market_asks = list(order_depth.sell_orders.items())
            market_asks.sort()
            for ask_price, ask_volume in market_asks:
                if pos_after_take >= 0:
                    break
                if ask_price <= fair - clear_width:
                    buy_vol = min(-ask_volume, -pos_after_take)
                    buy_order = Order(product, ask_price, buy_vol)
                    buy_orders.append(buy_order)
                    pos_after_take += buy_vol
                    order_depth.sell_orders[ask_price] += buy_vol
                    if order_depth.sell_orders[ask_price] == 0:
                        order_depth.sell_orders.pop(ask_price)
                    buy_order_volume += buy_vol

        return buy_orders + sell_orders, buy_order_volume, sell_order_volume

    def calc_fair_kelp(self):
        pass

    def kelp(self, order_depth: OrderDepth, position: int) -> list[Order]:
        return []

    def resin(self, order_depth: OrderDepth, position: int) -> list[Order]:
        take_orders, buy_order_volume, sell_order_volume = self.market_take(Product.RESIN, order_depth, 10000, 1, position, 20)
        clear_orders, buy_order_volume, sell_order_volume = self.clear_position(Product.RESIN, order_depth, 10000, position, buy_order_volume, sell_order_volume)

        return take_orders + clear_orders

    def run(self, state: TradingState) -> dict[str, list[Order]]:
        self.timestamp = state.timestamp
        order_depth = state.order_depths
        traderData = state.traderData

        result = {}

        for product in order_depth:
            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]

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