from datamodel import OrderDepth, UserId, TradingState, Order
import string
import jsonpickle


class Product:
    KELP = 'KELP'
    RESIN = 'RAINFOREST_RESIN'


class Trader:

    def market_take(self, product: str, order_depth: OrderDepth, fair: int | float, width: int | float, position: int, position_limit: int) -> list[Order]:
        sell_orders = []
        market_bids = list(order_depth.buy_orders.items())
        market_bids.sort(reverse=True)
        for bid_price, bid_volume in market_bids:
            if position <= -position_limit:
                break
            if bid_price >= fair + width:
                sell_price = bid_price
                sell_vol = min(bid_volume, position - -position_limit)
                sell_order = Order(product, sell_price, -sell_vol)
                sell_orders.append(sell_order)
                position -= sell_vol
        
        buy_orders = []
        market_asks = list(order_depth.sell_orders.items())
        market_asks.sort()
        for ask_price, ask_volume in market_asks:
            if position >= position_limit:
                break
            if ask_price <= fair - width:
                buy_price = ask_price
                buy_vol = min(ask_volume, position_limit - position)
                buy_order = Order(product, buy_price, buy_vol)
                buy_orders.append(buy_order)
                position += buy_vol
                
        return sell_orders + buy_orders

    def calc_fair_kelp(self):
        pass

    def kelp(self, order_depth: OrderDepth, position: int) -> list[Order]:
        return []

    def resin(self, order_depth: OrderDepth, position: int) -> list[Order]:
        take_orders = self.market_take(Product.RESIN, order_depth, 10000, 1, position, 20)
        return take_orders


    def run(self, state: TradingState) -> dict[str, list[Order]]:
        self.timestamp = state.timestamp
        order_depth = state.order_depths
        traderData = state.traderData

        result = {}

        for product in order_depth:
            position = state.position[product]
            order_depth = state.order_depths[product]

            if product == Product.RESIN:
                orders = self.resin(order_depth, position)

            elif product == Product.KELP:
                orders = self.kelp(order_depth, position)

            result[product] = orders
        
        print(f'Timestamp: {self.timestamp}; Results: {result}')
        conversions = 1
        return result, conversions, traderData