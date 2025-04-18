from datamodel import OrderDepth, UserId, TradingState, Order
import string
import jsonpickle
from copy import deepcopy

class Product:
    MACARONS = 'MAGNIFICENT_MACARONS'


class Trader:

    def market_take(self, product: str, order_depth: OrderDepth, fair: int | float, width: int | float, position: int, position_limit: int) -> tuple[list[Order], int, int]:
        to_buy = position_limit - position
        to_sell = position + position_limit

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        own_orders = []
        buy_order_volume = 0
        sell_order_volume = 0

        # max_buy_price = fair - width if position > self.limit * 0.5 else true_value
        # min_sell_price = fair + 1 if position < self.limit * -0.5 else true_value
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

        # if to_buy > 0 and hard_liquidate:
        #     quantity = to_buy // 2
        #     self.buy(true_value, quantity)
        #     to_buy -= quantity

        # if to_buy > 0 and soft_liquidate:
        #     quantity = to_buy // 2
        #     self.buy(true_value - 2, quantity)
        #     to_buy -= quantity

        # if to_buy > 0:
        #     popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        #     price = min(max_buy_price, popular_buy_price + 1)
        #     self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                own_orders.append(Order(product, price, -quantity))
                to_sell -= quantity
                order_depth.buy_orders[price] -= quantity
                if order_depth.buy_orders[price] == 0:
                    order_depth.buy_orders.pop(price)
                sell_order_volume += quantity

        # if to_sell > 0 and hard_liquidate:
        #     quantity = to_sell // 2
        #     self.sell(true_value, quantity)
        #     to_sell -= quantity

        # if to_sell > 0 and soft_liquidate:
        #     quantity = to_sell // 2
        #     self.sell(true_value + 2, quantity)
        #     to_sell -= quantity

        # if to_sell > 0:
        #     popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        #     price = max(min_sell_price, popular_sell_price - 1)
        #     self.sell(price, to_sell)

        return own_orders, buy_order_volume, sell_order_volume

    def clear_position(self, product: str, order_depth: OrderDepth, fair: float | int, position: int, buy_order_volume: int, sell_order_volume: int, position_limit: 20) -> list[Order]:
        pos_after_take = position + buy_order_volume - sell_order_volume
        own_orders = []

        clear_width = 0

        '''
        if position after take > 0 then we want to sell to clear inventory
        when creating sell orders, we check previous sell volume from take and add that to position to get anticipated position
        so position - sell_order_volume is the position we anticipate to enforce limit for sell orders = ant_sell_position
        so most we can sell is ant_sell_position - -position_limit
        enforce this to sanity check edge cases

        ideally we want to sell pos_after_take quantity to neutralize
        '''


        '''
        do we need to loop through everything and have this complex logic for clearing positions when we could just do the pranav clear logic

        '''
        
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
        
        '''
        let's say order depth after take and clear looks like this:

        price   | volume
        
        14      | 7
        13      | 15
        12      | 8

        09      | 6
        08      | 11
        07      | 2

        if fair is 10 to make, valid price levels to penny and joing include bidding at 9; asking at 12, 11
        for bid, only option is to join at 9, since at 10 we're not getting edge
        for ask, have to decide how much to penny at 11 and join at 12

        todo backtest these strategies:
        1. always penny if possible and only penny, only join if penny not possible
        2. divide equally between pennying and joining using some strategy

        for 1
        if best bid is 1 below fair, we can only join so join at best bid
        if best bid is more than 1 below fair, we can penny at best bid + 1

        so to be able to penny make on bid, best bid has to be minimum 2 below fair
        if 1 below fair then we can join
        
        '''
        orders = []

        penny_width = 2
        join_width = 1

        market_bids = list(order_depth.buy_orders.items())
        market_bids.sort(reverse=True)

        best_bid, _ = max(market_bids)
        if best_bid <= fair - penny_width:
            # we can penny at best_bid + 1
            bid_price = best_bid + 1
        elif best_bid <= fair - join_width:
            bid_price = best_bid
        else:
            bid_price = None

        bid_vol = position_limit - (position + buy_order_volume)

        if bid_price is not None and bid_vol > 0:
            orders.append(Order(product, bid_price, bid_vol))
            # update order depth to reflect our order
            if bid_price in order_depth.buy_orders:
                order_depth.buy_orders[bid_price] += bid_vol
            else:
                order_depth.buy_orders[bid_price] = bid_vol
            buy_order_volume += bid_vol
        
        market_asks = list(order_depth.sell_orders.items())
        market_asks.sort()

        best_ask, _ = min(market_asks)
        if best_ask >= fair + penny_width:
            # we can penny at best_ask - 1
            ask_price = best_ask - 1
        elif best_ask >= fair + join_width:
            ask_price = best_ask
        else:
            ask_price = None
        
        ask_vol = position - sell_order_volume - -position_limit
        ask_vol = -ask_vol

        if ask_price is not None and ask_vol > 0:
            orders.append(Order(product, ask_price, ask_vol))
            # update order depth to reflect our order
            if ask_price in order_depth.sell_orders:
                order_depth.sell_orders[ask_price] += ask_vol
            else:
                order_depth.sell_orders[ask_price] = ask_vol
            sell_order_volume += ask_vol

        # return [], buy_order_volume, sell_order_volume

        orders = []

        penny_bid = min(best_bid + 1, 9999)
        penny_ask = max(best_ask - 1, 10001)

        orders.append(Order(product, penny_bid, bid_vol))
        orders.append(Order(product, penny_ask, ask_vol))        
        
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