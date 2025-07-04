from datamodel import OrderDepth, UserId, TradingState, Order, Trade
import string
import jsonpickle
from copy import deepcopy
import numpy as np
import pandas as pd

class Product:
    RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    INK = "SQUID_INK"

    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PB1 = "PICNIC_BASKET1"
    PB2 = "PICNIC_BASKET2"
    PB2SYNTH = "PB2SYNTH"

    MAC = "MAGNIFICENT_MACARONS"

    VOLC_ROCK = "VOLCANIC_ROCK"
    VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"


POSITION_LIMIT = {
    Product.RESIN: 50,
    Product.KELP: 50,
    Product.INK: 50,
    Product.CROISSANTS: 250,
    Product.JAMS: 350,
    Product.DJEMBES: 60,
    Product.PB1: 60,
    Product.PB2: 100,
}

DEFAULT_PARAMS = {


}


PB2_SYNTH_WEIGHTS = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2
}

class Trader:
    def __init__(self, PARAMS=None):
        self.PARAMS = PARAMS if PARAMS is not None else DEFAULT_PARAMS

               
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
    ) -> tuple[int, int]:
        position_limit = POSITION_LIMIT[product]

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

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> tuple[list[Order], int, int]:
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

    def limit_buy(
        self,
        product: str,
        order_depth: OrderDepth,
        limit_price: int | float,
        position: int,
        position_limit: int,
        adverse_width: float = 999999,
    ) -> tuple[list[Order], int, int]:
        to_buy = position_limit - position
        market_sell_orders = sorted(order_depth.sell_orders.items())

        own_orders = []
        buy_order_volume = 0

        max_buy_price = limit_price

        for price, volume in market_sell_orders:
            if to_buy > 0 and price <= max_buy_price and limit_price - price <= adverse_width:
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
        adverse_width: float = 999999,
    ) -> tuple[list[Order], int, int]:
        to_sell = position - -position_limit

        market_buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        own_orders = []
        sell_order_volume = 0

        min_sell_price = limit_price

        for price, volume in market_buy_orders:
            if to_sell > 0 and price >= min_sell_price and price - limit_price <= adverse_width:
                quantity = min(to_sell, volume)
                own_orders.append(Order(product, price, -quantity))
                to_sell -= quantity
                order_depth.buy_orders[price] -= quantity
                if order_depth.buy_orders[price] == 0:
                    order_depth.buy_orders.pop(price)
                sell_order_volume += quantity

        return own_orders

    def person_trades(self, state: TradingState, result: dict):

        signals = []

        OLIVIA = 'Olivia'

        combined_trades: dict[str, list[Trade]] = {}

        for prod, trades in state.market_trades.items():
            combined_trades[prod] = combined_trades.get(prod, []) + trades

        for prod, trades in state.own_trades.items():
            combined_trades[prod] = combined_trades.get(prod, []) + trades


        for prod, trades in combined_trades.items():
            olivia_trades = []
            for t in trades:
                if t.buyer == t.seller:
                    continue
                elif t.buyer == OLIVIA:
                    olivia_trades.append(t)
                elif t.seller == OLIVIA:
                    t.quantity *= -1
                    olivia_trades.append(t)
                    
            if len(olivia_trades) == 0:
                continue

            trade = sorted(olivia_trades, key=lambda t: abs(t.quantity))[0]
            quantity = trade.quantity
            price = trade.price
            if quantity > 0:
                signals.append(('BUY', prod, price * 1.2))
            elif quantity < 0:
                signals.append(('SELL', prod, price * 0.8))

        for signal, prod, price in signals:
            od = deepcopy(state.order_depths.get(prod, {}))
            pos = state.position.get(prod, 0)
            prod_orders = []
            if signal == 'BUY':
                prod_orders += self.limit_buy(prod, od, price, pos, POSITION_LIMIT[prod])
            elif signal == 'SELL':
                prod_orders += self.limit_sell(prod, od, 0, pos, POSITION_LIMIT[prod])
            
            result[prod] = result.get(prod, []) + prod_orders

    def swmid(self, order_depth: OrderDepth) -> float:
        best_ask, ask_vol = min(order_depth.sell_orders.items())
        best_bid, bid_vol = max(order_depth.buy_orders.items())

        return (best_ask * bid_vol + best_bid * -ask_vol) / (bid_vol + -ask_vol)

    def get_price_history(self, product: str, trader_data: dict) -> list:        
        return trader_data.get(f'{product}_prices', [])

    def update_price_history(self, price: int, product: str, max_length: int, trader_data: dict):
        price_history = self.get_price_history(product, trader_data)
        updated_price_history = price_history + [price]
        if len(updated_price_history) > max_length:
            updated_price_history.pop(0)
        trader_data[f'{product}_prices'] = updated_price_history

    def swmid_synth(self, synth_weights: dict, order_depths: dict[str, OrderDepth]) -> float:
        synth_mid = 0
        for prod, weight in synth_weights.items():
            od = order_depths[prod]
            price = self.swmid(od)
            synth_mid += price * weight
        return synth_mid
    

    def fair_b2(self, order_depths: dict[str, OrderDepth], trader_data: dict):

        max_prices_len = 21

        beta = 0.97967075

        synth_mid = self.swmid_synth(PB2_SYNTH_WEIGHTS, order_depths)
        
        self.update_price_history(synth_mid, Product.PB2SYNTH, max_prices_len, trader_data)

        synth_prices = self.get_price_history(Product.PB2SYNTH, trader_data)

        b2_mid = self.swmid(order_depths[Product.PB2])

        if len(synth_prices) < max_prices_len: 
            return b2_mid
        
        pct_synth = synth_prices[-1] / synth_prices[-max_prices_len] - 1

        pct_b2 = pct_synth * beta

        basket_fair = b2_mid * (1 + pct_b2)

        return basket_fair
    
    def get_trader_var(self, var_name: str, trader_data: dict) -> float:
        trader_var = trader_data.get(var_name, 0)
        return trader_var
    
    def set_trader_var(self, var_name: str, value, trader_data: dict):
        trader_data[var_name] = value
        return trader_data
    
    
    
    def pb2(self, order_depths: dict[str, OrderDepth], trader_data: dict, position: int):
        
        pb2_fair = self.fair_b2(order_depths, trader_data)
        
        pb2_mid = self.swmid(order_depths[Product.PB2])

        '''
        TODO trade entire spread
        if diff greater than threshold: sell pb2 buy synth, has to be fixed quantity

        '''

        pb2_orders = []
        croissant_orders = []
        jam_orders = []
        djembe_orders = []

        diff = pb2_mid - pb2_fair

        quantity = 1

        thresh = 20
        # thresh = 10
        # diff = abs(diff)

        if diff > thresh:
            pb2_orders.append(Order(Product.PB2, 0, -1*quantity))
            croissant_orders.append(Order(Product.CROISSANTS, 999999, 4*quantity))
            jam_orders.append(Order(Product.JAMS, 999999, 2*quantity))
        elif diff < thresh:
        # elif diff < -thresh:
            pb2_orders.append(Order(Product.PB2, 999999, 1*quantity))
            croissant_orders.append(Order(Product.CROISSANTS, 0, -4*quantity))
            jam_orders.append(Order(Product.JAMS, 0, -2*quantity))

        return {
            Product.PB2: pb2_orders,
            Product.CROISSANTS: croissant_orders,
            Product.JAMS: jam_orders,
            Product.DJEMBES: djembe_orders
        }


        # if diff > 

        # # pb2_mid = self.swmid(order_depths[Product.PB2])

        # buy_orders = self.limit_buy(
        #     Product.PB2,
        #     order_depths[Product.PB2],
        #     pb2_fair - 5,
        #     position,
        #     POSITION_LIMIT[Product.PB2],
        # )

        # sell_orders = self.limit_sell(
        #     Product.PB2,
        #     order_depths[Product.PB2],
        #     pb2_fair + 5,
        #     position,
        #     POSITION_LIMIT[Product.PB2],
        # )

        # take_orders = buy_orders + sell_orders

        # # TODO: clear

        # take_orders, buy_order_volume, sell_order_volume = self.take_orders(
        #     Product.PB2,
        #     order_depths[Product.PB2],
        #     pb2_fair,
        #     10,
        #     position
        # )

        # return take_orders

    def run(self, state: TradingState) -> dict[str, list[Order]]:
        self.timestamp = state.timestamp
        trader_data = {}
        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)

        result = {}

        for prod in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES, Product.PB1, Product.PB2]:
            state.position[prod] = state.position.get(prod, 0)

        # self.person_trades(state, result)

        if Product.PB2 in state.order_depths:
            order_depths = deepcopy(state.order_depths)
            # result[Product.PB2] = self.pb2(order_depths, trader_data, state.position[Product.PB2])
            for prod, orders in self.pb2(order_depths, trader_data, state.position[Product.PB2]).items():
                result[prod] = result.get(prod, []) + orders

        traderData = jsonpickle.encode(trader_data)
        conversions = 0

        return result, conversions, traderData