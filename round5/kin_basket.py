from datamodel import OrderDepth, UserId, TradingState, Order, Trade
import string
import jsonpickle
from copy import deepcopy
import numpy as np
import pandas as pd



class OrderType:
    BUY = 'BUY'
    SELL = 'SELL'


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
    

    def buy_synth(self, limit_price: int, positions: dict[str, int], synth_od: OrderDepth, weights: dict[str, int]):
        
        can_buy = 999999

        for prod, weight in weights.items():
            # if weight is greater than 0 we want to look at market sells to match
            # don't want to exceed upper position limit
            if weight > 0:
                chunks = POSITION_LIMIT[prod] - positions[prod]
            # if weight is less than 0 we want to look at market buys and not exceed lower position limit
            else:
                chunks = positions[prod] - -POSITION_LIMIT[prod]
            can_buy = min(chunks // abs(weight), can_buy)
        
        if can_buy == 0:
            return {}

        # get synth volume to buy
        # we match with synth market sells to buy synth
        # only part different between buy and sell?
        synth_market_sells = synth_od.sell_orders
        acc_synth_market_sells = [(p, q) for p, q in synth_market_sells.items() if p <= limit_price]

        acc_market_sell_vol = sum(x[1] for x in acc_synth_market_sells) if acc_synth_market_sells else 0

        to_buy = min(-acc_market_sell_vol, can_buy)

        if to_buy == 0:
            return {}

        result_orders = {}

        for prod, weight in weights.items():
            # here q would automatically be negative if it's negative weight
            q = weight * to_buy
            match_price = 0 if weight < 0 else 9999999
            result_orders[prod] = [Order(prod, match_price, q)]

        return result_orders

    def sell_synth(self, limit_price: int, positions: dict[str, int], synth_od: OrderDepth, weights: dict[str, int]):

        weights = {prod: -weights[prod] for prod in weights}
        
        can_sell = 999999

        for prod, weight in weights.items():
            if weight > 0:
                chunks = POSITION_LIMIT[prod] - positions[prod]
            else:
                chunks = positions[prod] - -POSITION_LIMIT[prod]
            can_sell = min(chunks // abs(weight), can_sell)

        synth_market_buys = synth_od.buy_orders
        acc_synth_market_buys = [(p, q) for p, q in synth_market_buys.items() if p >= limit_price]

        acc_market_buy_vol = sum(x[1] for x in acc_synth_market_buys) if acc_synth_market_buys else 0

        to_buy = min(acc_market_buy_vol, can_sell)

        if to_buy == 0:
            return {}
        
        result_orders = {}

        for prod, weight in weights.items():
            q = weight * to_buy
            match_price = 0 if weight < 0 else 9999999
            result_orders[prod] = [Order(prod, match_price, q)]

        return result_orders
    
    
    def pb2(self, order_depths: dict[str, OrderDepth], trader_data: dict, positions):

        # PB2_SYNTH_WEIGHTS = {
        #     Product.CROISSANTS: 4,
        #     Product.JAMS: 2
        # }

        # spread_weights = {
        #     Product.PB2: 1,
        #     **{p: -w for p, w in PB2_SYNTH_WEIGHTS.items()}
        # }

        # spread_weights = {
        #     Product.PB2: 1,
        #     **{p: -w for p, w in PB2_SYNTH_WEIGHTS.items()}
        # }

        spreads = [
            {
                'spread_weights': {'PICNIC_BASKET1': 1, 'CROISSANTS': -6, 'JAMS': -3, 'DJEMBES': -1},
                'mean': -38.2404,
                # 'median': -50, # pranav backtested median
                'median': -63.4594,
                'std': 120.3913
            },
            {
                'spread_weights': {'PICNIC_BASKET2': 1, 'CROISSANTS': -4, 'JAMS': -2},
                'mean': 76.0372,
                'median': 78.9464,
                'std': 64.0115
            },
            {
                'spread_weights': {'PICNIC_BASKET1': 1, 'PICNIC_BASKET2': -1, 'CROISSANTS': -2, 'JAMS': -1, 'DJEMBES': -1},
                'mean': -114.2776,
                'median': -162.5233,
                'std': 154.8832
            },
            {
                'spread_weights': {'PICNIC_BASKET1': 2, 'PICNIC_BASKET2': -3, 'DJEMBES': -2},
                'mean': -304.5923,
                'median': -402.2917,
                'std': 356.7032
            },
            {
                'spread_weights': {'DJEMBES': 1, 'PICNIC_BASKET1': -1, 'PICNIC_BASKET2': 1.5},
                'mean': 152.2962,
                'median': 201.1458,
                'std': 178.3516
            }
        ]

        selected_spread = 1

        spread_weights = spreads[selected_spread]['spread_weights']
        mean = spreads[selected_spread]['mean']
        median_price = spreads[selected_spread]['median']
        std = spreads[selected_spread]['std']

        
        spread_od = self.construct_synth_od(order_depths, spread_weights)

        # swmid_spread = self.swmid(spread_od)

        thresh = 2 * std

        synth_buys = self.buy_synth(median_price - std * 2, positions, spread_od, spread_weights)
        synth_sells = self.sell_synth(median_price + std * 2, positions, spread_od, spread_weights)

        # exp_price = 76
        # std = 64

        # exp_price = spre

        # thresh = 1.5 * std

        # synth_buys = self.buy_synth(exp_price - std * 2, positions, spread_od, spread_weights)
        # synth_sells = self.sell_synth(exp_price + std * 2, positions, spread_od, spread_weights)


        # if synth_buys or synth_sells:
        #     print(synth_buys)
        #     print(synth_sells)

        if synth_buys:
            # print('BUYING SPREAD', end=' ')
            return synth_buys
        
        # print('SELLING SPREAD', end=' ')
        return synth_sells



        # swmid_comp_od = self.swmid_synth(spread_weights, order_depths)

        # print(f'{swmid_spread_od=}')
        # print(f'{swmid_comp_od=}')

        
        # pb2_fair = self.fair_b2(order_depths, trader_data)
        
        # pb2_mid = self.swmid(order_depths[Product.PB2])

        # pb2_orders = []
        # croissant_orders = []
        # jam_orders = []
        # djembe_orders = []

        # # diff = pb2_mid - pb2_fair
        # pb2synth_swmid = self.swmid_synth(PB2_SYNTH_WEIGHTS, order_depths)

        # diff = pb2_mid - pb2synth_swmid
        # # diff = pb2_mid - pb2_fair

        # quantity = 1

        # mean_diff = 76
        # std = 64

        # thresh = 2 * std
        # # diff = abs(diff)

        # if diff > mean_diff + thresh:
        #     print('SELLING SPREAD')
        #     print(order_depths[Product.PB2].buy_orders)
        #     print(order_depths[Product.CROISSANTS].sell_orders)
        #     print(order_depths[Product.JAMS].sell_orders)
            
        #     pb2_orders.append(Order(Product.PB2, 0, -1*quantity))
        #     croissant_orders.append(Order(Product.CROISSANTS, 999999, 4*quantity))
        #     jam_orders.append(Order(Product.JAMS, 999999, 2*quantity))

        # elif diff < mean_diff - thresh:
        #     print('BUYING SPREAD')
        #     print(order_depths[Product.PB2].sell_orders)
        #     print(order_depths[Product.CROISSANTS].buy_orders)
        #     print(order_depths[Product.JAMS].buy_orders)

        #     pb2_orders.append(Order(Product.PB2, 999999, 1*quantity))
        #     croissant_orders.append(Order(Product.CROISSANTS, 0, -4*quantity))
        #     jam_orders.append(Order(Product.JAMS, 0, -2*quantity))

        # return {
        #     Product.PB2: pb2_orders,
        #     Product.CROISSANTS: croissant_orders,
        #     Product.JAMS: jam_orders,
        #     Product.DJEMBES: djembe_orders
        # }

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

            basket_orders = self.pb2(order_depths, trader_data, state.position)

            # if basket_orders:
            #     print(basket_orders)

            for prod, orders in basket_orders.items():
                for i in range(len(orders)):
                    orders[i].quantity = int(orders[i].quantity)
                result[prod] = result.get(prod, []) + orders

        traderData = jsonpickle.encode(trader_data)
        conversions = 0

        return result, conversions, traderData