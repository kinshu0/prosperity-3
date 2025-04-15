from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math

from math import log, sqrt, exp
from statistics import NormalDist

class Product:
    ROCK = 'VOLCANIC_ROCK'
    VOUCHER_9500 = 'VOLCANIC_ROCK_VOUCHER_9500'
    VOUCHER_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
    VOUCHER_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
    VOUCHER_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
    VOUCHER_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'


PARAMS = {
    Product.VOUCHER_10000: {
        "mean_volatility": 0.138,
        "strike": 10000,
        "starting_time_to_expiry": 1,
        "std_window": 30,
        # "zscore_threshold": 1.5
        "zscore_threshold": 3
    },
}

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        # print(f"d1: {d1}")
        # print(f"vol: {volatility}")
        # print(f"spot: {spot}")
        # print(f"strike: {strike}")
        # print(f"time: {time_to_expiry}")
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.ROCK: 400,
            Product.VOUCHER_10000: 200,
        }

    # Returns buy_order_volume, sell_order_volume
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

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

    def take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ) -> (int, int):

        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
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
            if abs(best_bid_amount) <= adverse_volume:
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

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

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
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
            )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_coconut_coupon_mid_price(
        self, coconut_coupon_order_depth: OrderDepth, traderData: Dict[str, Any]
    ):
        if (
            len(coconut_coupon_order_depth.buy_orders) > 0
            and len(coconut_coupon_order_depth.sell_orders) > 0
        ):
            best_bid = max(coconut_coupon_order_depth.buy_orders.keys())
            best_ask = min(coconut_coupon_order_depth.sell_orders.keys())
            traderData["prev_coupon_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_coupon_price"]
        
    def coconut_hedge_orders(
        self,
        coconut_order_depth: OrderDepth,
        coconut_coupon_order_depth: OrderDepth,
        coconut_coupon_orders: List[Order],
        coconut_position: int,
        coconut_coupon_position: int,
        delta: float
    ) -> List[Order]:
        if coconut_coupon_orders == None or len(coconut_coupon_orders) == 0:
            coconut_coupon_position_after_trade = coconut_coupon_position
        else:
            coconut_coupon_position_after_trade = coconut_coupon_position + sum(order.quantity for order in coconut_coupon_orders)
        
        target_coconut_position = -delta * coconut_coupon_position_after_trade
        
        if target_coconut_position == coconut_position:
            return None
        
        target_coconut_quantity = target_coconut_position - coconut_position

        orders: List[Order] = []
        if target_coconut_quantity > 0:
            # Buy COCONUT
            best_ask = min(coconut_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_coconut_quantity),
                self.LIMIT[Product.ROCK] - coconut_position,
            )
            if quantity > 0:
                orders.append(Order(Product.ROCK, best_ask, round(quantity)))
        
        elif target_coconut_quantity < 0:
            # Sell COCONUT
            best_bid = max(coconut_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_coconut_quantity),
                self.LIMIT[Product.ROCK] + coconut_position,
            )
            if quantity > 0:
                orders.append(Order(Product.ROCK, best_bid, -round(quantity)))
        
        return orders

    def coconut_coupon_orders(
        self,
        coconut_coupon_order_depth: OrderDepth,
        coconut_coupon_position: int,
        traderData: Dict[str, Any],
        volatility: float,
    ) -> List[Order]:
        traderData['past_coupon_vol'].append(volatility)
        if len(traderData['past_coupon_vol']) < self.params[Product.VOUCHER_10000]['std_window']:
            return None, None

        if len(traderData['past_coupon_vol']) > self.params[Product.VOUCHER_10000]['std_window']:
            traderData['past_coupon_vol'].pop(0)
        
        vol_z_score = (volatility - self.params[Product.VOUCHER_10000]['mean_volatility']) / np.std(traderData['past_coupon_vol'])
        # print(f"vol_z_score: {vol_z_score}")
        # print(f"zscore_threshold: {self.params[Product.COCONUT_COUPON]['zscore_threshold']}")
        if (
            vol_z_score 
            >= self.params[Product.VOUCHER_10000]['zscore_threshold']
        ):
            if coconut_coupon_position != -self.LIMIT[Product.VOUCHER_10000]:
                target_coconut_coupon_position = -self.LIMIT[Product.VOUCHER_10000]
                if len(coconut_coupon_order_depth.buy_orders) > 0:
                    best_bid = max(coconut_coupon_order_depth.buy_orders.keys())
                    target_quantity = abs(target_coconut_coupon_position - coconut_coupon_position)
                    quantity = min(
                        target_quantity,
                        abs(coconut_coupon_order_depth.buy_orders[best_bid]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VOUCHER_10000, best_bid, -quantity)], []
                    else:
                        return [Order(Product.VOUCHER_10000, best_bid, -quantity)], [Order(Product.VOUCHER_10000, best_bid, -quote_quantity)]

        elif (
            vol_z_score
            <= -self.params[Product.VOUCHER_10000]["zscore_threshold"]
        ):
            if coconut_coupon_position != self.LIMIT[Product.VOUCHER_10000]:
                target_coconut_coupon_position = self.LIMIT[Product.VOUCHER_10000]
                if len(coconut_coupon_order_depth.sell_orders) > 0:
                    best_ask = min(coconut_coupon_order_depth.sell_orders.keys())
                    target_quantity = abs(target_coconut_coupon_position - coconut_coupon_position)
                    quantity = min(
                        target_quantity,
                        abs(coconut_coupon_order_depth.sell_orders[best_ask]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VOUCHER_10000, best_ask, quantity)], []
                    else:
                        return [Order(Product.VOUCHER_10000, best_ask, quantity)], [Order(Product.VOUCHER_10000, best_ask, quote_quantity)]

        return None, None

    def get_past_returns(
        self,
        traderObject: Dict[str, Any],
        order_depths: Dict[str, OrderDepth],
        timeframes: Dict[str, int],
    ):
        returns_dict = {}

        for symbol, timeframe in timeframes.items():
            traderObject_key = f"{symbol}_price_history"
            if traderObject_key not in traderObject:
                traderObject[traderObject_key] = []

            price_history = traderObject[traderObject_key]

            if symbol in order_depths:
                order_depth = order_depths[symbol]
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    current_price = (
                        max(order_depth.buy_orders.keys())
                        + min(order_depth.sell_orders.keys())
                    ) / 2
                else:
                    if len(price_history) > 0:
                        current_price = float(price_history[-1])
                    else:
                        returns_dict[symbol] = None
                        continue
            else:
                if len(price_history) > 0:
                    current_price = float(price_history[-1])
                else:
                    returns_dict[symbol] = None
                    continue

            price_history.append(
                f"{current_price:.1f}"
            )  # Convert float to string with 1 decimal place

            if len(price_history) > timeframe:
                price_history.pop(0)

            if len(price_history) == timeframe:
                past_price = float(price_history[0])  # Convert string back to float
                returns = (current_price - past_price) / past_price
                returns_dict[symbol] = returns
            else:
                returns_dict[symbol] = None

        return returns_dict

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)


        result = {}
        conversions = 0


        if Product.VOUCHER_10000 not in traderObject:
            traderObject[Product.VOUCHER_10000] = {
                "prev_coupon_price": 0,
                "past_coupon_vol": []
            }

        if (
            Product.VOUCHER_10000 in self.params
            and Product.VOUCHER_10000 in state.order_depths
        ):
            coconut_coupon_position = (
                state.position[Product.VOUCHER_10000]
                if Product.VOUCHER_10000 in state.position
                else 0
            )

            coconut_position = (
                state.position[Product.ROCK]
                if Product.ROCK in state.position
                else 0
            )
            # print(f"coconut_coupon_position: {coconut_coupon_position}")
            # print(f"coconut_position: {coconut_position}")
            coconut_order_depth = state.order_depths[Product.ROCK]
            coconut_coupon_order_depth = state.order_depths[Product.VOUCHER_10000]
            coconut_mid_price = (
                min(coconut_order_depth.buy_orders.keys())
                + max(coconut_order_depth.sell_orders.keys())
            ) / 2
            coconut_coupon_mid_price = self.get_coconut_coupon_mid_price(
                coconut_coupon_order_depth, traderObject[Product.VOUCHER_10000]
            )
            # tte = (
            #     self.params[Product.VOUCHER_10000]["starting_time_to_expiry"]
            #     - (state.timestamp) / 7_000_000
            # )
            # day = 1
            # # 1 year = 7 trading days
            # time_in_years = (state.timestamp + 1_000_000 * (day - 1)) / 7_000_000
            tte = (7 - state.timestamp / 1_000_000) / (1 / 250 / 10000)
            # tte = (7 - state.timestamp / 1_000_000) / (1 / 250 / 10000)

            volatility = BlackScholes.implied_volatility(
                coconut_coupon_mid_price,
                coconut_mid_price,
                self.params[Product.VOUCHER_10000]["strike"],
                tte,
            )
            delta = BlackScholes.delta(
                coconut_mid_price,
                self.params[Product.VOUCHER_10000]["strike"],
                tte,
                volatility,
            )
    
            coconut_coupon_take_orders, coconut_coupon_make_orders = self.coconut_coupon_orders(
                state.order_depths[Product.VOUCHER_10000],
                coconut_coupon_position,
                traderObject[Product.VOUCHER_10000],
                volatility,
            )

            coconut_orders = self.coconut_hedge_orders(
                state.order_depths[Product.ROCK],
                state.order_depths[Product.VOUCHER_10000],
                coconut_coupon_take_orders,
                coconut_position,
                coconut_coupon_position,
                delta
            )

            if coconut_coupon_take_orders != None or coconut_coupon_make_orders != None:
                result[Product.VOUCHER_10000] = coconut_coupon_take_orders + coconut_coupon_make_orders
                # print(f'Market buys {coconut_coupon_order_depth.buy_orders}')
                # print(f'Market sells {coconut_coupon_order_depth.sell_orders}')
                # print(f"COCONUT_COUPON: {result[Product.VOUCHER_10000]}")

            if coconut_orders != None:
                result[Product.ROCK] = coconut_orders
                # print(f"COCONUT: {result[Product.ROCK]}")

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
