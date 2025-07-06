import json
import math
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from statistics import NormalDist
from typing import Any, TypeAlias

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

def BS_CALL(S, K, T, r, sigma):
    N = NormalDist().cdf
    d1 = (math.log(S/K) + (r + sigma**2/2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * N(d1) - K * math.exp(-r*T)* N(d2)


class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0

        self.act(state)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return list(self.window)

    def load(self, data: JSON) -> None:
        self.window = deque(data)

class AmethystsStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10_000

class StarfruitStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round((popular_buy_price + popular_sell_price) / 2)

class OrchidsStrategy(Strategy):
    def act(self, state: TradingState) -> None:
        position = state.position.get(self.symbol, 0)
        self.convert(-1 * position)

        obs = state.observations.conversionObservations.get(self.symbol, None)
        if obs is None:
            return

        buy_price = obs.askPrice + obs.transportFees + obs.importTariff
        self.sell(max(int(obs.bidPrice - 0.5), int(buy_price + 1)), self.limit)

class SignalStrategy(Strategy):
    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.sell_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position

        self.buy(price, to_buy)

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.buy_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_sell = self.limit + position

        self.sell(price, to_sell)

class GiftBasketStrategy(SignalStrategy):
    def act(self, state: TradingState) -> None:
        if any(symbol not in state.order_depths for symbol in ["CHOCOLATE", "STRAWBERRIES", "ROSES", "GIFT_BASKET"]):
            return

        chocolate = self.get_mid_price(state, "CHOCOLATE")
        strawberries = self.get_mid_price(state, "STRAWBERRIES")
        roses = self.get_mid_price(state, "ROSES")
        gift_basket = self.get_mid_price(state, "GIFT_BASKET")

        diff = gift_basket - 4 * chocolate - 6 * strawberries - roses

        # if diff < 260:
        #     self.go_long(state)
        # elif diff > 355:
        #     self.go_short(state)

        long_threshold, short_threshold = {
            "CHOCOLATE": (230, 355),
            "STRAWBERRIES": (195, 485),
            "ROSES": (325, 370),
            "GIFT_BASKET": (290, 355),
        }[self.symbol]

        if diff < long_threshold:
            self.go_long(state)
        elif diff > short_threshold:
            self.go_short(state)

        # premium, threshold = {
        #     "CHOCOLATE": (285, 0.19),
        #     "STRAWBERRIES": (340, 0.43),
        #     "ROSES": (350, 0.05),
        #     "GIFT_BASKET": (325, 0.12),
        # }[self.symbol]

        # if diff < premium * (1.0 - threshold):
        #     self.go_long(state)
        # elif diff > premium * (1.0 + threshold):
        #     self.go_short(state)

        # if diff < 355 * 0.9:
        #     self.go_long(state)
        # elif diff > 355 * 1.1:
        #     self.go_short(state)
class CoconutStrategy(SignalStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.threshold = None
        self.delta_target = None  # 💡 new target position

    def set_target_delta(self, delta: float, option_position: int) -> None:
        self.delta_target = round(delta * option_position)

    def act(self, state: TradingState) -> None:
        position = state.position.get(self.symbol, 0)

        if self.delta_target is not None:
            diff = self.delta_target - position
            order_depth = state.order_depths[self.symbol]
            if diff > 0:
                # Need to buy coconuts
                price = min(order_depth.sell_orders.keys())
                self.buy(price, min(diff, self.limit - position))
            elif diff < 0:
                # Need to sell coconuts
                price = max(order_depth.buy_orders.keys())
                self.sell(price, min(-diff, self.limit + position))
        else:
            # fallback logic if no delta set — optional
            price = self.get_mid_price(state, self.symbol)
            if self.threshold is None:
                self.threshold = price
            if price > self.threshold:
                self.go_long(state)
            elif price < self.threshold:
                self.go_short(state)

    def save(self) -> JSON:
        return {"threshold": self.threshold, "delta_target": self.delta_target}

    def load(self, data: JSON) -> None:
        if isinstance(data, dict):
            self.threshold = data.get("threshold", None)
            self.delta_target = data.get("delta_target", None)

class CoconutCouponStrategy(SignalStrategy):
    def act(self, state: TradingState) -> None:
        if any(
            symbol not in state.order_depths or
            len(state.order_depths[symbol].buy_orders) == 0 or
            len(state.order_depths[symbol].sell_orders) == 0
            for symbol in ["VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_10000"]
        ):
            return

        coco = self.get_mid_price(state, "VOLCANIC_ROCK")
        coup = self.get_mid_price(state, "VOLCANIC_ROCK_VOUCHER_10000")

        S = coco
        K = 10000
        day = 1
        T = 8 - day - state.timestamp / 1_000_000
        r = 0
        sigma = 0.137766

        # Compute d1 and delta
        d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
        delta = NormalDist().cdf(d1)

        # 💡 SET HEDGE TARGET
        coconut_strategy = self.get_coconut_strategy()  # defined below
        option_position = state.position.get("VOLCANIC_ROCK_VOUCHER_10000", 0)
        coconut_strategy.set_target_delta(delta, option_position)

        fair = BS_CALL(S, K, T, r, sigma)

        if coup > fair + 2:
            self.go_short(state)
        elif coup < fair - 2:
            self.go_long(state)

    def get_coconut_strategy(self) -> CoconutStrategy:
        # Assumes your Trader sets this back-reference
        return self.coconut_strategy_ref

class Trader:
    def __init__(self) -> None:
        limits = {
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
        }

        self.coconut_strategy = CoconutStrategy("VOLCANIC_ROCK", limits["VOLCANIC_ROCK"])
        self.coupon_strategy = CoconutCouponStrategy("VOLCANIC_ROCK_VOUCHER_10000", limits["VOLCANIC_ROCK_VOUCHER_10000"])
        self.coupon_strategy.coconut_strategy_ref = self.coconut_strategy  # 💥 connect hedge controller

        self.strategies: dict[Symbol, Strategy] = {
            "VOLCANIC_ROCK": self.coconut_strategy,
            "VOLCANIC_ROCK_VOUCHER_10000": self.coupon_strategy,
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            if symbol in state.order_depths:
                strategy_orders, strategy_conversions = strategy.run(state)
                orders[symbol] = strategy_orders
                conversions += strategy_conversions

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        return orders, conversions, trader_data
