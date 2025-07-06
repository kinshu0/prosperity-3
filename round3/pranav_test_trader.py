from datamodel import OrderDepth, UserId, TradingState, Order, Trade, Symbol, Listing, Observation, ProsperityEncoder, Product
from typing import List
import string
import jsonpickle
import math
from copy import deepcopy
import json
import numpy as np


from typing import Any
SQRT_2PI = math.sqrt(2*math.pi)

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Product:
    RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    INK = "SQUID_INK"

    CROISSANTS = "CROISSANTS"
    JAMS       = "JAMS"
    DJEMBES    = "DJEMBES"

    PB1 = "PICNIC_BASKET1"
    PB2 = "PICNIC_BASKET2"

Product.VOLC_ROCK                = "VOLCANIC_ROCK"
Product.VOUCHER_9500             = "VOLCANIC_ROCK_VOUCHER_9500"
Product.VOUCHER_9750             = "VOLCANIC_ROCK_VOUCHER_9750"
Product.VOUCHER_10000            = "VOLCANIC_ROCK_VOUCHER_10000"
Product.VOUCHER_10250            = "VOLCANIC_ROCK_VOUCHER_10250"
Product.VOUCHER_10500            = "VOLCANIC_ROCK_VOUCHER_10500"

VOUCHERS = {
    Product.VOUCHER_9500 :  9500,
    Product.VOUCHER_9750 :  9750,
    Product.VOUCHER_10000: 10000,
    Product.VOUCHER_10250: 10250,
    Product.VOUCHER_10500: 10500,
}


POSITION_LIMITS = {
    Product.RESIN:      50,
    Product.KELP:       50,
    Product.INK:        50,

    Product.CROISSANTS: 250,
    Product.JAMS:       350,
    Product.DJEMBES:     60,

    Product.PB1:         60,
    Product.PB2:        100,
}

OPT_CFG = {
    "edge"           : 100,   # minimum mis‑pricing (SeaShells) before we act
    "max_trade_qty"  : 10,    # per hit
    "vol_window"     : 120,   # ticks to estimate σ
    "min_sigma"      : 0.15,  # floor volatility (annualised)
    "days_left"      : 7,     # round‑3 ⇒ 5 days until expiry
}


PARAMS = {
    Product.RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": 0,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.INK: {
        'ink_change_threshold_pct': 0.015,
        'ink_window_size': 25,
        'ink_position_limit': 50,
        'clear_price_thresh': 4
    },

    # individual picnic legs – we wnat tiny passive edges
    Product.CROISSANTS: {"edge": 5},
    Product.JAMS:       {"edge": 20},
    Product.DJEMBES:    {"edge": 20},

    # basket mis‑pricing thresholds, max for 1
    Product.PB1: {"spread_threshold": 60, "max_qty_per_leg": 10},
    Product.PB2: {"spread_threshold": 50,  "max_qty_per_leg": 8},
    
}

# mapping basket to components
BASKET_RECIPE = {
    Product.PB1: {Product.CROISSANTS: 6, Product.JAMS: 3, Product.DJEMBES: 1},
    Product.PB2: {Product.CROISSANTS: 4, Product.JAMS: 2},
}

ARBITRAGE_CFG = {
    "edge"            : 10,   # minimum guaranteed profit (SeaShells) before we hit
    "max_spread_qty"  : 20,   # don’t cross the whole book in one shot
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RESIN: 50,
            Product.KELP: 50,
            Product.INK: 50
        }
        # limits for new products:
        self.LIMIT[Product.CROISSANTS] = 250
        self.LIMIT[Product.JAMS]       = 350
        self.LIMIT[Product.DJEMBES]    = 60
        self.LIMIT[Product.PB1]    = 60
        self.LIMIT[Product.PB2]    = 100

        for v in VOUCHERS:
            POSITION_LIMITS[v] = 200
        POSITION_LIMITS[Product.VOLC_ROCK] = 400

        # Add to Trader.LIMIT inside __init__
        self.LIMIT.update(POSITION_LIMITS)

    
    # ───────── small helpers ─────────
    @staticmethod
    def _best_bid_ask(depth: OrderDepth):
        best_bid = max(depth.buy_orders)  if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask

    @staticmethod
    def _mid(self, depth: OrderDepth):
        bid, ask = self._best_bid_ask(depth)
        return (bid + ask) / 2 if bid is not None and ask is not None else None

    def _cross(self, product: str, depth: OrderDepth, qty: int,
               side: str, orders: List[Order]):
        """Cross the market up to *qty* contracts (greedy fill)."""
        remaining = qty
        levels = (sorted(depth.sell_orders.items())
                  if side == 'buy'
                  else sorted(depth.buy_orders.items(), reverse=True))
        for px, vol in levels:
            if remaining <= 0:
                break
            hit = min(remaining, abs(vol))
            orders.append(Order(product, px, hit if side == 'buy' else -hit))
            remaining -= hit
            # mutate local copy so that next legs see consistent depth
            book = depth.sell_orders if side == 'buy' else depth.buy_orders
            book[px] += hit if side == 'buy' else -hit
            if book[px] == 0:
                book.pop(px)
                      
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

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None
    
    def ink_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.INK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.INK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("ink_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["ink_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("ink_last_price", None) != None:
                last_price = traderObject["ink_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.INK]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["ink_last_price"] = mmmid_price
            return fair
        return None

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

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume
    
    def mid_price(self, order_depth: OrderDepth) -> float:
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        mid = (best_ask + best_bid) / 2
        return mid

    def limit_buy(self, product: str, order_depth: OrderDepth, limit_price: int | float, position: int, position_limit: int) -> tuple[list[Order], int, int]:
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
    
    
    def limit_sell(self, product: str, order_depth: OrderDepth, limit_price: int | float, position: int, position_limit: int) -> tuple[list[Order], int, int]:
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
    
    def _best_bid_ask(self, depth: OrderDepth) -> tuple[int|None,int|None]:
        best_bid = max(depth.buy_orders)  if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask
    
    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def bs_call(self, S: float, K: float, T: float, sigma: float) -> tuple[float,float]:
        if T == 0:
            value = max(0.0, S - K)
            delta = 1.0 if S > K else 0.0
            return value, delta
        sig_sqrt = sigma * math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sig_sqrt
        d2 = d1 - sig_sqrt
        Nd1 = self.norm_cdf(d1)
        Nd2 = self.norm_cdf(d2)
        return S * Nd1 - K * Nd2, Nd1
    
    def trade_vouchers(self: "Trader", state: TradingState,
                   book: dict[str,list[Order]], mem: dict):
        if Product.VOLC_ROCK not in state.order_depths:
            return

        # ── 5.1  update rock price history & σ estimate ────────────────────────
        depth_r           = state.order_depths[Product.VOLC_ROCK]
        bid_r, ask_r      = self._best_bid_ask(depth_r)
        if bid_r is None or ask_r is None:
            return
        mid_r             = 0.5 * (bid_r + ask_r)

        hist              = mem.get("rock_hist", [])
        hist.append(mid_r)
        if len(hist) > OPT_CFG["vol_window"]:
            hist.pop(0)
        mem["rock_hist"]  = hist

        log_rets = [math.log(hist[i]/hist[i-1]) for i in range(1, len(hist))]
        if len(log_rets) < 2:          # need at least 2 returns for variance
            return
        sigma_d  = math.sqrt(sum(x*x for x in log_rets) / (len(log_rets) - 1))

        sigma    = max(OPT_CFG["min_sigma"], sigma_d * math.sqrt(252))
        T        = OPT_CFG["days_left"] / 252

        pos_r    = state.position.get(Product.VOLC_ROCK, 0)

        # ── 5.2  scan each voucher ─────────────────────────────────────────────
        for v, K in VOUCHERS.items():
            if v not in state.order_depths:
                continue
            depth_v           = state.order_depths[v]
            bid_v, ask_v      = self._best_bid_ask(depth_v)
            if bid_v is None or ask_v is None:
                continue
            mid_v             = 0.5 * (bid_v + ask_v)

            theo, delta       = self.bs_call(mid_r, K, T, sigma)
            misprice          = mid_v - theo

            pos_v             = state.position.get(v, 0)

            # overpriced ⇒ sell, underpriced ⇒ buy
            if   misprice > OPT_CFG["edge"]:
                side   = "sell"
                limit  = bid_v
                qty_v  = min(OPT_CFG["max_trade_qty"],
                            -depth_v.buy_orders[bid_v],
                            self.LIMIT[v] + pos_v)
            elif misprice < -OPT_CFG["edge"]:
                side   = "buy"
                limit  = ask_v
                qty_v  = min(OPT_CFG["max_trade_qty"],
                            depth_v.sell_orders[ask_v],
                            self.LIMIT[v] - pos_v)
            else:
                continue

            if qty_v <= 0:
                continue

            # ── 5.3  add voucher order ────────────────────────────────────────
            book[v].append(Order(v, limit,  qty_v if side=="buy" else -qty_v))
            pos_v +=  qty_v if side=="buy" else -qty_v

            # ── 5.4  delta hedge with rock ────────────────────────────────────
            hedge_qty         = round(delta * qty_v)
            if hedge_qty == 0:
                continue
            if side == "buy":          # long call ⇒ +delta ⇒ short rock
                hedge_qty        = -hedge_qty

            if hedge_qty > 0:
                hedge_qty = min(hedge_qty,
                                self.LIMIT[Product.VOLC_ROCK] - pos_r,
                                depth_r.sell_orders[ask_r])
                if hedge_qty > 0:
                    book[Product.VOLC_ROCK].append(Order(Product.VOLC_ROCK,
                                                        ask_r, hedge_qty))
                    pos_r += hedge_qty
            else:
                hedge_qty = -hedge_qty
                hedge_qty = min(hedge_qty,
                                self.LIMIT[Product.VOLC_ROCK] + pos_r,
                                depth_r.buy_orders[bid_r])
                if hedge_qty > 0:
                    book[Product.VOLC_ROCK].append(Order(Product.VOLC_ROCK,
                                                        bid_r, -hedge_qty))
                    pos_r -= hedge_qty

    ############################ 4.  Voucher arbitrage engine ####################
    def rock_and_voucher_arb(self, state: TradingState,
                            result: dict[str,list[Order]]):
        """Add rock/voucher arbitrage orders into *result* in‑place."""
        if Product.VOLC_ROCK not in state.order_depths:
            return  # underlying not visible yet

        depth_r   = deepcopy(state.order_depths[Product.VOLC_ROCK])
        bid_r,ask_r = self._best_bid_ask(depth_r)
        if bid_r is None or ask_r is None:
            return

        pos_r   = state.position.get(Product.VOLC_ROCK, 0)

        for voucher, K in VOUCHERS.items():
            if voucher not in state.order_depths:        # not tradable this tick
                continue
            depth_v        = deepcopy(state.order_depths[voucher])
            bid_v, ask_v   = self._best_bid_ask(depth_v)
            if bid_v is None or ask_v is None:
                continue
            pos_v          = state.position.get(voucher, 0)

            # ---------- case 1 : voucher too expensive  -------------------------
            # guaranteed profit  = bid_v + K - ask_r
            guaranteed      = bid_v + K - ask_r
            if guaranteed > ARBITRAGE_CFG["edge"]:
                # size limited by depth & risk limits
                size = min(
                    abs(depth_v.buy_orders[bid_v]),
                    abs(depth_r.sell_orders[ask_r]),
                    self.LIMIT[voucher] + pos_v,             # we short voucher
                    self.LIMIT[Product.VOLC_ROCK] - pos_r,   # we long rock
                    ARBITRAGE_CFG["max_spread_qty"],
                )
                if size > 0:
                    self._cross(voucher, depth_v, size, 'sell', result[voucher])
                    self._cross(Product.VOLC_ROCK, depth_r, size, 'buy', result[Product.VOLC_ROCK])
                    pos_v -= size
                    pos_r += size
                    continue  # don’t evaluate cheap case this tick

            # ---------- case 2 : voucher too cheap  -----------------------------
            # negative guaranteed profit means rock overpriced vs voucher
            guaranteed_low = ask_v + K - bid_r
            if guaranteed_low < -ARBITRAGE_CFG["edge"]:
                size = min(
                    abs(depth_v.sell_orders[ask_v]),
                    abs(depth_r.buy_orders[bid_r]),
                    self.LIMIT[Product.VOLC_ROCK] + pos_r,   # we short rock
                    self.LIMIT[voucher] - pos_v,             # we long voucher
                    ARBITRAGE_CFG["max_spread_qty"],
                )
                if size > 0:
                    self._cross(voucher, depth_v, size, 'buy',  result[voucher])
                    self._cross(Product.VOLC_ROCK, depth_r, size, 'sell', result[Product.VOLC_ROCK])
                    pos_v += size
                    pos_r -= size


    
    
    def trade_vouchers(self: "Trader", state: TradingState,
                         book: dict[str,list[Order]], mem: dict):
        
        if Product.VOLC_ROCK not in state.order_depths:
            return
        
        depth_rock = state.order_depths[Product.VOLC_ROCK]
        if not depth_rock.buy_orders or not depth_rock.sell_orders:
            return
        
        # get the rock mid price
        rock_best_bid = max(depth_rock.buy_orders)
        rock_best_ask = min(depth_rock.sell_orders)

        rock_mid_price = (rock_best_bid + rock_best_ask) / 2

        # compute TTE
        timestamp = state.timestamp
        time = int(state.timestamp)
        day = 2
        # day_decrease = (day + 2)
        tte = (8 - day - (timestamp / 1000000)) / 365

        def fitted_iv(m):
            return 0.237 * m ** 2 + -0.003 * m + 0.149
        
        def norm_cdf(x: float) -> float:
            """Standard normal cumulative distribution function using math.erf."""
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        def bs_call_price(St: float, K: float, TTE: float, sigma: float) -> float:
            """Black-Scholes price of a call option with zero interest rate."""
            if TTE <= 0 or sigma <= 0:
                return max(St - K, 0)
            
            d1 = (math.log(St / K) + 0.5 * sigma ** 2 * TTE) / (sigma * math.sqrt(TTE))
            d2 = d1 - sigma * math.sqrt(TTE)
            return St * norm_cdf(d1) - K * norm_cdf(d2)

        def implied_vol(St: float, Vt: float, K: float, TTE: float,
                        tol: float = 1e-6, max_iter: int = 100) -> float:
            """
            Calculate implied volatility from:
                St = underlying price
                Vt = option (voucher) price
                K  = strike
                TTE = time to expiry (in years)
            """
            if Vt <= 0 or St <= 0 or TTE <= 0:
                return float("nan")

            # Bisection bounds
            low = 1e-4
            high = 3.0

            for _ in range(max_iter):
                mid = (low + high) / 2
                price = bs_call_price(St, K, TTE, mid)
                diff = price - Vt

                if abs(diff) < tol:
                    return mid
                elif diff > 0:
                    high = mid
                else:
                    low = mid

            return mid
        
        threshold = 0.03  # Minimum deviation from surface to trade
        max_qty = 5

        for K, symbol in {
            9500: Product.VOUCHER_9500,
            9750: Product.VOUCHER_9750,
            10000: Product.VOUCHER_10000,
            10250: Product.VOUCHER_10250,
            10500: Product.VOUCHER_10500,
        }.items():
            if symbol not in state.order_depths:
                continue

            depth = state.order_depths[symbol]
            if not depth.buy_orders or not depth.sell_orders:
                continue

            # Step 4: Extract voucher underlying price
            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)
            mid_price = (best_bid + best_ask) / 2
            voucher_price = mid_price

            # print(np.log(rock_mid_price / voucher_price))
            # print("TTE", tte)
            # print(np.sqrt(tte))
            # m_t
            m_t = np.log(K / rock_mid_price) / np.sqrt(tte)
            print(f"m_t: {m_t}, S: {rock_mid_price}, V: {voucher_price}, T: {tte}, K: {K}, time: {timestamp}")


            volatility_expected = fitted_iv(m_t)
            # print(f"Voucher: {symbol}, m_t: {m_t}, volatility_expected: {volatility_expected}")
            volatility_actual = implied_vol(rock_mid_price, mid_price, K, tte)
            # print(f"volatility_actual: {volatility_actual}")

            diff = volatility_expected - volatility_actual

            print(f"Voucher: {symbol}, DIF: {diff}, ")


            pos = state.position.get(symbol, 0)
            pos_rock = state.position.get(Product.VOLC_ROCK, 0)

            # Step 6: Trade decision
            if diff < -threshold:
                # Option underpriced → BUY
                print("BUYING OPTION")
                qty = min(max_qty, -depth.sell_orders[best_ask], self.LIMIT[symbol] - pos)
                if qty > 0:
                    book[symbol].append(Order(symbol, best_ask, qty))
                    book[Product.VOLC_ROCK].append(Order(Product.VOLC_ROCK, rock_best_bid, -qty))  # delta hedge

            elif diff > threshold:
                print("SELLING OPTION")
                # Option overpriced → SELL
                qty = min(max_qty, depth.buy_orders[best_bid], self.LIMIT[symbol] + pos)
                if qty > 0:
                    book[symbol].append(Order(symbol, best_bid, -qty))
                    book[Product.VOLC_ROCK].append(Order(Product.VOLC_ROCK, rock_best_ask, qty))  # delta hedge
    
        


    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result: Dict[str, List[Order]] = {p: [] for p in state.order_depths}


        # if Product.RESIN in self.params and Product.RESIN in state.order_depths:
        #     resin_position = (
        #         state.position[Product.RESIN]
        #         if Product.RESIN in state.position
        #         else 0
        #     )
        #     od = deepcopy(state.order_depths[Product.RESIN])
        #     resin_take_orders, buy_order_volume, sell_order_volume = (
        #         self.take_orders(
        #             Product.RESIN,
        #             od,
        #             self.params[Product.RESIN]["fair_value"],
        #             self.params[Product.RESIN]["take_width"],
        #             resin_position,
        #         )
        #     )
        #     resin_clear_orders, buy_order_volume, sell_order_volume = (
        #         self.clear_orders(
        #             Product.RESIN,
        #             od,
        #             self.params[Product.RESIN]["fair_value"],
        #             self.params[Product.RESIN]["clear_width"],
        #             resin_position,
        #             buy_order_volume,
        #             sell_order_volume,
        #         )
        #     )
        #     resin_make_orders, _, _ = self.make_orders(
        #         Product.RESIN,
        #         od,
        #         self.params[Product.RESIN]["fair_value"],
        #         resin_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #         self.params[Product.RESIN]["disregard_edge"],
        #         self.params[Product.RESIN]["join_edge"],
        #         self.params[Product.RESIN]["default_edge"],
        #         True,
        #         self.params[Product.RESIN]["soft_position_limit"],
        #     )
        #     result[Product.RESIN] = (
        #         resin_take_orders + resin_clear_orders + resin_make_orders
        #     )

        # if Product.KELP in self.params and Product.KELP in state.order_depths:
        #     kelp_position = (
        #         state.position[Product.KELP]
        #         if Product.KELP in state.position
        #         else 0
        #     )
        #     kelp_fair_value = self.kelp_fair_value(
        #         state.order_depths[Product.KELP], traderObject
        #     )
        #     od = deepcopy(state.order_depths[Product.KELP])
        #     kelp_take_orders, buy_order_volume, sell_order_volume = (
        #         self.take_orders(
        #             Product.KELP,
        #             od,
        #             kelp_fair_value,
        #             self.params[Product.KELP]["take_width"],
        #             kelp_position,
        #             self.params[Product.KELP]["prevent_adverse"],
        #             self.params[Product.KELP]["adverse_volume"],
        #         )
        #     )
        #     kelp_clear_orders, buy_order_volume, sell_order_volume = (
        #         self.clear_orders(
        #             Product.KELP,
        #             od,
        #             kelp_fair_value,
        #             self.params[Product.KELP]["clear_width"],
        #             kelp_position,
        #             buy_order_volume,
        #             sell_order_volume,
        #         )
        #     )
        #     kelp_make_orders, _, _ = self.make_orders(
        #         Product.KELP,
        #         od,
        #         kelp_fair_value,
        #         kelp_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #         self.params[Product.KELP]["disregard_edge"],
        #         self.params[Product.KELP]["join_edge"],
        #         self.params[Product.KELP]["default_edge"],
        #     )
        #     result[Product.KELP] = (
        #         kelp_take_orders + kelp_clear_orders + kelp_make_orders
        #     )

        # if Product.INK in self.params and Product.INK in state.order_depths:

        #     ink_position = (
        #         state.position[Product.INK]
        #         if Product.INK in state.position
        #         else 0
        #     )

        #     od = deepcopy(state.order_depths[Product.INK])

        #     orders = self.ink(od, ink_position, traderObject)

        #     result[Product.INK] = orders


        # mids = {}
        # for leg in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]:
        #     if leg not in state.order_depths:
        #         continue
        #     depth = state.order_depths[leg]
        #     mid = self._mid(self, depth)
        #     if mid is None:
        #         continue
        #     mids[leg] = mid
        #     edge = self.params[leg]['edge']
        #     pos  = state.position.get(leg, 0)
        #     size = max(1, int(0.2 * (self.LIMIT[leg] - abs(pos))))
        #     if size == 0:
        #         continue
        #     # if product is Jams, then the buy edge is 150
        #     if leg == Product.JAMS:
        #         result[leg].append(Order(leg, math.floor(mid - edge) - 150,  size))
        #         result[leg].append(Order(leg, math.ceil (mid + edge), -size))
        #     else:
        #         result[leg].append(Order(leg, math.floor(mid - edge),  size))
        #         result[leg].append(Order(leg, math.ceil (mid + edge), -size))

        # # BASKEY we try arbitrage
        # for basket, recipe in BASKET_RECIPE.items():
        #     if basket not in state.order_depths or not all(c in mids for c in recipe):
        #         continue
        #     fv = sum(qty * mids[c] for c, qty in recipe.items())
        #     depth_b = state.order_depths[basket]
        #     bid_b, ask_b = self._best_bid_ask(depth_b)
        #     if bid_b is None or ask_b is None:
        #         continue
        #     thresh = self.params[basket]['spread_threshold']
        #     max_leg = self.params[basket]['max_qty_per_leg']
        #     pos_b   = state.position.get(basket, 0)

        #     # if basket is rich then sell basket / buy legs
        #     if bid_b - fv > thresh:
        #         qty_b = min(depth_b.buy_orders[bid_b], self.LIMIT[basket] + pos_b, max_leg)
        #         if qty_b > 0:
        #             result[basket].append(Order(basket, bid_b, -qty_b))
        #             for leg, mult in recipe.items():
        #                 depth_leg = deepcopy(state.order_depths[leg])
        #                 leg_pos   = state.position.get(leg, 0)
        #                 qty_leg   = min(qty_b * mult, self.LIMIT[leg] - leg_pos)
        #                 self._cross(leg, depth_leg, qty_leg, 'buy', result[leg])

        #     # if basket cheap gpal is to buy basket / sell legs
        #     elif ask_b - fv < -thresh:
        #         qty_b = min(-depth_b.sell_orders[ask_b], self.LIMIT[basket] - pos_b, max_leg)
        #         if qty_b > 0:
        #             result[basket].append(Order(basket, ask_b, qty_b))
        #             for leg, mult in recipe.items():
        #                 depth_leg = deepcopy(state.order_depths[leg])
        #                 leg_pos   = state.position.get(leg, 0)

        #                 qty_leg   = min(qty_b * mult, leg_pos + self.LIMIT[leg])
        #                 self._cross(leg, depth_leg, qty_leg, 'sell', result[leg])
        
        # VOUCHER ARBITRAGE
        # self.rock_and_voucher_arb(state, result)
        # self.trade_vouchers(state, result, traderObject)

        self.trade_vouchers(state, result, traderObject)

        logger.print("seen:", state.order_depths.keys())



        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        logger.flush(
            state,
            result,
            conversions,
            traderData
        )

        return result, conversions, traderData
