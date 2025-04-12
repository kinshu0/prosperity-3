from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import math
from copy import deepcopy
import statistics


import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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
    SQUID_INK = "SQUID_INK"

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
        "adverse_volume": 20,
        "reversion_beta": -0.18172393033850867,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
     Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 30,
        "reversion_beta": -0.18172393033850867,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.RESIN: 50, Product.KELP: 50, Product.SQUID_INK: 50}

        # ── Squid‑Ink mean‑reversion knobs (instance attrs so no AttributeError) ──
        # if self.params.WINDOW is not None, set it, else default to 10
        self.Window = self.params.get("WINDOW", 50)
        self.Z_ENTRY = self.params.get("Z_ENTRY", 2.5)
        self.Z_EXIT = self.params.get("Z_EXIT", 0.2)
        self.MIN_VOL = self.params.get("MIN_VOL", 1)



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


    def take_best_orders_all(
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
            # sort sell orders by price
            cycle_sell_orders = dict(
                sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
            )

            cur_position = position

            for ask, ask_amount in cycle_sell_orders.items():
                # best_ask = min(order_depth.sell_orders.keys())
                # best_ask_amount = -1 * order_depth.sell_orders[best_ask]
                best_ask = ask
                best_ask_amount = -1 * ask_amount

                if cur_position >= position_limit:
                    break

                if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                    if best_ask <= fair_value - take_width:
                        quantity = min(
                            best_ask_amount, position_limit - cur_position
                        )  # max amt to buy
                        if quantity > 0:
                            orders.append(Order(product, best_ask, quantity))
                            buy_order_volume += quantity
                            cur_position += quantity
                            order_depth.sell_orders[best_ask] += quantity
                            if order_depth.sell_orders[best_ask] == 0:
                                del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
             # sort sell orders by price
            cycle_buy_orders = dict(
                sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
            )

            cur_position = position

            for bid, bid_amount in cycle_buy_orders.items():

                best_bid = bid
                best_bid_amount = bid_amount
                if cur_position <= -1 * position_limit:
                    break

                if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                    if best_bid >= fair_value + take_width:
                        quantity = min(
                            best_bid_amount, position_limit + cur_position
                        )  # should be the max we can sell
                        if quantity > 0:
                            orders.append(Order(product, best_bid, -1 * quantity))
                            sell_order_volume += quantity
                            cur_position -= quantity
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
    
    def squid_fair_value(self, order_depth: OrderDepth, traderObject) -> float:

        # find the mean of the

        return round(normal_fair_price)

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

    def take_orders_resin(
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
        position_limit = self.LIMIT[product]

        for ask, ask_amount in order_depth.sell_orders.items():
            if int(ask) < fair_value and position + buy_order_volume <= position_limit:
                order_amt = min(position_limit - position, -ask_amount)
                buy_order_volume += order_amt
                print("BUY", str(-ask_amount) + "x", ask)
                orders.append(Order(product, ask, -ask_amount))
        
        # SELL stuff        
        for bid, bid_amount in order_depth.buy_orders.items():
            if int(bid) > fair_value and position - sell_order_volume >= -position_limit:
                order_amt = max(-position_limit - position - sell_order_volume, -bid_amount)
                sell_order_volume += -1 * order_amt
                print("SELL", str(-bid_amount) + "x", bid)
                orders.append(Order(product, bid, -bid_amount))

        return orders, buy_order_volume, sell_order_volume



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

        buy_order_volume, sell_order_volume = self.take_best_orders_all(
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

    def chat_get_squid_fair_value(self, order_depth: OrderDepth, traderObject: dict) -> float:
        # Step 1: Base mid-price
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return traderObject.get("ink_last_mid", 0) or 0  # fallback
        

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # Step 2: Order book imbalance
        bid_vol = sum(order_depth.buy_orders[p] for p in sorted(order_depth.buy_orders.keys(), reverse=True)[:5])
        ask_vol = sum(-order_depth.sell_orders[p] for p in sorted(order_depth.sell_orders.keys())[:5])
        imbalance_sign = 1 if bid_vol - ask_vol > 10 else -1 if ask_vol - bid_vol > 10 else 0
        traderObject["imbalance_sign"] = imbalance_sign

        # if bid_vol > 15:
        #     traderObject["ink_last_block_sign"] = bid_vol / 15 + 1 
        # elif ask_vol > 15:
        #     traderObject["ink_last_block_sign"] = -ask_vol / 15 + 1 

        # Step 3: Momentum sign
        last_mid = traderObject.get("ink_last_mid", None)
        if last_mid is not None:
            delta1 = mid_price - last_mid
            delta2 = traderObject.get("ink_delta1", 0)
            mom_sign = 1 if delta1 > 0 and delta2 > 0 else -1 if delta1 < 0 and delta2 < 0 else 0
            traderObject["ink_mom_sign"] = mom_sign
            traderObject["ink_delta1"] = delta1
        else:
            mom_sign = 0
            traderObject["ink_mom_sign"] = 0
            traderObject["ink_delta1"] = 0

        traderObject["ink_last_mid"] = mid_price

        # Step 4: Combine signals
        block_sign = traderObject.get("ink_last_block_sign", 0)
        mom_sign   = traderObject.get("ink_mom_sign", 0)
        imbalance_sign = traderObject.get("imbalance_sign", 0)

        fair_value = mid_price
        fair_value += 1.2 * block_sign
        fair_value += 0.8 * mom_sign
        fair_value += 0.6 * imbalance_sign

        fair_value = mid_price + 0.5 * block_sign + 0.2 * imbalance_sign


        return round(fair_value)
    
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

    
    # --------------------------------------------------------------------
    #  Squid Ink fair‑value + order logic (mean‑reversion with z‑score)
    # --------------------------------------------------------------------
    def _squid_orders(self,state: TradingState,trader_obj,) -> List[Order]:
        product = Product.SQUID_INK
        depth = state.order_depths[product]
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return []  # no actionable market yet

        mid = (best_bid + best_ask) / 2


        # rolling history kept inside trader_obj["ink_hist"]
        hist: List[float] = trader_obj.get("ink_hist", [])
        if len(hist) > self.Window:
            hist = hist[-self.Window :]
        trader_obj["ink_hist"] = hist  # save back for next tick

        # need at least 5 points to compute stdev robustly
        if len(hist) < 5:
            hist.append(mid)
            return []

        mean = statistics.mean(hist)
        std = statistics.stdev(hist) or 1.0
        z = (mid - mean) / std

        pos = state.position.get(product, 0)
        pos_limit = self.LIMIT[product]

        orders: List[Order] = []
        outskirt_order = False

        # ENTRY signals ---------------------------------------------------
        if z > self.Z_ENTRY and pos > -pos_limit:
            outskirt_order = True
            # over‑priced ⇒ hit best bid to go short
            vol = min(depth.buy_orders[best_bid], pos_limit + pos)
            if vol >= self.MIN_VOL:
                orders.append(Order(product, best_bid, -vol))

        elif z < -self.Z_ENTRY and pos < pos_limit:
            outskirt_order = True
            # under‑priced ⇒ lift best ask to go long
            vol = min(-depth.sell_orders[best_ask], pos_limit - pos)
            if vol >= self.MIN_VOL:
                orders.append(Order(product, best_ask, vol))

        if outskirt_order == False: 
            # add to history
            hist.append(mid)

        # EXIT signal -----------------------------------------------------
        if abs(z) < self.Z_EXIT and pos != 0:
            if pos > 0:
                vol = min(pos, depth.buy_orders[best_bid])
                orders.append(Order(product, best_bid, -vol))
            else:  # pos < 0
                vol = min(-pos, -depth.sell_orders[best_ask])
                orders.append(Order(product, best_ask, vol))

        return orders


    def run(self, state: TradingState):
        # ─── 1)  pull back whatever you stored last tick ────────────────────────────
        # traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}
        traderObject: Dict[str, Any] = (
            jsonpickle.decode(state.traderData) if state.traderData else {}
        )


        # ─── 2)  initialise any new fields *once*  (setdefault = no‑overwrite) ─────
        # traderObject.setdefault("ink_last_block_sign", 0)   #  +1 buy, ‑1 sell, 0 none
        # traderObject.setdefault("ink_block_expiry",   0)    #  timestamp of expiry
        # traderObject.setdefault("ink_last_mid",       None)
        # traderObject.setdefault("ink_delta1",         0)    #  last mid‑price change
        # traderObject.setdefault("ink_mom_sign",       0)    #  +1 up‑mom, ‑1 down, 0 flat

        result = {}

        if Product.RESIN in self.params and Product.RESIN in state.order_depths:
            order_depth = deepcopy(state.order_depths[Product.RESIN])
            resin_position = (
                state.position[Product.RESIN]
                if Product.RESIN in state.position
                else 0
            )
            resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RESIN,
                    order_depth,
                    self.params[Product.RESIN]["fair_value"],
                    self.params[Product.RESIN]["take_width"],
                    resin_position,
                )
                # self.take_orders_resin(
                #     Product.RESIN,
                #     order_depth,
                #     self.params[Product.RESIN]["fair_value"],
                #     self.params[Product.RESIN]["take_width"],
                #     resin_position,
                # )
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RESIN,
                    order_depth,
                    self.params[Product.RESIN]["fair_value"],
                    self.params[Product.RESIN]["clear_width"],
                    resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            resin_make_orders, _, _ = self.make_orders(
                Product.RESIN,
                order_depth,
                self.params[Product.RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RESIN]["disregard_edge"],
                self.params[Product.RESIN]["join_edge"],
                self.params[Product.RESIN]["default_edge"],
                True,
                self.params[Product.RESIN]["soft_position_limit"],
            )
            result[Product.RESIN] = (
                # resin_take_orders + resin_clear_orders
                resin_take_orders + resin_clear_orders + resin_make_orders
            )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_order_depth = deepcopy(state.order_depths[Product.KELP])

            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    kelp_order_depth,
                    kelp_fair_value,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    kelp_order_depth,
                    kelp_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            kelp_make_orders, _, _ = self.make_orders(
                Product.KELP,
                kelp_order_depth,
                kelp_fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                kelp_take_orders + kelp_clear_orders + kelp_make_orders
            )

        # if Product.SQUID_INK in state.order_depths:
        #     squid_orders = self._squid_orders(state, traderObject)
        #     result[Product.SQUID_INK] = squid_orders

        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        logger.flush(state, result, conversions, traderData)


        return result, conversions, traderData
