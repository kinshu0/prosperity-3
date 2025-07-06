from datamodel import OrderDepth, Order, TradingState, Symbol, Listing, Trade, Observation, ProsperityEncoder
from typing import List, Dict, Any
import jsonpickle
from math import log, sqrt
from statistics import NormalDist
import json


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
    ROCK = 'VOLCANIC_ROCK'
    VOUCHER_9500 = 'VOLCANIC_ROCK_VOUCHER_9500'
    VOUCHER_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
    VOUCHER_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
    VOUCHER_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
    VOUCHER_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'

# Black-Scholes model implementation
class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def implied_volatility(call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10):
        low_vol, high_vol = 0.01, 1.0
        volatility = (low_vol + high_vol) / 2.0
        
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
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
    def __init__(self):
        self.LIMIT = {
            Product.ROCK: 400,
            Product.VOUCHER_9500: 200,
            Product.VOUCHER_9750: 200,
            Product.VOUCHER_10000: 200,
            Product.VOUCHER_10250: 200,
            Product.VOUCHER_10500: 200,
        }
        self.PARAMS = {
            "mean_volatility": 0.138,
            "strike": 10000,
            "zscore_threshold": 3,
            "std_window": 30
        }

    def get_mid_price(self, order_depth):
        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return None

    def run(self, state):
        # Initialize trader data
        trader_data = {}
        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)

        # Initialize product data if needed
        if Product.VOUCHER_10000 not in trader_data:
            trader_data[Product.VOUCHER_10000] = {
                "prev_price": 0,
                "past_vol": []
            }

        result = {}
        
            
            # rock_pos_after = rock_position + sum(order.quantity for order in rock_orders)
            # voucher_pos_after = voucher_position + sum(order.quantity for order in voucher_orders)
            
            # to_hedge = voucher_pos_after * delta
            # target_rock_pos = round(to_hedge)
            # rock_quantity = target_rock_pos - rock_pos_after
            # if rock_quantity > 0:
            #     best_market_ask, best_volume = min(rock_order_depth.sell_orders.items(), default=(float('inf'), 0))
            #     best_volume = -best_volume
            #     if best_volume > 0:
            #         rock_quantity = min(rock_quantity, best_volume)
            #         rock_orders.append(Order(Product.ROCK, best_market_ask, rock_quantity))
            # elif rock_quantity < 0:
            #     best_market_bid, best_volume = max(rock_order_depth.buy_orders.items(), default=(0, 0))
            #     if best_volume > 0:
            #         rock_quantity = min(-rock_quantity, best_volume)
            #         rock_orders.append(Order(Product.ROCK, best_market_bid, -rock_quantity))

            # result[Product.ROCK] = rock_orders

        # for PRODUCT in [Product.VOUCHER_9500, Product.VOUCHER_9750, Product.VOUCHER_10000, Product.VOUCHER_10250, Product.VOUCHER_10500]:

        

        for PRODUCT in [Product.VOUCHER_10000, Product.VOUCHER_10250, Product.VOUCHER_10500]:
        # for PRODUCT in [Product.VOUCHER_9750, Product.VOUCHER_10000, Product.VOUCHER_10250, Product.VOUCHER_10500]:
            voucher_position = state.position.get(PRODUCT, 0)
            rock_position = state.position.get(Product.ROCK, 0)
            
            # Get order depths
            rock_order_depth = state.order_depths[Product.ROCK]
            voucher_order_depth = state.order_depths[PRODUCT]
            
            # Calculate mid prices
            rock_mid = self.get_mid_price(rock_order_depth)
            voucher_mid = self.get_mid_price(voucher_order_depth)

            if rock_mid is None or voucher_mid is None:
                return result, 1, {}

            # iv = 0.005847 * m_t^2 + -0.001162 * m_t + 0.124922
            # m_t = np.log(strike / mk['mid_price_rock']) / mk['tte']

            day = 2
            tte = (8 - day - state.timestamp / 1_000_000) / 250
            strike = {Product.VOUCHER_9500: 9500, Product.VOUCHER_9750: 9750, Product.VOUCHER_10000: 10000, Product.VOUCHER_10250: 10250, Product.VOUCHER_10500: 10500}[PRODUCT]

            m_t = log(strike / rock_mid) / tte
            # Calculate implied volatility using a polynomial approximation
            fair_iv = 0.005847 * m_t**2 - 0.001162 * m_t + 0.124922
            
            # Calculate implied volatility and delta
            obs_iv = BlackScholes.implied_volatility(
                voucher_mid,
                rock_mid,
                strike,
                tte
            )
            
            delta = BlackScholes.delta(
                rock_mid,
                strike,
                tte,
                fair_iv
            )

            '''
            Strike 10000 - Mean Difference: 0.000308, Std Dev: 0.005438, Min: -0.022797, Max: 0.024732
            Strike 10250 - Mean Difference: 0.000088, Std Dev: 0.001925, Min: -0.006760, Max: 0.005741
            Strike 10500 - Mean Difference: -0.000395, Std Dev: 0.002334, Min: -0.008265, Max: 0.012089
            '''

            mean_diff = {
                # Product.VOUCHER_9750: 0.000308,
                Product.VOUCHER_10000: 0.000308,
                Product.VOUCHER_10250: 0.000088,
                Product.VOUCHER_10500: -0.000395
            }[PRODUCT]
            std_dev = {
                Product.VOUCHER_10000: 0.005438,
                Product.VOUCHER_10250: 0.001925,
                Product.VOUCHER_10500: 0.002334
            }[PRODUCT]

            z_thresh = {
                # Product.VOUCHER_10000: 1,
                # Product.VOUCHER_10250: 2.5,
                # Product.VOUCHER_10500: 0.1
                # Product.VOUCHER_9750: 1,
                Product.VOUCHER_10000: 1,
                Product.VOUCHER_10250: 1.5,
                Product.VOUCHER_10500: 2.5
                # Product.VOUCHER_10000: 1,
                # Product.VOUCHER_10250: 1,
                # Product.VOUCHER_10500: 1
            }[PRODUCT]
            

            diff = obs_iv - fair_iv
            z = (diff - mean_diff) / std_dev
            

            voucher_orders = []
            rock_orders = []


            # if PRODUCT == Product.VOUCHER_10500:
            #     print(fair_iv)


            if z >= z_thresh:
                # sell voucher
                target_voucher_pos = -self.LIMIT[PRODUCT]
                sell_quantity = voucher_position - target_voucher_pos
                best_market_bid, best_volume = max(voucher_order_depth.buy_orders.items(), default=(0, 0))
                if sell_quantity > 0 and best_volume > 0:
                    sell_quantity = min(sell_quantity, best_volume)
                    voucher_orders.append(Order(PRODUCT, best_market_bid, -sell_quantity))
                    # print(f"Sell {PRODUCT}: {sell_quantity} at {best_market_bid}")

            elif z <= -z_thresh:
                # buy voucher
                target_voucher_pos = self.LIMIT[PRODUCT]
                buy_quantity = target_voucher_pos - voucher_position
                best_market_ask, best_volume = min(voucher_order_depth.sell_orders.items(), default=(0, 0))
                best_volume = -best_volume  # Convert to positive volume
                if buy_quantity > 0 and best_volume > 0:
                    buy_quantity = min(buy_quantity, best_volume)
                    # print(f"Buy {PRODUCT}: {buy_quantity} at {best_market_ask}")
                    voucher_orders.append(Order(PRODUCT, best_market_ask, buy_quantity))

            elif z <=0.5 and z >= -0.5:
                if voucher_position > 0:
                    # If we have a position, we might want to sell it to reduce risk
                    best_market_bid, best_volume = max(voucher_order_depth.buy_orders.items(), default=(0, 0))
                    if best_volume > 0:
                        voucher_orders.append(Order(PRODUCT, best_market_bid, -voucher_position))
                elif voucher_position < 0:
                    # If we have a short position, we might want to buy it back
                    best_market_ask, best_volume = min(voucher_order_depth.sell_orders.items(), default=(99999, 0))
                    if best_volume > 0:
                        voucher_orders.append(Order(PRODUCT, best_market_ask, -voucher_position))
        
            
            # rock_pos_after = rock_position + sum(order.quantity for order in rock_orders)
            # voucher_pos_after = voucher_position + sum(order.quantity for order in voucher_orders)
            
            # to_hedge = voucher_pos_after * delta
            # target_rock_pos = round(to_hedge)
            # rock_quantity = target_rock_pos - rock_pos_after
            # if rock_quantity > 0:
            #     best_market_ask, best_volume = min(rock_order_depth.sell_orders.items(), default=(float('inf'), 0))
            #     best_volume = -best_volume
            #     if best_volume > 0:
            #         rock_quantity = min(rock_quantity, best_volume)
            #         rock_orders.append(Order(Product.ROCK, best_market_ask, rock_quantity))
            # elif rock_quantity < 0:
            #     best_market_bid, best_volume = max(rock_order_depth.buy_orders.items(), default=(0, 0))
            #     if best_volume > 0:
            #         rock_quantity = min(-rock_quantity, best_volume)
            #         rock_orders.append(Order(Product.ROCK, best_market_bid, -rock_quantity))

            # result[Product.ROCK] = rock_orders



            result[PRODUCT] = voucher_orders

            
        logger.flush(
            state,
            result,
            1,
            ''
        )


        return result, 1, jsonpickle.encode(trader_data)