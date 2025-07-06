from datamodel import OrderDepth, Order
from typing import List, Dict, Any
import jsonpickle
from math import log, sqrt
from statistics import NormalDist

class Product:
    ROCK = 'VOLCANIC_ROCK'
    VOUCHER_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'

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
            Product.VOUCHER_10000: 200,
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

    def voucher_orders(self, voucher_order_depth, voucher_position, trader_data, volatility):
        # Store volatility
        trader_data['past_vol'].append(volatility)
        if len(trader_data['past_vol']) < self.PARAMS["std_window"]:
            return None, None

        if len(trader_data['past_vol']) > self.PARAMS["std_window"]:
            trader_data['past_vol'].pop(0)
        
        # Fixed volatility standard deviation
        vol_std = 0.007362
        
        # Calculate z-score
        vol_z_score = (volatility - self.PARAMS["mean_volatility"]) / vol_std
        
        # Handle case where volatility is high (short options)
        if vol_z_score >= self.PARAMS["zscore_threshold"]:
            if voucher_position != -self.LIMIT[Product.VOUCHER_10000]:
                target_position = -self.LIMIT[Product.VOUCHER_10000]
                if len(voucher_order_depth.buy_orders) > 0:
                    best_bid = max(voucher_order_depth.buy_orders.keys())
                    target_quantity = abs(target_position - voucher_position)
                    quantity = min(
                        target_quantity,
                        abs(voucher_order_depth.buy_orders[best_bid]),
                    )
                    quote_quantity = target_quantity - quantity
                    
                    if quote_quantity == 0:
                        return [Order(Product.VOUCHER_10000, best_bid, -quantity)], []
                    else:
                        return [Order(Product.VOUCHER_10000, best_bid, -quantity)], [Order(Product.VOUCHER_10000, best_bid, -quote_quantity)]

        # Handle case where volatility is low (long options)
        elif vol_z_score <= -self.PARAMS["zscore_threshold"]:
            if voucher_position != self.LIMIT[Product.VOUCHER_10000]:
                target_position = self.LIMIT[Product.VOUCHER_10000]
                if len(voucher_order_depth.sell_orders) > 0:
                    best_ask = min(voucher_order_depth.sell_orders.keys())
                    target_quantity = abs(target_position - voucher_position)
                    quantity = min(
                        target_quantity,
                        abs(voucher_order_depth.sell_orders[best_ask]),
                    )
                    quote_quantity = target_quantity - quantity
                    
                    if quote_quantity == 0:
                        return [Order(Product.VOUCHER_10000, best_ask, quantity)], []
                    else:
                        return [Order(Product.VOUCHER_10000, best_ask, quantity)], [Order(Product.VOUCHER_10000, best_ask, quote_quantity)]

        return None, None

    def hedge_orders(self, rock_order_depth, voucher_orders, rock_position, voucher_position, delta):
        # Calculate position after trades
        if voucher_orders is None or len(voucher_orders) == 0:
            voucher_position_after_trade = voucher_position
        else:
            voucher_position_after_trade = voucher_position + sum(order.quantity for order in voucher_orders)
        
        # Calculate target position based on delta
        target_rock_position = -delta * voucher_position_after_trade
        
        # No action needed if already at target
        if target_rock_position == rock_position:
            return None
        
        # Calculate how much to buy/sell
        target_rock_quantity = target_rock_position - rock_position

        orders = []
        if target_rock_quantity > 0:
            # Buy ROCK
            if len(rock_order_depth.sell_orders) > 0:
                best_ask = min(rock_order_depth.sell_orders.keys())
                quantity = min(
                    abs(target_rock_quantity),
                    self.LIMIT[Product.ROCK] - rock_position,
                )
                if quantity > 0:
                    orders.append(Order(Product.ROCK, best_ask, round(quantity)))
        
        elif target_rock_quantity < 0:
            # Sell ROCK
            if len(rock_order_depth.buy_orders) > 0:
                best_bid = max(rock_order_depth.buy_orders.keys())
                quantity = min(
                    abs(target_rock_quantity),
                    self.LIMIT[Product.ROCK] + rock_position,
                )
                if quantity > 0:
                    orders.append(Order(Product.ROCK, best_bid, -round(quantity)))
        
        return orders

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
        
        # Check if we can trade options
        if Product.VOUCHER_10000 in state.order_depths and Product.ROCK in state.order_depths:
            # Get positions
            voucher_position = state.position.get(Product.VOUCHER_10000, 0)
            rock_position = state.position.get(Product.ROCK, 0)
            
            # Get order depths
            rock_order_depth = state.order_depths[Product.ROCK]
            voucher_order_depth = state.order_depths[Product.VOUCHER_10000]
            
            # Calculate mid prices
            rock_mid = self.get_mid_price(rock_order_depth)
            voucher_mid = self.get_mid_price(voucher_order_depth)
            
            if not rock_mid or not voucher_mid:
                return result, jsonpickle.encode(trader_data)
            
            # Calculate time to expiry
            tte = (7 - state.timestamp / 1_000_000) / (1 / 250 / 10000)
            
            # Calculate implied volatility and delta
            volatility = BlackScholes.implied_volatility(
                voucher_mid,
                rock_mid,
                self.PARAMS["strike"],
                tte
            )
            
            delta = BlackScholes.delta(
                rock_mid,
                self.PARAMS["strike"],
                tte,
                volatility
            )
            
            # Generate orders
            voucher_take_orders, voucher_make_orders = self.voucher_orders(
                voucher_order_depth,
                voucher_position,
                trader_data[Product.VOUCHER_10000],
                volatility
            )
            
            # Create option orders
            if voucher_take_orders or voucher_make_orders:
                orders = []
                if voucher_take_orders:
                    orders.extend(voucher_take_orders)
                if voucher_make_orders:
                    orders.extend(voucher_make_orders)
                result[Product.VOUCHER_10000] = orders
            
            # Create hedge orders
            rock_orders = self.hedge_orders(
                rock_order_depth,
                voucher_take_orders,
                rock_position,
                voucher_position,
                delta
            )
            
            if rock_orders:
                result[Product.ROCK] = rock_orders

        return result, jsonpickle.encode(trader_data)