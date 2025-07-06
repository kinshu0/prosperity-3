from datamodel import OrderDepth, UserId, TradingState, Order
import string
import jsonpickle
from copy import deepcopy
import numpy as np
import pandas as pd

class Product:
    MACARON = 'MAGNIFICENT_MACARONS'

POSITION_LIMIT = {
    Product.MACARON: 75
}

DEFAULT_PARAMS = {
    'macaron_hysteresis_high_thresh': 0.001,
    'macaron_hysteresis_low_thresh': 0.0005,
    'macaron_sunlight_window_size': 25,
    'macaron_sunlight_rising_threshold': 0.02,
    'macaron_sunlight_falling_threshold': -0.02,
    'macaron_sunlight_neutral_upper': 0.005,
    'macaron_sunlight_neutral_lower': -0.005,
    'macaron_sunlight_rolling_window': 10,
    'macaron_sunlight_diff_period': 2
}

class Trader:
    def __init__(self, PARAMS=None):
        self.PARAMS = PARAMS if PARAMS is not None else DEFAULT_PARAMS
        
    def hysteresis_inflections(self, series, high_thresh=None, low_thresh=None):
        # Use parameters from DEFAULT_PARAMS if not provided
        high_thresh = high_thresh if high_thresh is not None else self.PARAMS['macaron']['hysteresis']['high_thresh']
        low_thresh = low_thresh if low_thresh is not None else self.PARAMS['macaron']['hysteresis']['low_thresh']
        
        dot_dot = series.dropna()
        idxs = dot_dot.index
        values = dot_dot.values

        inflections = []
        can_signal_pos = True
        can_signal_neg = True

        for i in range(len(values)):
            val = values[i]
            idx = idxs[i]

            if can_signal_pos and val > high_thresh:
                inflections.append(('positive', idx))
                can_signal_pos = False  # turn off until we fall below low_thresh
            elif val < low_thresh:
                can_signal_pos = True

            if can_signal_neg and val < -high_thresh:
                inflections.append(('negative', idx))
                can_signal_neg = False
            elif val > -low_thresh:
                can_signal_neg = True

        return inflections

    def macaron(self, order_depth, position, trader_data, observation):
        # Get parameters
        params = self.PARAMS['macaron']
        sunlight_params = params['sunlight']
        
        # order_depth = state.order_depths[Product.MACARON]
        macaron_orders = []

        # position = state.position.get(Product.MACARON, 0)
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        best_bid, best_bid_vol = sorted(buy_orders.items(), reverse=True)[0]
        best_ask, best_ask_vol = sorted(sell_orders.items(), reverse=False)[0]
        mid = (best_bid + best_ask) / 2

        conversion_bid = observation.bidPrice
        conversion_ask = observation.askPrice
        transport_fees = observation.transportFees
        export_tariff = observation.exportTariff
        import_tariff = observation.importTariff
        sugar_price = observation.sugarPrice
        sunlightIndex = observation.sunlightIndex

        required_sunlight_window_size = sunlight_params['window_size']

        sunlight_history: list = trader_data.get('sunlight_history', [])
        sunlight_history.append(sunlightIndex)
        if len(sunlight_history) > required_sunlight_window_size:
            sunlight_history.pop(0)
        trader_data['sunlight_history'] = sunlight_history

        rolling_window = sunlight_params['rolling_window']
        diff_period = sunlight_params['diff_period']

        sunlight: pd.Series = pd.Series(sunlight_history)
        dot_sunlight = sunlight.diff().rolling(rolling_window).mean().rolling(rolling_window).mean()
        dot_dot_sunlight = dot_sunlight.diff(diff_period)

        dot_inflections = self.hysteresis_inflections(dot_dot_sunlight)

        signal = None
        regime = None

        # the minus two indicates how long the signal for the regime change will run
        if dot_inflections and dot_inflections[-1][1] >= required_sunlight_window_size - 2:
            inflection_sign, index = dot_inflections[-1]
            last_dot_sunlight = dot_sunlight.iloc[-1]
            
            if last_dot_sunlight > sunlight_params['rising_threshold']:
                regime = 'RISING'
            elif last_dot_sunlight < sunlight_params['falling_threshold']:
                regime = 'FALLING'
            elif sunlight_params['neutral_lower'] <= last_dot_sunlight <= sunlight_params['neutral_upper']:
                regime = 'NEUTRAL'
        
        # regime to signal
        if regime == 'RISING':
            signal = 'SELL'
        elif regime == 'FALLING':
            signal = 'BUY'
        elif regime == 'NEUTRAL':
            signal = 'CLEAR'

        # signal to execution
        if signal == 'BUY':
            to_buy = min(-best_ask_vol, POSITION_LIMIT[Product.MACARON] - position)
            if to_buy > 0:
                macaron_orders.append(Order(Product.MACARON, best_ask, to_buy))

        elif signal == 'SELL':
            to_sell = min(best_bid_vol, position - -POSITION_LIMIT[Product.MACARON])
            if to_sell > 0:
                macaron_orders.append(Order(Product.MACARON, best_bid, -to_sell))

        elif signal == 'CLEAR':
            if position > 0:
                macaron_orders.append(Order(Product.MACARON, best_bid, -position))
            elif position < 0:
                macaron_orders.append(Order(Product.MACARON, best_ask, -position))
        
        return macaron_orders

    def run(self, state: TradingState) -> dict[str, list[Order]]:
        self.timestamp = state.timestamp
        trader_data = {}
        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)

        result = {}

        if Product.MACARON in state.order_depths:
            order_depth = state.order_depths[Product.MACARON]
            macaron_orders = self.macaron(order_depth, state.position.get(Product.MACARON, 0), trader_data, state.observations.conversionObservations.get(Product.MACARON))

            result[Product.MACARON] = macaron_orders

        traderData = jsonpickle.encode(trader_data)
        conversions = 0

        return result, conversions, traderData