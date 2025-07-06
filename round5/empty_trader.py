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

}

class Trader:
    def __init__(self, PARAMS=None):
        self.PARAMS = PARAMS if PARAMS is not None else DEFAULT_PARAMS

    def hysteresis_inflections(self, series, high_thresh=0.001, low_thresh=0.0005):
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

    def run(self, state: TradingState) -> dict[str, list[Order]]:
        self.timestamp = state.timestamp
        trader_data = {}
        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)

        result = {}
        order_depth = state.order_depths[Product.MACARON]
        macaron_orders = []

        position = state.position.get(Product.MACARON, 0)
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        best_bid, best_bid_vol = sorted(buy_orders.items(), reverse=True)[0]
        best_ask, best_ask_vol = sorted(sell_orders.items(), reverse=False)[0]
        mid = (best_bid + best_ask) / 2

        # conversion observation
        observation = state.observations.conversionObservations.get(Product.MACARON)

        conversionBid = observation.bidPrice
        conversionAsk = observation.askPrice
        transportFees = observation.transportFees
        exportTariff = observation.exportTariff
        importTariff = observation.importTariff
        sugarPrice = observation.sugarPrice
        sunlightIndex = observation.sunlightIndex


        # fixed parameters
        threshold = 0.02
        required_sunlight_window_size = 25


        sunlight_history: list = trader_data.get('sunlight_history', [])
        sunlight_history.append(sunlightIndex)
        if len(sunlight_history) > required_sunlight_window_size:
            sunlight_history.pop(0)
        trader_data['sunlight_history'] = sunlight_history

        sunlight: pd.Series = pd.Series(sunlight_history)
        dot_sunlight = sunlight.diff().rolling(10).mean().rolling(10).mean()
        dot_dot_sunlight = dot_sunlight.diff(2)

        dot_inflections = self.hysteresis_inflections(dot_dot_sunlight)

        signal = None
        regime = None

        # the minus two indicates how long the signal for the regime change will run
        if dot_inflections and dot_inflections[-1][1] >= required_sunlight_window_size - 2:
            inflection_sign, index = dot_inflections[-1]
            last_dot_sunlight = dot_sunlight.iloc[-1]
            if last_dot_sunlight > 0.02:
                regime = 'RISING'
            elif last_dot_sunlight < -0.02:
                regime = 'FALLING'
            elif -0.005 <= last_dot_sunlight <= 0.005:
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


        # result[Product.MACARON] = macaron_orders

        traderData = jsonpickle.encode(trader_data)
        conversions = 0

        print(f'OWN TRADES {state.own_trades}')
        print(f'MARKET TRADES {state.market_trades}')

        return result, conversions, traderData