from datamodel import Order

class OrderDepth:
    def __init__(self):
        self.buy_orders = {}  # Dictionary with price -> quantity
        self.sell_orders = {}  # Dictionary with price -> quantity (negative quantities)
    
    def __str__(self):
        return f"Buy orders: {self.buy_orders}\nSell orders: {self.sell_orders}"

class OrderType:
    BUY = 'BUY'
    SELL = 'SELL'

def chunk_orders(orders: list[tuple], chunk_size: int, order_type=OrderType.BUY) -> dict[int, int]:
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

def construct_synth_buy_orders(order_depths: dict[str, OrderDepth], weights: dict[str, int]) -> dict[int, int]:

    synth_buy_orders = {}
    
    prod_chunked_buy = {}

    for prod, weight in weights.items():
        if prod not in order_depths:
            return {}

        # if weight is positve, market buy orders
        if weight > 0:
            buy_orders = order_depths[prod].buy_orders.items()
            chunked_buys = chunk_orders(buy_orders, weight)
            chunked_buys = sorted(chunked_buys.items(), reverse=True)
        # if weight is negative, market sell orders
        elif weight < 0:
            buy_orders = order_depths[prod].sell_orders.items()
            chunked_buys = chunk_orders(buy_orders, abs(weight), order_type=OrderType.SELL)
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

def construct_synth_od(order_depths: dict[str, OrderDepth], weights: dict[str, int]) -> OrderDepth:
        
    order_depth = OrderDepth()
    order_depth.buy_orders = construct_synth_buy_orders(order_depths, weights)

    neg_weights = {prod: -weights[prod] for prod in weights}
    sell_orders = construct_synth_buy_orders(order_depths, neg_weights)
    sell_orders_pos = {}
    for price, q in sell_orders.items():
        sell_orders_pos[-price] = -q

    order_depth.sell_orders = sell_orders_pos

    return order_depth

def buy_synth(limit_price: int, position_limits: dict[str, int], positions: dict[str, int], synth_od: OrderDepth, order_depths: dict[str, OrderDepth], weights: dict[str, int]):
    
    can_buy = 999999

    for prod, weight in weights.items():
        # if weight is greater than 0 we want to look at market sells to match
        # don't want to exceed upper position limit
        if weight > 0:
            chunks = position_limits[prod] - positions[prod]
        # if weight is less than 0 we want to look at market buys and not exceed lower position limit
        else:
            chunks = positions[prod] - -position_limits[prod]
        can_buy = min(chunks // abs(weight), can_buy)
    

    # get synth volume to buy
    # we match with synth market sells to buy synth
    # only part different between buy and sell?
    synth_market_sells = synth_od.sell_orders
    acc_synth_market_sells = [(p, q) for p, q in synth_market_sells.items() if p <= limit_price]

    acc_market_sell_vol = sum(x[1] for x in acc_synth_market_sells) if acc_synth_market_sells else 0

    to_buy = min(-acc_market_sell_vol, can_buy)

    if to_buy == 0:
        return []
    
    orders = []

    for prod, weight in weights.items():
        # here q would automatically be negative if it's negative weight
        q = weight * to_buy
        match_price = 0 if weight < 0 else 9999999
        orders.append(Order(prod, match_price, q))

    return orders

def sell_synth(limit_price: int, position_limits: dict[str, int], positions: dict[str, int], synth_od: OrderDepth, order_depths: dict[str, OrderDepth], weights: dict[str, int]):

    weights = {prod: -weights[prod] for prod in weights}
    
    can_sell = 999999

    for prod, weight in weights.items():
        if weight > 0:
            chunks = position_limits[prod] - positions[prod]
        else:
            chunks = positions[prod] - -position_limits[prod]
        can_sell = min(chunks // abs(weight), can_sell)

    synth_market_buys = synth_od.buy_orders
    acc_synth_market_buys = [(p, q) for p, q in synth_market_buys.items() if p >= limit_price]

    acc_market_buy_vol = sum(x[1] for x in acc_synth_market_buys) if acc_synth_market_buys else 0

    to_buy = min(acc_market_buy_vol, can_sell)

    if to_buy == 0:
        return []
    
    orders = []

    for prod, weight in weights.items():
        q = weight * to_buy
        match_price = 0 if weight < 0 else 9999999
        orders.append(Order(prod, match_price, q))

    return orders

class Product:
    BASKET1 = "PICNIC_BASKET1"
    JAMS = "JAMS"
    CROISSANTS = "CROISSANTS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"


order_depths = {
    Product.BASKET1: OrderDepth(),
    Product.CROISSANTS: OrderDepth(),
    Product.JAMS: OrderDepth(),
    Product.DJEMBES: OrderDepth(),
}

# BASKET1 orders - realistic market with bids below asks and small price increments
order_depths[Product.BASKET1].buy_orders = {395: 2, 394: 5, 393: 3}
order_depths[Product.BASKET1].sell_orders = {397: -1, 398: -4, 399: -2}

# CROISSANTS orders 
order_depths[Product.CROISSANTS].buy_orders = {39: 5, 38: 7, 37: 3}
order_depths[Product.CROISSANTS].sell_orders = {41: -10, 42: -8, 43: -2}

# JAMS orders
order_depths[Product.JAMS].buy_orders = {32: 5, 31: 6, 30: 4}
order_depths[Product.JAMS].sell_orders = {33: -7, 34: -8, 35: -2}

# DJEMBES orders
order_depths[Product.DJEMBES].buy_orders = {24: 6, 23: 8, 22: 5}
order_depths[Product.DJEMBES].sell_orders = {25: -9, 26: -8, 27: -9}

synth_basket_weights = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}

spread_weights = {
    Product.BASKET1: 1,
    Product.CROISSANTS: -6,
    Product.JAMS: -3,
    Product.DJEMBES: -1,
}

positions = {
    Product.BASKET1: 0,
    Product.CROISSANTS: 0,
    Product.JAMS: 0,
    Product.DJEMBES: 0,
}

position_limits = {
    Product.CROISSANTS: 250,
    Product.JAMS: 350,
    Product.DJEMBES: 60,
    Product.BASKET1: 60,
}

# od = construct_synth_buy_orders(order_depths, BASKET1_WEIGHTS)
# od = construct_synth_buy_orders(order_depths, spread_weights)
# od, buy_pieces, sell_pieces = construct_synth_buy(order_depths, spread_weights)
# agg_orders, pieces_orders = construct_synth_buy(order_depths, spread_weights)

od = construct_synth_od(order_depths, spread_weights)
# sell_orders = sell_synth(23, position_limit, position_limit, od, order_depths, spread_weights)
orders = sell_synth(9, position_limits, positions, od, order_depths, spread_weights)

print(od)
print(orders)