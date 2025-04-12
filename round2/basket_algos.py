
'''
weights = 6c 3j 1d

c [(42, 10), (40, 8), (28, 2)]
j [(36, 7), (35, 8), (34, 2)]
d [(27, 9), (26, 8), (25, 9)]

7
12
27

can sell 7 at best bid

7 - 7 = 0
12 - 7 = 5
27 - 7 = 20

pull back next price level

c [(42, 10), (40, 8), (28, 2)]
j [(15, 7), (35, 8), (34, 2)]
d [(27, 9), (26, 8), (25, 9)]

[(x, 2), (x, 2), (x, 1), (x, 3)]

dealing with negative wegihts:
if negative weights, we basically do the same thing except run the sell side for it instead?


'''

# change chunk_orders to work with negative quantities
'''
if ascending we're sorting market sell orders
so if ascending then we know quantity is likely negative
'''

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

'''

Say we have a synthetic with compositin

spread_synth_weights = {
    basket1: 1
    croissants: -6,
    jams: -3,
    djembes: -1,
}


we want to create buy and sell sides of order depth

for order depth market buy_orders:
- we'll use basket1 market buy orders
- rest we'll use product sell orders

for order depth market sell_orders:
flip the sign and that's what we'll be using same as above but simply with sign flipped, will be able to reuse same function
- we'll use basket1 market sell orders
- rest we'll use product buy orders

'''

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
    order_depth.sell_orders = construct_synth_buy_orders(order_depths, neg_weights)

    return order_depth

class Product:
    BASKET1 = "PICNIC_BASKET1"
    JAMS = "JAMS"
    CROISSANTS = "CROISSANTS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"

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

order_depths = {
    Product.BASKET1: OrderDepth(),
    Product.CROISSANTS: OrderDepth(),
    Product.JAMS: OrderDepth(),
    Product.DJEMBES: OrderDepth(),
}

'''
{397: 2, 350: 3, 380: 5}
{397: -2, 350: -3, 380: -5}


od.buy_orders for synth basket1
{387: 1, 383: 1, 373: 1}
'''

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

# od = construct_synth_buy_orders(order_depths, BASKET1_WEIGHTS)
# od = construct_synth_buy_orders(order_depths, spread_weights)
od = construct_synth_od(order_depths, spread_weights)

# print(f'{od.buy_orders}')
# print(f'{od.sell_orders}')

print(od)