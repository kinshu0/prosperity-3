from datamodel import OrderDepth



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


'''

def chunk_orders(orders: list[tuple], chunk_size: int, ascending=True):
    orders = sorted(orders, reverse = not ascending)
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
    return chunk_depth

# chunk_depth = chunk_orders([(10, 42), (8, 40), (2, 28)], 6, ascending=False)

def construct_synth_buy_orders(order_depths: dict[str, OrderDepth], weights: dict[str, int]) -> dict[int, int]:

    synth_buy_orders = {}
    
    prod_chunked_buy = {}

    for prod, weight in weights.items():
        buy_orders = order_depths[prod].buy_orders.items()
        if len(buy_orders) == 0:
            return synth_buy_orders
        chunked_buys = chunk_orders(buy_orders, weight, ascending=False)
        chunked_buys = sorted(chunked_buys.items(), reverse=True)
        prod_chunked_buy[prod] = chunked_buys

    exhausted = False

    while not exhausted:
        quantity = 999999999
        for prod, chunked_buys in prod_chunked_buy.items():
            best_bid, vol = chunked_buys[0]
            quantity = min(quantity, vol)

        synth_bid = 0

        for prod, chunked_buys in prod_chunked_buy.items():
            best_bid, vol = chunked_buys[0]
            synth_bid += best_bid
            newvol = vol - quantity
            chunked_buys[0] = best_bid, newvol
            if newvol == 0:
                chunked_buys.pop(0)
            if len(chunked_buys) == 0:
                exhausted = True
            prod_chunked_buy[prod] = chunked_buys
            
        synth_buy_orders[synth_bid] = quantity

    return synth_buy_orders

def construct_synth_sell_orders(order_depths: dict[str, OrderDepth], weights: dict[str, int]) -> dict[int, int]:

    synth_sell_orders = {}

    prod_chunked_sell = {}

    for prod, weight in weights.items():
        sell_orders = order_depths[prod].sell_orders.items()
        if len(sell_orders) == 0:
            return synth_sell_orders
        sell_orders = [(price, -quantity) for price, quantity in sell_orders]
        chunked_sells = chunk_orders(sell_orders, weight, ascending=True)
        chunked_sells = sorted(chunked_sells.items())
        prod_chunked_sell[prod] = chunked_sells

    exhausted = False

    while not exhausted:
        quantity = 999999999
        for prod, chunked_sells in prod_chunked_sell.items():
            best_ask, vol = chunked_sells[0]
            quantity = min(quantity, vol)

        synth_ask = 0

        for prod, chunked_sells in prod_chunked_sell.items():
            best_ask, vol = chunked_sells[0]
            synth_ask += best_ask
            newvol = vol - quantity
            chunked_sells[0] = best_ask, newvol
            if newvol == 0:
                chunked_sells.pop(0)
            if len(chunked_sells) == 0:
                exhausted = True
            prod_chunked_sell[prod] = chunked_sells
            
        synth_sell_orders[synth_ask] = -quantity


def construct_synth_od(order_depths: dict[str, OrderDepth], weights: dict[str, int]) -> OrderDepth:
        
    order_depth = OrderDepth()
    order_depth.buy_orders = construct_synth_buy_orders(order_depths, weights)
    order_depth.sell_orders = construct_synth_sell_orders(order_depths, weights)

    return order_depth


class Product:
    BASKET1 = "PICNIC_BASKET1"
    JAMS = "JAMS"
    CROISSANTS = "CROISSANTS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"

BASKET1_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}

order_depths = {
    Product.CROISSANTS: OrderDepth(),
    Product.JAMS: OrderDepth(),
    Product.DJEMBES: OrderDepth()
}


order_depths[Product.CROISSANTS].buy_orders = {k: v for k, v in [(42, 10), (40, 8), (28, 2)]}
order_depths[Product.JAMS].buy_orders = {k: v for k, v in [(36, 7), (35, 8), (34, 2)]}
order_depths[Product.DJEMBES].buy_orders = {k: v for k, v in [(27, 9), (26, 8), (25, 9)]}

od = construct_synth_od(order_depths, BASKET1_WEIGHTS)

print(f'{od.buy_orders}')
print(f'{od.sell_orders}')