
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

'''
Order depth of spread:

Buy orders: {25: 1, 23: 1, 16: 1}
Sell orders: {-44: 1, -51: 1}

Let's say we want to sell spread for 23, how do we create this synthetic order?

We want to take all the orders upto our position limit, how do we do that as well?

These are given position limits
- `CROISSANT`: 250
- `JAM`: 350
- `DJEMBE`: 60
- `PICNIC_BASKET1`: 60
- `PICNIC_BASKET2`: 100

Our current position also matters.

First we subtract individual position limits based on current position, for simplicity let's say we're at 0 for all

Then for each product, max we can sell are the position limits.

Find max products we can sell as well, which is:

croissant: 250 // 6
jam: 350 // 3
...

find the min of above acording to the position limit

We can sell 1 spread at 25, 1 at 23 based on the specified limit price and order depth

Issue comes up with orders fragmented at different price levels. Is this an issue since price-time priority should guarantee execution with proper prices.

So we just pick arbitrarily low price to sell at and it gets matched against specified price regardless as long as we send proper volume in.

Let's do that.

spread_weights = {
    Product.BASKET1: 1,
    Product.CROISSANTS: -6,
    Product.JAMS: -3,
    Product.DJEMBES: -1,
}

'''


'''
{397: 2, 350: 3, 380: 5}
{397: -2, 350: -3, 380: -5}


od.buy_orders for synth basket1
{387: 1, 383: 1, 373: 1}


Buy orders: {25: 1, 23: 1, 16: 1}
Sell orders: {-44: 1, -51: 1}
'''

# def chunk_orders(orders: list[tuple], chunk_size: int, order_type=OrderType.BUY) -> dict[int, int]:
#     if order_type == OrderType.BUY:
#         orders = sorted(orders, reverse=True)
#     else:
#         orders = sorted([(p, -q) for p, q in orders])

#     chunk_depth = {}
#     chunk_depth_orders = {}
#     i = 0
#     while i < len(orders):
#         price, quantity = orders[i]
#         num_chunks = quantity // chunk_size
#         leftover_pieces = quantity % chunk_size
#         chunk_price = price * chunk_size
#         if num_chunks > 0:
#             chunk_depth[chunk_price] = chunk_depth.get(chunk_price, 0) + num_chunks
#             chunk_depth_orders[chunk_price] = [(price, num_chunks * chunk_size),]
#         if leftover_pieces > 0:
#             order_pieces = [(price, leftover_pieces)]
#             hybrid_price = leftover_pieces * price
#             i += 1
#             while i < len(orders):
#                 next_price, next_quantity = orders[i]
#                 to_fill = chunk_size - leftover_pieces
#                 filled = min(next_quantity, to_fill)
#                 order_pieces.append((next_price, filled))
#                 leftover_pieces += filled
#                 hybrid_price += filled * next_price
#                 next_quantity -= filled
#                 orders[i] = next_price, next_quantity
#                 if leftover_pieces == chunk_size:
#                     break
#                 i += 1
#             if leftover_pieces == chunk_size:
#                 chunk_depth[hybrid_price] = chunk_depth.get(hybrid_price, 0) + 1
#                 chunk_depth_orders[hybrid_price] = order_pieces
#         else:
#             i += 1
    
#     if order_type == OrderType.SELL:
#         for price in chunk_depth:
#             q = chunk_depth[price]
#             chunk_depth[price] = -q

#             chunk_depth_orders[price] = [(p, -q) for p, q in chunk_depth_orders[price]]

#     return chunk_depth, chunk_depth_orders


# # chunked_orders = chunk_orders([(p, q) for p, q in {33: -7, 34: -8, 35: -2}.items()], 3, OrderType.SELL)

# # print(chunked_orders)

# def construct_synth_buy(order_depths: dict[str, OrderDepth], weights: dict[str, int]):

#     synth_agg_orders = {}
#     synth_prod_orders = {}
    
#     prod_chunked_buys = {}
#     prod_chunk_pieces = {}

#     for prod, weight in weights.items():
#         if prod not in order_depths:
#             return {}

#         # if weight is positve, market buy orders
#         if weight > 0:
#             buy_orders = order_depths[prod].buy_orders.items()
#             chunked_buys, chunk_pieces = chunk_orders(buy_orders, weight)
#             chunked_buys = sorted(chunked_buys.items(), reverse=True)
#         # if weight is negative, market sell orders
#         elif weight < 0:
#             buy_orders = order_depths[prod].sell_orders.items()
#             chunked_buys, chunk_pieces = chunk_orders(buy_orders, abs(weight), order_type=OrderType.SELL)
#             chunked_buys = sorted(chunked_buys.items())

#         if len(buy_orders) == 0:
#             return {}

#         prod_chunked_buys[prod] = chunked_buys
#         prod_chunk_pieces[prod] = chunk_pieces

#     exhausted = False

#     while not exhausted:
#         quantity = 999999999
#         for prod, chunked_buys in prod_chunked_buys.items():
#             best_price, vol = chunked_buys[0]
#             quantity = min(quantity, abs(vol))

#         synth_bid = 0

#         price_level_pieces = {}

#         for prod, chunked_buys in prod_chunked_buys.items():
#             best_price, vol = chunked_buys[0]
#             # dealing with negative weighted products
#             if vol < 0:
#                 synth_bid -= best_price
#                 newvol = vol + quantity
#             else:
#                 synth_bid += best_price
#                 newvol = vol - quantity

#             weight = weights[prod]
#             chunk_pieces = prod_chunk_pieces[prod][best_price]
#             price_level_pieces[prod] = chunk_pieces

#             chunked_buys[0] = best_price, newvol
#             if newvol == 0:
#                 chunked_buys.pop(0)
#             if len(chunked_buys) == 0:
#                 exhausted = True
#             prod_chunked_buys[prod] = chunked_buys

#         synth_agg_orders[synth_bid] = quantity
#         synth_prod_orders[synth_bid] = price_level_pieces

#     return synth_agg_orders, synth_prod_orders

# def construct_synth_od(order_depths: dict[str, OrderDepth], weights: dict[str, int]) -> OrderDepth:
        
#     order_depth = OrderDepth()
#     order_depth.buy_orders, buy_pieces = construct_synth_buy_orders(order_depths, weights)
#     neg_weights = {prod: -weights[prod] for prod in weights}
#     order_depth.sell_orders, sell_pieces = construct_synth_buy_orders(order_depths, neg_weights)

#     return order_depth, buy_pieces, sell_pieces
