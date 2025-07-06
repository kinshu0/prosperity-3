

class Product:
    CROISSANTS = 'CROISSANTS'
    JAMS = 'JAMS'
    DJEMBES = 'DJEMBES'
    BASKET = 'BASKET'



mids = {}
    for leg in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]:
        if leg not in state.order_depths:
            continue
        depth = state.order_depths[leg]
        mid = self._mid(depth)
        if mid is None:
            continue
        mids[leg] = mid
        edge = self.params[leg]['edge']
        pos  = state.position.get(leg, 0)
        size = max(1, int(0.2 * (self.LIMIT[leg] - abs(pos))))
        if size == 0:
            continue
        # if product is Jams, then the buy edge is 150
        if leg == Product.JAMS:
            result[leg].append(Order(leg, math.floor(mid - edge) - 150,  size))
            result[leg].append(Order(leg, math.ceil (mid + edge), -size))
        else:
            result[leg].append(Order(leg, math.floor(mid - edge),  size))
            result[leg].append(Order(leg, math.ceil (mid + edge), -size))

    # BASKEY we try arbitrage
    for basket, recipe in BASKET_RECIPE.items():
        if basket not in state.order_depths or not all(c in mids for c in recipe):
            continue
        fv = sum(qty * mids[c] for c, qty in recipe.items())
        depth_b = state.order_depths[basket]
        bid_b, ask_b = self._best_bid_ask(depth_b)
        if bid_b is None or ask_b is None:
            continue
        thresh = self.params[basket]['spread_threshold']
        max_leg = self.params[basket]['max_qty_per_leg']
        pos_b   = state.position.get(basket, 0)

        # if basket is rich then sell basket / buy legs
        if bid_b - fv > thresh:
            qty_b = min(depth_b.buy_orders[bid_b], self.LIMIT[basket] + pos_b, max_leg)
            if qty_b > 0:
                result[basket].append(Order(basket, bid_b, -qty_b))
                # for leg, mult in recipe.items():
                #     depth_leg = deepcopy(state.order_depths[leg])
                #     leg_pos   = state.position.get(leg, 0)
                #     qty_leg   = min(qty_b * mult, self.LIMIT[leg] - leg_pos)
                #     self._cross(leg, depth_leg, qty_leg, 'buy', result[leg])

        # if basket cheap gpal is to buy basket / sell legs
        elif ask_b - fv < -thresh:
            qty_b = min(-depth_b.sell_orders[ask_b], self.LIMIT[basket] - pos_b, max_leg)
            if qty_b > 0:
                result[basket].append(Order(basket, ask_b, qty_b))
                # for leg, mult in recipe.items():
                #     depth_leg = deepcopy(state.order_depths[leg])
                #     leg_pos   = state.position.get(leg, 0)

                #     qty_leg   = min(qty_b * mult, leg_pos + self.LIMIT[leg])
                #     self._cross(leg, depth_leg, qty_leg, 'sell', result[leg])


# This is the core idea, I was getting kinda lost and decide to just start from scratch, but the leg code was just not working