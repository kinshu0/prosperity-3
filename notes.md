# Trading Strategy Notes

## Results Comparison

### Tutorial Round Results
**Opponent:**
- RAINFOREST_RESIN:
  - Final PnL: 1976.0
  - Highest PnL: 1976.0
  - Lowest PnL: 0.0
- KELP:
  - Final PnL: 1503.84375
  - Highest PnL: 1571.3046875
  - Lowest PnL: 0.0
- Total number of trades: 728

**My Results:**
- Resin final profit: 3526.0
- Kelp final profit: 778.49609375
- Total profit: 4304.49609375

## Analysis

### Univariate Linear Regression
Predicting future pct_change from past pct_change:
- Return Window: 1
- beta_1 = -0.18172393033850867
- slope coefficient = 0.0
- R-squared: 0.03293613278400631
- MSE: 2.430140904154592e-08

## Trading Strategy Components

### Trader Overview
1. **Market Take**
   - Threshold from fair at which to take
   - Position to take

2. **Market Clear**
   - Clears inventory risk, not actively trying to make edge on it
   - Profit is byproduct of risk management
   - Considers current orders as well as trying to clear previous positions

3. **Market Make**
   - Pennying: threshold at which to beat best and by how much
   - Joining: threshold at which to join best and by how much

*All components have volume specifications*

## Parameters

### General Parameters
- **fair_value**: function to specify fair value calculation

### Take Parameters
- **take_width**: min margin from fair to submit trade on
- **prevent_adverse**: if we are preventing adverse, we don't take order
- **adverse_volume**: if the best ask volume is less than a specified adverse volume, we're not adversely selecting

### Clear Parameters
- **clear_width**: price threshold from fair to clear positions at (only takes, does not make)

### Make Parameters
- **disregard_edge**: disregard trades within this edge for pennying or joining
- **join_edge**: orders within this far from (fair value + disregard_edge) will be joined instead of undercut
- **default_edge**: standard edge value
- **manage_position**: try to manage position by asymmetric quote to neutralize position
- **soft_position_limit**: position limit threshold

## Execution Order
1. Market make
2. Market clear
3. Market take

*After every algorithm, order depths are updated to make sure next algorithm doesn't find trades already matched against*

## Questions and Issues

### Main Questions
- Why are we only buying orders with best bid/ask? Why not other price levels that also satisfy requirements?
- Do we really need to try to manage position if we're confident in fair value? How much impact does end-of-day position closing have on net PnL?
- Is adverse volume and prevent_adverse legitimate? Does price history indicate such problems?
- How do we test market making parameters?

### To-Do
- Are the different price levels from order depth typically sorted?
- What is the volume for each price level typically?
- Maybe the best bid/ask always have the largest volume and that's why we only need to trade with them?

### Interesting Idea
Would be interesting to make a game where different agents trade against each other:
- Could solve problem of time precedence by randomly determining each iteration's sequence for taking orders

## Findings

### Starfruit Analysis
- Bimodal distribution of bid_volume_1, 2, 3 and ask volumes
- Left: 0-10 volume orders
- Right: 20-30 volume orders
- EXACTLY ZERO orders with volume between 10-20
- Makes sense that only orders of more than 20 volume are counted for mid price
- Separate participants? Bot market makers in the contest?
- Trading with them would be adverse?
- Use last timestamp mm price over small taker from current timestamp

### Linear Regression Insights
- Using linreg on previous returns to get fair value of starfruit
- Sentiment wasn't that we don't use linreg, but that using linreg on previous few return values is the same as doing mean
- Makes sense because this is linreg on past few correlated variables, not on past few actual RETURNS
- "IT'S JUST REALLY STUPID TO USE THE PAST PRICES AS REGRESSION VARIABLES INSTEAD OF RETURNS BC OF MULTICOLLINEARITY, WHICH I LEARNED IN ECONOMETRICS"