# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# %%
'''Position limits for the newly introduced products:

- `VOLCANIC_ROCK`: 400

`VOLCANIC_ROCK_VOUCHER_9500` :

- Position Limit: 200
- Strike Price: 9,500 SeaShells
- Expiration deadline: 7 days (1 round = 1 day) starting from round 1

`VOLCANIC_ROCK_VOUCHER_9750` :

- Position Limit: 200
- Strike Price: 9,750 SeaShells
- Expiration deadline: 7 days (1 round = 1 day) starting from round 1

`VOLCANIC_ROCK_VOUCHER_10000` :

- Position Limit: 200
- Strike Price: 10,000 SeaShells
- Expiration deadline: 7 days (1 round = 1 day) starting from round 1

`VOLCANIC_ROCK_VOUCHER_10250` :

- Position Limit: 200
- Strike Price: 10,250 SeaShells
- Expiration deadline: 7 days (1 round = 1 day) starting from round 1

`VOLCANIC_ROCK_VOUCHER_10500` :

- Position Limit: 200
- Strike Price: 10,500 SeaShells
- Expiration deadline: 7 days (1 round = 1 day) starting from round 1'''


class Product:
    ROCK = 'VOLCANIC_ROCK'
    VOUCHER_9500 = 'VOLCANIC_ROCK_VOUCHER_9500'
    VOUCHER_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
    VOUCHER_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
    VOUCHER_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
    VOUCHER_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'

position_limits = {
    Product.ROCK: 400,
    Product.VOUCHER_9500: 200,
    Product.VOUCHER_9750: 200,
    Product.VOUCHER_10000: 200,
    Product.VOUCHER_10250: 200,
    Product.VOUCHER_10500: 200
}

# %%
prices = pd.concat([pd.read_csv(f'round3/prices_round_3_day_{i}.csv', sep=';') for i in range(3)], ignore_index=True)
trades = pd.concat([pd.read_csv(f'round3/trades_round_3_day_{i}_nn.csv', sep=';') for i in range(3)], ignore_index=True)

# %%
prices['swmid'] = (prices['bid_price_1'] * prices['ask_volume_1'] + prices['ask_price_1'] * prices['bid_volume_1']) / (prices['ask_volume_1'] + prices['bid_volume_1'])

# %%
rock = prices[prices["product"] == Product.ROCK].reset_index(drop=True).copy()
voucher_9500 = prices[prices["product"] == Product.VOUCHER_9500].reset_index(drop=True).copy()
voucher_9750 = prices[prices["product"] == Product.VOUCHER_9750].reset_index(drop=True).copy()
voucher_10000 = prices[prices["product"] == Product.VOUCHER_10000].reset_index(drop=True).copy()
voucher_10250 = prices[prices["product"] == Product.VOUCHER_10250].reset_index(drop=True).copy()
voucher_10500 = prices[prices["product"] == Product.VOUCHER_10500].reset_index(drop=True).copy()

# %%
[rock, voucher_9500, voucher_9750, voucher_10000, voucher_10250, voucher_10500] = [df.rename(columns={'ask_price_1': 'ask_price', 'bid_price_1': 'bid_price', 'ask_volume_1': 'ask_volume', 'bid_volume_1': 'bid_volume'}) for df in [rock, voucher_9500, voucher_9750, voucher_10000, voucher_10250, voucher_10500]]

# %%
rock = rock.drop(columns=['product'], axis=1).rename(columns={col: col + '_rock' for col in rock.columns if col not in ['timestamp', 'day']})
voucher_9500 = voucher_9500.drop(columns=['product'], axis=1).rename(columns={col: col + '_voucher9500' for col in voucher_9500.columns if col not in ['timestamp', 'day']})
voucher_9750 = voucher_9750.drop(columns=['product'], axis=1).rename(columns={col: col + '_voucher9750' for col in voucher_9750.columns if col not in ['timestamp', 'day']})
voucher_10000 = voucher_10000.drop(columns=['product'], axis=1).rename(columns={col: col + '_voucher10000' for col in voucher_10000.columns if col not in ['timestamp', 'day']})
voucher_10250 = voucher_10250.drop(columns=['product'], axis=1).rename(columns={col: col + '_voucher10250' for col in voucher_10250.columns if col not in ['timestamp', 'day']})
voucher_10500 = voucher_10500.drop(columns=['product'], axis=1).rename(columns={col: col + '_voucher10500' for col in voucher_10500.columns if col not in ['timestamp', 'day']})

# %%
# join croissants, james, djembes, basket1, basket2 on timestamp
mk = rock.merge(voucher_9500, on=['day', 'timestamp'])
mk = mk.merge(voucher_9750, on=['day', 'timestamp'])
mk = mk.merge(voucher_10000, on=['day', 'timestamp'])
mk = mk.merge(voucher_10250, on=['day', 'timestamp'])
mk = mk.merge(voucher_10500, on=['day', 'timestamp'])
mk

# %%
q = mk['day'] == 2
df = mk[q].copy()

fig = go.Figure()

fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mid_price_rock'], mode='lines', name='Rock'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mid_price_voucher10000'], mode='lines', name='Voucher 10000', yaxis='y2'))
fig.update_layout(
    title='Rock and Voucher 10000 Prices',
    xaxis_title='Timestamp',
    yaxis_title='Price',
    yaxis2=dict(
        title='Voucher 10000 Price',
        overlaying='y',
        side='right'
    )
)
fig.show()

# %%
from scipy.stats import norm

def realized_vol(df, window, step_size):
    df[f'log_return_{step_size}'] = np.log(df['mid_price_rock'].to_numpy()/df['mid_price_rock'].shift(step_size).to_numpy())
    dt = step_size / 250 / 10000 
    df[f'realized_vol_{step_size}'] = df[f'log_return_{step_size}'].rolling(window=window).apply(lambda x: np.mean(x[::step_size]**2) / dt)
    df[f'realized_vol_{step_size}'] = np.sqrt(df[f'realized_vol_{step_size}'].to_numpy())
    return df

def black_scholes_call(spot, strike, time_to_expiry, volatility):
    d1 = (np.log(spot / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    call_price = (spot * norm.cdf(d1) - strike * norm.cdf(d2))
    return call_price

# %%
# np.log(df['mid_price_rock']/df['mid_price_rock'].shift(1)).var() / (1/250/10000)
np.log(df['mid_price_rock']/df['mid_price_rock'].shift(1)).var() / (1/250/10000)

# %%
df['mid_price_rock']

# %%
starting_days_to_expiry = 6
strike = 10_000

# hist_vol = 0.1643
hist_vol = 0.09859

realized_vol(df, 60, 1)
df['tte'] = tte = (starting_days_to_expiry - df['timestamp'] / 1_000_000) / 250
# df['tte'] = tte = (starting_days_to_expiry - df['timestamp'] / 1_000_000) / 250
# df['theo_call'] = df.apply(lambda row: black_scholes_call(row['mid_price_rock'], strike, row['tte'], row['realized_vol_1']), axis=1)
df['theo_call'] = df.apply(lambda row: black_scholes_call(row['mid_price_rock'], strike, row['tte'], hist_vol), axis=1)

# plot theo_call and mid_price_voucher10000
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['theo_call'], mode='lines', name='Theoretical Call Price'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mid_price_voucher10000'], mode='lines', name='Voucher 10000 Price'))
fig.update_layout(
    title='Theoretical Call Price and Voucher 10000 Prices',
    xaxis_title='Timestamp',
    yaxis_title='Price'
)
fig.show()

# %%
# plot difference between theo_call and mid_price_voucher10000
df['diff'] = df['theo_call'] - df['mid_price_voucher10000']
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['diff'], mode='lines', name='Difference'))
fig.update_layout(
    title='Difference between Theoretical Call Price and Voucher 10000 Prices',
    xaxis_title='Timestamp',
    yaxis_title='Difference'
)
fig.show()

# %%
# plot realized_vol_1 against timestamp
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['realized_vol_1'], mode='lines', name='Realized Volatility'))
fig.update_layout(
    title='Realized Volatility',
    xaxis_title='Timestamp',
    yaxis_title='Realized Volatility'
)
fig.show()

# %%
from scipy.optimize import brentq

def implied_volatility(call_price, spot, strike, time_to_expiry):
    # Define the equation where the root is the implied volatility
    def equation(volatility):
        estimated_price = black_scholes_call(spot, strike, time_to_expiry, volatility)
        return estimated_price - call_price

    # Using Brent's method to find the root of the equation
    implied_vol = brentq(equation, 1e-10, 3.0, xtol=1e-10)
    return implied_vol

df['iv'] = df.apply(lambda row: implied_volatility(row['mid_price_voucher10000'], row['mid_price_rock'], strike, row['tte']), axis=1)

df['iv'].plot()

# %%
df['iv'].describe()

# %%
df['iv'].median()

# %%
df['iv'].hist(bins=100)

# %%
def delta(spot, strike, time_to_expiry, volatility):
    d1 = (np.log(spot) - np.log(strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    return norm.cdf(d1)

df['delta'] = df.apply(lambda row: delta(row['mid_price_rock'], strike, row['tte'], row['iv']), axis=1)

# %%
df['delta'].plot()


