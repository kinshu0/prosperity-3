import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Define products and position limits
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

# Strike prices for each voucher
strike_prices = {
    Product.VOUCHER_9500: 9500,
    Product.VOUCHER_9750: 9750,
    Product.VOUCHER_10000: 10000,
    Product.VOUCHER_10250: 10250,
    Product.VOUCHER_10500: 10500
}

# Helper functions for options pricing and analysis
def realized_vol(df, window, step_size):
    """Calculate realized volatility over a rolling window"""
    df[f'log_return_{step_size}'] = np.log(df['mid_price_rock'].to_numpy()/df['mid_price_rock'].shift(step_size).to_numpy())
    dt = step_size / 250 / 10000 
    df[f'realized_vol_{step_size}'] = df[f'log_return_{step_size}'].rolling(window=window).apply(lambda x: np.mean(x[::step_size]**2) / dt)
    df[f'realized_vol_{step_size}'] = np.sqrt(df[f'realized_vol_{step_size}'].to_numpy())
    return df

def black_scholes_call(spot, strike, time_to_expiry, volatility):
    """Calculate theoretical call price using Black-Scholes model"""
    d1 = (np.log(spot / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    call_price = (spot * norm.cdf(d1) - strike * norm.cdf(d2))
    return call_price

def implied_volatility(call_price, spot, strike, time_to_expiry):
    """Calculate implied volatility from market prices"""
    def equation(volatility):
        estimated_price = black_scholes_call(spot, strike, time_to_expiry, volatility)
        return estimated_price - call_price
    
    try:
        # Using Brent's method to find the root of the equation
        implied_vol = brentq(equation, 1e-10, 3.0, xtol=1e-10)
        return implied_vol
    except:
        return np.nan

def delta(spot, strike, time_to_expiry, volatility):
    """Calculate delta of an option"""
    d1 = (np.log(spot) - np.log(strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    return norm.cdf(d1)

# Data loading function
def load_data(num_days=3):
    """Load price and trade data for all days"""
    prices = pd.concat([pd.read_csv(f'round3/prices_round_3_day_{i}.csv', sep=';') for i in range(num_days)], ignore_index=True)
    trades = pd.concat([pd.read_csv(f'round3/trades_round_3_day_{i}_nn.csv', sep=';') for i in range(num_days)], ignore_index=True)
    
    # Calculate mid prices
    prices['swmid'] = (prices['bid_price_1'] * prices['ask_volume_1'] + prices['ask_price_1'] * prices['bid_volume_1']) / (prices['ask_volume_1'] + prices['bid_volume_1'])
    
    return prices, trades

# Data preprocessing function
def preprocess_data(prices):
    """Preprocess and join data for all products"""
    # Filter and rename columns for each product
    product_dfs = {}
    for product_name in [Product.ROCK] + list(strike_prices.keys()):
        df = prices[prices["product"] == product_name].reset_index(drop=True).copy()
        df = df.rename(columns={
            'ask_price_1': 'ask_price', 
            'bid_price_1': 'bid_price', 
            'ask_volume_1': 'ask_volume', 
            'bid_volume_1': 'bid_volume'
        })
        
        short_name = product_name.replace('VOLCANIC_ROCK_VOUCHER_', 'voucher').replace('VOLCANIC_ROCK', 'rock')
        
        if product_name == Product.ROCK:
            product_dfs[product_name] = df.drop(columns=['product'], axis=1).rename(
                columns={col: f'{col}_rock' for col in df.columns if col not in ['timestamp', 'day']}
            )
        else:
            product_dfs[product_name] = df.drop(columns=['product'], axis=1).rename(
                columns={col: f'{col}_{short_name}' for col in df.columns if col not in ['timestamp', 'day']}
            )
    
    # Join all dataframes on timestamp and day
    merged_df = product_dfs[Product.ROCK]
    for product_name in list(strike_prices.keys()):
        merged_df = merged_df.merge(product_dfs[product_name], on=['day', 'timestamp'])
        
    return merged_df

# Analysis function
def analyze_vouchers(df, starting_days_to_expiry=6, fixed_vol=None):
    """Calculate theoretical prices and implied volatility for all vouchers"""
    results = []
    
    for day in df['day'].unique():
        day_df = df[df['day'] == day].copy()
        
        # Calculate historical volatility if not provided
        if fixed_vol is None:
            hist_vol = np.sqrt(np.log(day_df['mid_price_rock']/day_df['mid_price_rock'].shift(1)).var() / (1/250/10000))
        else:
            hist_vol = fixed_vol
            
        # Calculate time to expiry
        day_df['tte'] = (starting_days_to_expiry - day - day_df['timestamp'] / 1_000_000) / 250
        
        # Calculate realized volatility
        day_df = realized_vol(day_df, 60, 1)
        
        # For each voucher, calculate theoretical price, implied vol, and delta
        for product_name, strike in strike_prices.items():
            short_name = product_name.replace('VOLCANIC_ROCK_VOUCHER_', 'voucher')
            
            # Calculate theoretical price
            day_df[f'theo_call_{short_name}'] = day_df.apply(
                lambda row: black_scholes_call(row['mid_price_rock'], strike, row['tte'], hist_vol), 
                axis=1
            )
            
            # Calculate difference between theoretical and market price
            mid_price_col = f'mid_price_{short_name}'
            day_df[f'diff_{short_name}'] = day_df[f'theo_call_{short_name}'] - day_df[mid_price_col]
            
            # Calculate implied volatility
            day_df[f'iv_{short_name}'] = day_df.apply(
                lambda row: implied_volatility(row[mid_price_col], row['mid_price_rock'], strike, row['tte']), 
                axis=1
            )
            
            # Calculate delta
            day_df[f'delta_{short_name}'] = day_df.apply(
                lambda row: delta(row['mid_price_rock'], strike, row['tte'], row[f'iv_{short_name}']), 
                axis=1
            )
        
        results.append(day_df)
    
    return pd.concat(results)

# Visualization functions
def plot_prices_comparison(df, day=None):
    """Plot comparison of rock price vs voucher prices"""
    if day is not None:
        df = df[df['day'] == day].copy()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add rock price
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['mid_price_rock'], mode='lines', name='Rock'),
        secondary_y=False
    )
    
    # Add voucher prices
    for product_name in strike_prices.keys():
        short_name = product_name.replace('VOLCANIC_ROCK_VOUCHER_', 'voucher')
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df[f'mid_price_{short_name}'], mode='lines', name=f'{short_name}'),
            secondary_y=True
        )
    
    # Update layout
    day_title = f'Day {day}' if day is not None else 'All Days'
    fig.update_layout(
        title=f'Rock vs Voucher Prices - {day_title}',
        xaxis_title='Timestamp',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Rock Price", secondary_y=False)
    fig.update_yaxes(title_text="Voucher Prices", secondary_y=True)
    
    return fig

def plot_implied_volatility(df, day=None):
    """Plot implied volatility for all vouchers"""
    if day is not None:
        df = df[df['day'] == day].copy()
    
    fig = go.Figure()
    
    # Add implied volatility for each voucher
    for product_name in strike_prices.keys():
        short_name = product_name.replace('VOLCANIC_ROCK_VOUCHER_', 'voucher')
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df[f'iv_{short_name}'], mode='lines', name=f'IV {short_name}')
        )
    
    # Update layout
    day_title = f'Day {day}' if day is not None else 'All Days'
    fig.update_layout(
        title=f'Implied Volatility - {day_title}',
        xaxis_title='Timestamp',
        yaxis_title='Implied Volatility',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_theoretical_vs_market(df, product_name, day=None):
    """Plot comparison between theoretical and market prices for a specific voucher"""
    if day is not None:
        df = df[df['day'] == day].copy()
    
    short_name = product_name.replace('VOLCANIC_ROCK_VOUCHER_', 'voucher')
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                         subplot_titles=(f'Theoretical vs Market Price - {short_name}', 
                                         f'Price Difference - {short_name}'))
    
    # Add theoretical and market prices
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df[f'theo_call_{short_name}'], mode='lines', name='Theoretical Price'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df[f'mid_price_{short_name}'], mode='lines', name='Market Price'),
        row=1, col=1
    )
    
    # Add price difference
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df[f'diff_{short_name}'], mode='lines', name='Difference'),
        row=2, col=1
    )
    
    # Update layout
    day_title = f'Day {day}' if day is not None else 'All Days'
    fig.update_layout(
        title=f'{short_name} Analysis - {day_title}',
        xaxis2_title='Timestamp',
        yaxis_title='Price',
        yaxis2_title='Difference',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_delta(df, day=None):
    """Plot delta for all vouchers"""
    if day is not None:
        df = df[df['day'] == day].copy()
    
    fig = go.Figure()
    
    # Add delta for each voucher
    for product_name in strike_prices.keys():
        short_name = product_name.replace('VOLCANIC_ROCK_VOUCHER_', 'voucher')
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df[f'delta_{short_name}'], mode='lines', name=f'Delta {short_name}')
        )
    
    # Update layout
    day_title = f'Day {day}' if day is not None else 'All Days'
    fig.update_layout(
        title=f'Option Delta - {day_title}',
        xaxis_title='Timestamp',
        yaxis_title='Delta',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_iv_distribution(df, day=None):
    """Plot implied volatility distribution for all vouchers"""
    if day is not None:
        df = df[df['day'] == day].copy()
        
    fig = make_subplots(rows=1, cols=len(strike_prices), 
                         subplot_titles=[f'IV Distribution - {p.replace("VOLCANIC_ROCK_VOUCHER_", "voucher")}' 
                                         for p in strike_prices.keys()])
    
    # Add histogram for each voucher
    for i, product_name in enumerate(strike_prices.keys(), 1):
        short_name = product_name.replace('VOLCANIC_ROCK_VOUCHER_', 'voucher')
        fig.add_trace(
            go.Histogram(x=df[f'iv_{short_name}'].dropna(), nbinsx=50, name=short_name),
            row=1, col=i
        )
    
    # Update layout
    day_title = f'Day {day}' if day is not None else 'All Days'
    fig.update_layout(
        title=f'Implied Volatility Distribution - {day_title}',
        showlegend=False,
        height=400
    )
    
    return fig

def plot_iv_statistics(df):
    """Plot implied volatility statistics for all vouchers by day"""
    iv_stats = []
    
    for day in df['day'].unique():
        day_df = df[df['day'] == day]
        
        for product_name in strike_prices.keys():
            short_name = product_name.replace('VOLCANIC_ROCK_VOUCHER_', 'voucher')
            iv_col = f'iv_{short_name}'
            
            # Calculate statistics
            stats = {
                'day': day,
                'product': short_name,
                'mean': day_df[iv_col].mean(),
                'median': day_df[iv_col].median(),
                'std': day_df[iv_col].std(),
                'min': day_df[iv_col].min(),
                'max': day_df[iv_col].max()
            }
            
            iv_stats.append(stats)
    
    iv_stats_df = pd.DataFrame(iv_stats)
    
    # Create figure with subplots
    fig = make_subplots(rows=2, cols=2, 
                         subplot_titles=('Mean IV by Day', 'Median IV by Day', 
                                         'Min/Max IV by Day', 'IV Standard Deviation by Day'))
    
    # Colors for different vouchers
    colors = px.colors.qualitative.Plotly
    
    # Plot mean IV
    for i, product in enumerate(iv_stats_df['product'].unique()):
        product_df = iv_stats_df[iv_stats_df['product'] == product]
        fig.add_trace(
            go.Scatter(x=product_df['day'], y=product_df['mean'], mode='lines+markers', 
                       name=product, line=dict(color=colors[i])),
            row=1, col=1
        )
    
    # Plot median IV
    for i, product in enumerate(iv_stats_df['product'].unique()):
        product_df = iv_stats_df[iv_stats_df['product'] == product]
        fig.add_trace(
            go.Scatter(x=product_df['day'], y=product_df['median'], mode='lines+markers', 
                       name=product, line=dict(color=colors[i]), showlegend=False),
            row=1, col=2
        )
    
    # Plot min/max IV
    for i, product in enumerate(iv_stats_df['product'].unique()):
        product_df = iv_stats_df[iv_stats_df['product'] == product]
        
        # Min values
        fig.add_trace(
            go.Scatter(x=product_df['day'], y=product_df['min'], mode='lines+markers',
                       name=f'{product} Min', line=dict(color=colors[i], dash='dot'), showlegend=False),
            row=2, col=1
        )
        
        # Max values
        fig.add_trace(
            go.Scatter(x=product_df['day'], y=product_df['max'], mode='lines+markers',
                       name=f'{product} Max', line=dict(color=colors[i]), showlegend=False),
            row=2, col=1
        )
    
    # Plot standard deviation
    for i, product in enumerate(iv_stats_df['product'].unique()):
        product_df = iv_stats_df[iv_stats_df['product'] == product]
        fig.add_trace(
            go.Scatter(x=product_df['day'], y=product_df['std'], mode='lines+markers',
                       name=product, line=dict(color=colors[i]), showlegend=False),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='Implied Volatility Statistics by Day',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=800
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Mean IV", row=1, col=1)
    fig.update_yaxes(title_text="Median IV", row=1, col=2)
    fig.update_yaxes(title_text="Min/Max IV", row=2, col=1)
    fig.update_yaxes(title_text="IV Std Dev", row=2, col=2)
    
    return fig

# Main function to run the analysis
def main():
    # Load data
    prices, trades = load_data()
    
    # Preprocess data
    merged_df = preprocess_data(prices)
    
    # Run analysis
    results_df = analyze_vouchers(merged_df)
    
    # Generate visualizations
    
    # 1. Overall price comparison for all days
    fig_prices_all = plot_prices_comparison(results_df)
    fig_prices_all.show()
    
    # 2. Price comparison by day
    for day in results_df['day'].unique():
        fig_prices_day = plot_prices_comparison(results_df, day)
        fig_prices_day.show()
    
    # 3. Implied volatility for all days
    fig_iv_all = plot_implied_volatility(results_df)
    fig_iv_all.show()
    
    # 4. Implied volatility by day
    for day in results_df['day'].unique():
        fig_iv_day = plot_implied_volatility(results_df, day)
        fig_iv_day.show()
    
    # 5. Theoretical vs market price for each voucher
    for product_name in strike_prices.keys():
        fig_theo_market = plot_theoretical_vs_market(results_df, product_name)
        fig_theo_market.show()
    
    # 6. Delta for all vouchers
    fig_delta = plot_delta(results_df)
    fig_delta.show()
    
    # 7. Implied volatility distribution
    fig_iv_dist = plot_iv_distribution(results_df)
    fig_iv_dist.show()
    
    # 8. Implied volatility statistics by day
    fig_iv_stats = plot_iv_statistics(results_df)
    fig_iv_stats.show()
    
    # Return results for further analysis if needed
    return results_df

if __name__ == "__main__":
    results = main()