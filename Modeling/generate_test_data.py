"""
Test Data Generator

This script generates synthetic stock price data files for testing the time series models.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_stock_data(symbol, days=365, volatility=0.02, trend=0.0001):
    """
    Generate synthetic stock price data.
    
    Args:
        symbol (str): Stock symbol
        days (int): Number of days of data to generate
        volatility (float): Daily volatility factor
        trend (float): Daily trend factor (positive for uptrend, negative for downtrend)
    
    Returns:
        pd.DataFrame: DataFrame with generated data
    """
    # Start from roughly a year ago
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Set initial price (random between 50 and 200)
    initial_price = np.random.uniform(50, 200)
    
    # Generate returns with random walk (plus trend)
    returns = np.random.normal(trend, volatility, len(date_range))
    
    # Convert returns to prices
    price_path = initial_price * (1 + np.cumsum(returns))
    
    # Create DataFrame
    df = pd.DataFrame({
        'ds': date_range,
        'y': price_path
    })
    
    return df

def main():
    # Create Data directory if it doesn't exist
    data_dir = "../Data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Generate data for 3 different stocks
    stocks = [
        {'symbol': 'AAPL', 'volatility': 0.015, 'trend': 0.0002},
        {'symbol': 'MSFT', 'volatility': 0.012, 'trend': 0.0003},
        {'symbol': 'AMZN', 'volatility': 0.020, 'trend': 0.0001}
    ]
    
    for stock in stocks:
        # Generate data
        df = generate_stock_data(
            symbol=stock['symbol'], 
            volatility=stock['volatility'], 
            trend=stock['trend']
        )
        
        # Save to CSV
        file_path = os.path.join(data_dir, f"{stock['symbol']}_prices.csv")
        df.to_csv(file_path, index=False)
        print(f"Generated {len(df)} days of data for {stock['symbol']} -> {file_path}")

if __name__ == "__main__":
    main() 