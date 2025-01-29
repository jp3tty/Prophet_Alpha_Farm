import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import time

# List of ticker symbols to analyze
TICKER_SYMBOLS = [
    'AAPL',  # Apple
    'GOOGL', # Google
    'MSFT',  # Microsoft
    'AMZN',  # Amazon
    # Add more tickers here as needed
]

def get_stock_prices(symbol, period='1y'):
    """
    Fetch historical stock prices for a given symbol.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
        period (str): Time period to fetch (default: '1y' for 1 year)
    
    Returns:
        pandas.DataFrame: DataFrame with date and closing prices
    """
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Get historical data
        df = ticker.history(period=period)
        
        # Reset index to make date a column
        df.reset_index(inplace=True)
        
        # Select only Date and Close columns
        price_data = df[['Date', 'Close']]
        
        # Rename columns
        price_data.columns = ['Date', 'Price']
        
        # Convert date to string format YYYY-MM-DD
        price_data['Date'] = price_data['Date'].dt.strftime('%Y-%m-%d')
        
        return price_data
        
    except Exception as e:
        print(f"Error fetching price data for {symbol}: {str(e)}")
        return pd.DataFrame()

def save_to_csv(price_data, symbol):
    """
    Save price data to a CSV file.
    
    Args:
        price_data (pandas.DataFrame): DataFrame with price data
        symbol (str): Stock symbol for filename
    """
    if price_data.empty:
        print(f"No price data to save for {symbol}")
        return
        
    # Create 'price_data' directory if it doesn't exist
    os.makedirs('price_data', exist_ok=True)
    
    # Create filename with current date
    filename = f"price_data/{symbol}_prices_{datetime.now().strftime('%Y%m%d')}.csv"
    
    # Save to CSV
    price_data.to_csv(filename, index=False)
    print(f"Saved price data for {symbol} to {filename}")

def main():
    # First, make sure we have the required package
    try:
        import yfinance
    except ImportError:
        print("yfinance package not found. Installing...")
        import pip
        pip.main(['install', 'yfinance'])
    
    print(f"Starting price data collection for {len(TICKER_SYMBOLS)} symbols...")
    
    for symbol in TICKER_SYMBOLS:
        print(f"\nProcessing {symbol}...")
        
        # Get the price data
        price_data = get_stock_prices(symbol)
        
        # Save to CSV
        save_to_csv(price_data, symbol)
        
        # Add a small delay to avoid overwhelming the server
        time.sleep(1)
    
    print("\nPrice data collection completed!")

if __name__ == "__main__":
    main()