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
    'EGO',   # Eldorado Gold Corp
    'STM',   # STMicroelectronics NV
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
        
        if df.empty:
            print(f"No data received for {symbol}")
            return pd.DataFrame()
            
        # Create a new DataFrame with only the columns we want
        price_data = pd.DataFrame()
        price_data['Date'] = df.index
        price_data['Price'] = df['Close'].values
        
        # Convert date to string format YYYY-MM-DD
        price_data['Date'] = price_data['Date'].dt.strftime('%Y-%m-%d')
        
        print(f"Retrieved {len(price_data)} days of data for {symbol}")
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
        return False
        
    try:
        # Create 'price_data' directory if it doesn't exist
        os.makedirs('../Data/price_data', exist_ok=True)
        
        # Create filename with current date
        filename = f"../Data/price_data/{symbol}_prices.csv"
        
        # Save to CSV
        price_data.to_csv(filename, index=False)
        
        # Verify file was created and has data
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            print(f"Successfully saved {len(price_data)} rows of price data for {symbol} to {filename}")
            return True
        else:
            print(f"Failed to save data for {symbol} - file is empty or not created")
            return False
            
    except Exception as e:
        print(f"Error saving data for {symbol}: {str(e)}")
        return False

def main():
    print(f"Starting price data collection for {len(TICKER_SYMBOLS)} symbols...")
    successful_saves = 0
    
    for symbol in TICKER_SYMBOLS:
        print(f"\nProcessing {symbol}...")
        
        # Get the price data
        price_data = get_stock_prices(symbol)
        
        # Save to CSV and track success
        if save_to_csv(price_data, symbol):
            successful_saves += 1
        
        # Add a small delay to avoid overwhelming the server
        time.sleep(1)
    
    print(f"\nPrice data collection completed!")
    print(f"Successfully saved data for {successful_saves} out of {len(TICKER_SYMBOLS)} symbols")
    
    # List all saved files
    if successful_saves > 0:
        print("\nSaved files:")
        for filename in os.listdir('../Data/price_data'):
            filepath = os.path.join('../Data/price_data', filename)
            size = os.path.getsize(filepath)
            print(f"- {filename} ({size} bytes)")

if __name__ == "__main__":
    main()