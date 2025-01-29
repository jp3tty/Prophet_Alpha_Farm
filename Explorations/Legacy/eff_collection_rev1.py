import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

def fetch_stock_data(ticker, period='1y'):
    """
    Fetch stock data from Yahoo Finance
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist

def calculate_metrics(data, ticker):
    """
    Calculate key metrics for portfolio analysis
    """
    # Calculate daily returns
    daily_returns = data['Close'].pct_change()
    
    # Calculate annualized metrics
    trading_days = 252
    avg_return = daily_returns.mean() * trading_days
    volatility = daily_returns.std() * np.sqrt(trading_days)
    sharpe_ratio = avg_return / volatility  # Assuming risk-free rate of 0 for simplicity
    
    # Create metrics dictionary
    metrics = {
        'Ticker': ticker,
        'Annualized Return': avg_return,
        'Annualized Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio
    }
    
    return metrics, daily_returns

def analyze_stocks(tickers, period='1y', output_dir='stock_data'):
    """
    Analyze multiple stocks and save results to CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Store metrics for all stocks
    all_metrics = []
    all_returns = pd.DataFrame()
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        
        # Fetch and analyze data
        stock_data = fetch_stock_data(ticker, period)
        metrics, daily_returns = calculate_metrics(stock_data, ticker)
        
        # Add metrics to list
        all_metrics.append(metrics)
        
        # Add returns to DataFrame
        all_returns[ticker] = daily_returns
        
        # Save raw stock data to CSV
        stock_data_filename = os.path.join(output_dir, f"{ticker}_raw_data.csv")
        stock_data.to_csv(stock_data_filename)
        print(f"Raw data saved to: {stock_data_filename}")
        
        # Print metrics
        print(f"Annualized Return: {metrics['Annualized Return']:.2%}")
        print(f"Annualized Volatility: {metrics['Annualized Volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    
    # Save metrics summary to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_filename = os.path.join(output_dir, "all_metrics_summary.csv")
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"\nMetrics summary saved to: {metrics_filename}")
    
    # Save returns to CSV
    returns_filename = os.path.join(output_dir, "all_daily_returns.csv")
    all_returns.to_csv(returns_filename)
    print(f"Daily returns saved to: {returns_filename}")
    
    return metrics_df, all_returns

# Example usage
tickers = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'EGO', 'STM', 'IBM', 'NVDA', 'PLTR', 'SBUX', 'USO', 'GLD', 'F', 'DHI', 'APP'] # Stock inputs
metrics_df, returns_df = analyze_stocks(tickers, period='2y')          # Output directory and time period can be added: metrics_df, returns_df = analyze_stocks(tickers, period='2y', output_dir='my_analysis')

# Create correlation matrix visualization
plt.figure(figsize=(10, 8))
correlation_matrix = returns_df.corr()
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(tickers)), tickers, rotation=45)
plt.yticks(range(len(tickers)), tickers)
plt.title('Stock Returns Correlation Matrix')
plt.tight_layout()
plt.show()

# Create risk-return scatter plot
plt.figure(figsize=(10, 6))
for _, row in metrics_df.iterrows():
    plt.scatter(row['Annualized Volatility'], row['Annualized Return'], label=row['Ticker'])
plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.title('Risk-Return Profile by Stock')
plt.grid(True)
plt.legend()
plt.show()