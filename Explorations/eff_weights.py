import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar
import os

def fetch_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    return stock.history(period=period)

def get_optimal_portfolios(tickers, period='1y', num_simulations=1000):
    """
    Calculate and display optimal portfolios with clear allocations
    """
    print(f"\nFetching data for {len(tickers)} stocks...")
    
    # Fetch data and calculate returns
    returns = pd.DataFrame()
    for ticker in tickers:
        print(f"Downloading {ticker} data...")
        data = fetch_stock_data(ticker, period)
        returns[ticker] = data['Close'].pct_change()
    
    print(f"\nRunning {num_simulations:,} portfolio simulations...")
    
    # Run simulations
    num_assets = len(tickers)
    results = np.zeros((num_simulations, 3))
    all_weights = np.zeros((num_simulations, num_assets))
    
    # Use tqdm for progress bar
    for i in tqdm(range(num_simulations), desc="Simulating portfolios"):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_vol
        
        results[i] = [portfolio_return, portfolio_vol, sharpe_ratio]
        all_weights[i] = weights
    
    print("\nFinding optimal portfolios...")
    
    # Find optimal portfolios
    max_sharpe_idx = np.argmax(results[:, 2])
    min_vol_idx = np.argmin(results[:, 1])
    
    # Create portfolio objects with clear allocations
    max_sharpe_portfolio = {
        'name': 'Maximum Sharpe Ratio Portfolio',
        'return': results[max_sharpe_idx, 0],
        'volatility': results[max_sharpe_idx, 1],
        'sharpe': results[max_sharpe_idx, 2],
        'allocations': dict(zip(tickers, all_weights[max_sharpe_idx] * 100))
    }
    
    min_vol_portfolio = {
        'name': 'Minimum Volatility Portfolio',
        'return': results[min_vol_idx, 0],
        'volatility': results[min_vol_idx, 1],
        'sharpe': results[min_vol_idx, 2],
        'allocations': dict(zip(tickers, all_weights[min_vol_idx] * 100))
    }
    
    # Plot efficient frontier with optimal portfolios
    print("\nGenerating efficient frontier plot...")
    plt.figure(figsize=(12, 8))
    plt.scatter(results[:, 1], results[:, 0], c=results[:, 2], 
                cmap='viridis', alpha=0.5, label=f'Simulated Portfolios ({num_simulations:,})')
    
    # Plot optimal portfolios
    plt.scatter(max_sharpe_portfolio['volatility'], max_sharpe_portfolio['return'], 
                color='red', marker='*', s=200, label='Maximum Sharpe Ratio')
    plt.scatter(min_vol_portfolio['volatility'], min_vol_portfolio['return'], 
                color='green', marker='*', s=200, label='Minimum Volatility')
    
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Expected Volatility (%)')
    plt.ylabel('Expected Return (%)')
    plt.title(f'Efficient Frontier ({num_simulations:,} Simulated Portfolios)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return max_sharpe_portfolio, min_vol_portfolio

def print_portfolio(portfolio):
    """Helper function to print portfolio details"""
    print(f"\n{portfolio['name']}:")
    print(f"Expected Annual Return: {portfolio['return']*100:.2f}%")
    print(f"Expected Annual Volatility: {portfolio['volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {portfolio['sharpe']:.2f}")
    print("\nAllocation:")
    # Sort allocations by weight in descending order
    sorted_allocations = dict(sorted(portfolio['allocations'].items(), 
                                   key=lambda x: x[1], reverse=True))
    for stock, weight in sorted_allocations.items():
        print(f"{stock}: {weight:.1f}%")

# Example usage
tickers = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'EGO', 'STM', 'IBM', 'NVDA', 'PLTR', 'SBUX', 'USO', 'GLD', 'F', 'DHI', 'APP']
num_sims = 10000  # Increased number of simulations for better results

print(f"Starting portfolio optimization with {len(tickers)} stocks and {num_sims:,} simulations...")
max_sharpe, min_vol = get_optimal_portfolios(tickers, num_simulations=num_sims)

print("\n=== OPTIMIZATION RESULTS ===")
print(f"Total Simulations Run: {num_sims:,}")
print_portfolio(max_sharpe)
print_portfolio(min_vol)