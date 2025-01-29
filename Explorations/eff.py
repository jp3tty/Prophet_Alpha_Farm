import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

def fetch_stock_data(ticker, period='2y'):
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
    sharpe_ratio = avg_return / volatility
    
    metrics = {
        'Ticker': ticker,
        'Annualized Return': avg_return,
        'Annualized Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio
    }
    
    return metrics, daily_returns

def calculate_portfolio_metrics(weights, returns):
    """
    Calculate portfolio return and volatility
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_vol
    return portfolio_return, portfolio_vol, sharpe_ratio

def simulate_portfolios(returns, num_portfolios=1000):
    """
    Run Monte Carlo simulation to generate random portfolios
    """
    num_assets = returns.shape[1]
    results = np.zeros((num_portfolios, 3))  # Return, Volatility, Sharpe Ratio
    all_weights = np.zeros((num_portfolios, num_assets))
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        
        # Calculate portfolio metrics
        portfolio_return, portfolio_vol, sharpe_ratio = calculate_portfolio_metrics(weights, returns)
        
        # Store results
        results[i] = [portfolio_return, portfolio_vol, sharpe_ratio]
        all_weights[i] = weights
    
    return results, all_weights

def analyze_stocks(tickers, period='1y', output_dir='stock_data', num_simulations=1000):
    """
    Analyze multiple stocks and save results to CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []
    all_returns = pd.DataFrame()
    
    # Fetch and analyze individual stocks
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        stock_data = fetch_stock_data(ticker, period)
        metrics, daily_returns = calculate_metrics(stock_data, ticker)
        
        all_metrics.append(metrics)
        all_returns[ticker] = daily_returns
        
        stock_data_filename = os.path.join(output_dir, f"{ticker}_raw_data.csv")
        stock_data.to_csv(stock_data_filename)
        print(f"Raw data saved to: {stock_data_filename}")
        
    # Run portfolio simulation
    print("\nRunning portfolio simulation...")
    results, weights = simulate_portfolios(all_returns.dropna(), num_simulations)
    
    # Find optimal portfolios
    max_sharpe_idx = np.argmax(results[:, 2])  # Maximum Sharpe ratio
    min_vol_idx = np.argmin(results[:, 1])     # Minimum volatility
    
    # Save simulation results
    simulation_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe'])
    simulation_df['Portfolio'] = range(len(simulation_df))
    weights_df = pd.DataFrame(weights, columns=tickers)
    
    # Save optimal portfolio weights
    optimal_portfolios = pd.DataFrame({
        'Portfolio': ['Max Sharpe', 'Min Volatility'],
        'Return': [results[max_sharpe_idx, 0], results[min_vol_idx, 0]],
        'Volatility': [results[max_sharpe_idx, 1], results[min_vol_idx, 1]],
        'Sharpe': [results[max_sharpe_idx, 2], results[min_vol_idx, 2]]
    })
    
    for i, ticker in enumerate(tickers):
        optimal_portfolios[ticker] = [weights[max_sharpe_idx, i], weights[min_vol_idx, i]]
    
    # Save all results
    simulation_df.to_csv(os.path.join(output_dir, 'simulation_results.csv'), index=False)
    weights_df.to_csv(os.path.join(output_dir, 'portfolio_weights.csv'), index=False)
    optimal_portfolios.to_csv(os.path.join(output_dir, 'optimal_portfolios.csv'), index=False)
    
    # Plot efficient frontier
    plt.figure(figsize=(12, 8))
    plt.scatter(results[:, 1], results[:, 0], c=results[:, 2], cmap='viridis', alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(results[max_sharpe_idx, 1], results[max_sharpe_idx, 0], 
                color='red', marker='*', s=200, label='Maximum Sharpe ratio')
    plt.scatter(results[min_vol_idx, 1], results[min_vol_idx, 0], 
                color='green', marker='*', s=200, label='Minimum volatility')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'efficient_frontier.png'))
    plt.close()
    
    return simulation_df, optimal_portfolios

# Example usage
tickers = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'EGO', 'STM']
# tickers = ['AAPL', 'EGO', 'FSMD', 'MGRM', 'RIVN']
simulation_results, optimal_portfolios = analyze_stocks(tickers, num_simulations=1000)

# Print optimal portfolio allocations
print("\nOptimal Portfolio Allocations:")
print(optimal_portfolios)