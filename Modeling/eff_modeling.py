import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import random
import json

class PortfolioAnalyzer:
    def __init__(self, all_tickers, output_dir='stock_data'):
        self.all_tickers = all_tickers
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_random_portfolio(self, size):
        """
        Generate a random portfolio of specified size from available tickers
        """
        return random.sample(self.all_tickers, size)
        
    def fetch_stock_data(self, ticker, period='2y'):
        """
        Fetch stock data from Yahoo Finance
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def calculate_metrics(self, data, ticker):
        """
        Calculate key metrics for portfolio analysis
        """
        daily_returns = data['Close'].pct_change()
        
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

    def calculate_portfolio_metrics(self, weights, returns):
        """
        Calculate portfolio return and volatility
        """
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_vol
        return portfolio_return, portfolio_vol, sharpe_ratio

    def simulate_portfolios(self, returns, num_portfolios=1000):
        """
        Run Monte Carlo simulation to generate random portfolios
        """
        num_assets = returns.shape[1]
        results = np.zeros((num_portfolios, 3))
        all_weights = np.zeros((num_portfolios, num_assets))
        
        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights = weights / np.sum(weights)
            
            portfolio_return, portfolio_vol, sharpe_ratio = self.calculate_portfolio_metrics(weights, returns)
            
            results[i] = [portfolio_return, portfolio_vol, sharpe_ratio]
            all_weights[i] = weights
        
        return results, all_weights

    def run_single_simulation(self, portfolio_size, simulation_id):
        """
        Run a single portfolio simulation with random stock selection
        """
        # Generate random portfolio
        selected_tickers = self.generate_random_portfolio(portfolio_size)
        
        # Initialize DataFrames
        all_returns = pd.DataFrame()
        
        # Fetch data for selected tickers
        valid_tickers = []
        for ticker in selected_tickers:
            stock_data = self.fetch_stock_data(ticker)
            if stock_data is not None and not stock_data.empty:
                _, daily_returns = self.calculate_metrics(stock_data, ticker)
                all_returns[ticker] = daily_returns
                valid_tickers.append(ticker)
        
        # Run portfolio simulation
        results, weights = self.simulate_portfolios(all_returns.dropna(), num_portfolios=1000)
        
        # Find highest return portfolio
        max_return_idx = np.argmax(results[:, 0])
        
        # Create portfolio dictionary
        portfolio_dict = {ticker: float(weight) for ticker, weight in zip(valid_tickers, weights[max_return_idx])}
        
        # Create results dictionary with new format
        simulation_results = {
            'Simulation': simulation_id,
            'Portfolio Size': portfolio_size,
            'Return (%)': float(results[max_return_idx, 0] * 100),
            'Sharpe Ratio': float(results[max_return_idx, 2]),
            'Portfolio': portfolio_dict
        }
        
        return simulation_results
    
    def run_multiple_simulations(self, num_simulations=5000):
        """
        Run multiple simulations with varying portfolio sizes
        """
        all_results = []
        
        for sim_id in range(num_simulations):
            # Random portfolio size between 3 and 12
            portfolio_size = random.randint(3, 12)
            print(f"\nRunning simulation {sim_id + 1}/{num_simulations} with {portfolio_size} stocks")
            
            results = self.run_single_simulation(portfolio_size, sim_id + 1)
            all_results.append(results)
            
            # Print current simulation results
            print(f"Portfolio Return: {results['Return (%)']:.2f}%")
            print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
            print("Portfolio Allocation:")
            for ticker, weight in results['Portfolio'].items():
                print(f"{ticker}: {weight*100:.2f}%")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results
        results_df.to_csv(os.path.join(self.output_dir, 'simulation_results.csv'), index=False)
        
        # Also save as JSON for better dictionary preservation
        with open(os.path.join(self.output_dir, 'simulation_results.json'), 'w') as f:
            json.dump(all_results, f, indent=4)
        
        return results_df

def main():
    # Example list of all available tickers
    all_tickers = ['AAPL', 'TSLA', 'AMD', 'NVDA', 'BAC', 'F', 'PLTR', 'INTC', 'AMZN', 'META',
                   'NIO', 'SOFI', 'CCL', 'AAL', 'VALE', 'MSFT', 'PFE', 'GOOGL', 'KO', 'T', 'WBD',
                   'SNAP', 'UBER', 'GM', 'XOM', 'PYPL', 'VZ', 'LCID', 'HOOD', 'WFC', 'BABA',
                   'DIS', 'JPM', 'RIVN', 'C', 'GOOG', 'PBR', 'CSCO', 'NFLX', 'SHOP', 'JD', 'NOK',
                   'NU', 'DNA', 'COIN', 'PLUG', 'LYFT', 'PINS', 'LUMN', 'EGO']
    
    # Initialize analyzer
    analyzer = PortfolioAnalyzer(all_tickers)
    
    # Run multiple simulations
    results_df = analyzer.run_multiple_simulations(num_simulations=5000)
    
    # Print summary statistics
    print("\nSimulation Summary:")
    print("-" * 40)
    print(f"Average Return: {results_df['Return (%)'].mean():.2f}%")
    print(f"Best Return: {results_df['Return (%)'].max():.2f}%")
    print(f"Average Sharpe Ratio: {results_df['Sharpe Ratio'].mean():.2f}")
    print(f"Average Portfolio Size: {results_df['Portfolio Size'].mean():.1f}")

if __name__ == "__main__":
    main()