import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

class PortfolioAnalyzer:
    def __init__(self, tickers, period='1y', output_dir='stock_data'):
        self.tickers = tickers
        self.period = period
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def fetch_stock_data(self, ticker):
        """
        Fetch stock data from Yahoo Finance
        """
        stock = yf.Ticker(ticker)
        hist = stock.history(period=self.period)
        return hist

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

    def plot_efficient_frontier(self, results, max_return_idx, min_vol_idx):
        """
        Plot the efficient frontier with optimal portfolios
        """
        plt.figure(figsize=(12, 8))
        plt.scatter(results[:, 1], results[:, 0], c=results[:, 2], cmap='viridis', alpha=0.5)
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(results[max_return_idx, 1], results[max_return_idx, 0], 
                    color='red', marker='*', s=200, label='Maximum Return')
        plt.scatter(results[min_vol_idx, 1], results[min_vol_idx, 0], 
                    color='green', marker='*', s=200, label='Minimum volatility')
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.title('Portfolio Risk-Return Analysis')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'efficient_frontier.png'))
        plt.close()

    def print_portfolio_allocation(self, portfolio_weights, portfolio_type):
        """
        Print portfolio allocation in a formatted two-column layout
        """
        print(f"\n{portfolio_type} Portfolio Allocation:")
        print("-" * 40)
        print(f"{'Stock':<10} {'Allocation':<15}")
        print("-" * 40)
        for ticker, weight in portfolio_weights.items():
            if ticker not in ['Portfolio', 'Return', 'Volatility', 'Sharpe']:
                print(f"{ticker:<10} {weight*100:>8.2f}%")
        print("-" * 40)
        print(f"Expected Annual Return: {portfolio_weights['Return']*100:.2f}%")
        print(f"Portfolio Volatility: {portfolio_weights['Volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {portfolio_weights['Sharpe']:.2f}")

    def analyze(self, num_simulations=1000):
        """
        Main analysis method that orchestrates the entire analysis process
        """
        all_metrics = []
        all_returns = pd.DataFrame()
        
        # Fetch and analyze individual stocks
        for ticker in self.tickers:
            print(f"\nAnalyzing {ticker}...")
            stock_data = self.fetch_stock_data(ticker)
            metrics, daily_returns = self.calculate_metrics(stock_data, ticker)
            
            all_metrics.append(metrics)
            all_returns[ticker] = daily_returns
            
            stock_data_filename = os.path.join(self.output_dir, f"{ticker}_raw_data.csv")
            stock_data.to_csv(stock_data_filename)
            print(f"Raw data saved to: {stock_data_filename}")
        
        # Run portfolio simulation
        print("\nRunning portfolio simulation...")
        results, weights = self.simulate_portfolios(all_returns.dropna(), num_simulations)
        
        # Find optimal portfolios
        max_return_idx = np.argmax(results[:, 0])  # Changed from Sharpe to Return
        min_vol_idx = np.argmin(results[:, 1])
        
        # Generate and save reports
        simulation_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe'])
        simulation_df['Portfolio'] = range(len(simulation_df))
        weights_df = pd.DataFrame(weights, columns=self.tickers)
        
        optimal_portfolios = pd.DataFrame({
            'Portfolio': ['Max Return', 'Min Volatility'],
            'Return': [results[max_return_idx, 0], results[min_vol_idx, 0]],
            'Volatility': [results[max_return_idx, 1], results[min_vol_idx, 1]],
            'Sharpe': [results[max_return_idx, 2], results[min_vol_idx, 2]]
        })
        
        for i, ticker in enumerate(self.tickers):
            optimal_portfolios[ticker] = [weights[max_return_idx, i], weights[min_vol_idx, i]]
        
        # Save results
        simulation_df.to_csv(os.path.join(self.output_dir, 'simulation_results.csv'), index=False)
        weights_df.to_csv(os.path.join(self.output_dir, 'portfolio_weights.csv'), index=False)
        optimal_portfolios.to_csv(os.path.join(self.output_dir, 'optimal_portfolios.csv'), index=False)
        
        # Plot efficient frontier
        self.plot_efficient_frontier(results, max_return_idx, min_vol_idx)
        
        return simulation_df, optimal_portfolios

def main():
    # Example usage
    tickers = ['AAPL', 'TSLA', 'AMD', 'NVDA', 'BAC', 'F', 'PLTR', 'INTC', 'AMZN', 'META', 'NIO', 'SOFI', 'CCL', 'AAL', 'VALE', 'MSFT', 'PFE', 'GOOGL', 'KO', 'T', 'WBD', 'SNAP', 'UBER', 'GM', 'XOM', 'PYPL', 'VZ', 'LCID', 'HOOD', 'WFC', 'BABA', 'DIS', 'JPM', 'RIVN', 'C', 'GOOG', 'PBR', 'CSCO', 'NFLX', 'SHOP', 'JD', 'NOK', 'NU', 'DNA', 'COIN', 'PLUG', 'LYFT', 'PINS', 'LUMN', 'EGO']
    analyzer = PortfolioAnalyzer(tickers)
    simulation_results, optimal_portfolios = analyzer.analyze(num_simulations=20000)
    
    # Print only the highest return portfolio
    max_return_portfolio = optimal_portfolios.iloc[0]
    analyzer.print_portfolio_allocation(max_return_portfolio, "Maximum Return")

if __name__ == "__main__":
    main()