"""
Main Execution Module for Time Series Analysis

Orchestrates the execution of multiple time series models and generates comprehensive
comparison reports.

Key Features:
- Parallel model execution
- Comprehensive model comparison
- Detailed performance metrics
- Visualization generation
- CSV report generation

Metrics:
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- Last Known Price
- Forecast Mean
- Forecast Standard Deviation
"""

from prophet_timeseries import ProphetTimeSeriesModel
from sarima_timeseries import SarimaTimeSeriesModel
from theta_timeseries import ThetaTimeSeriesModel
from visualizations import TimeSeriesPlotter
import os
import pandas as pd
import numpy as np
from datetime import datetime
import concurrent.futures

class StockModelComparison:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create subdirectories for each model type
        for model_name in ['Prophet', 'SARIMA', 'Theta']:
            model_dir = os.path.join(output_dir, model_name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

    def get_stock_files(self):
        """Get all CSV files from the data directory."""
        return [f for f in os.listdir(self.data_dir) if f.endswith('_prices.csv')]

    def run_analysis(self):
        """Run analysis for all stock files."""
        stock_files = self.get_stock_files()
        print(f"Found {len(stock_files)} stock datasets to analyze")

        # Create summary DataFrame
        summary_data = []

        for stock_file in stock_files:
            stock_symbol = stock_file.split('_')[0]
            csv_path = os.path.join(self.data_dir, stock_file)
            print(f"\nAnalyzing {stock_symbol}...")

            try:
                # Run each model
                stock_results = self.analyze_stock(csv_path, stock_symbol)
                
                # Add results to summary
                for model_name, metrics in stock_results.items():
                    summary_data.append({
                        'Stock': stock_symbol,
                        'Model': model_name,
                        'MSE': metrics['mse'],
                        'RMSE': metrics['rmse'],
                        'Last Price': metrics['last_price'],
                        'Forecast Mean': metrics['forecast_mean'],
                        'Forecast Std': metrics['forecast_std']
                    })

                self.results[stock_symbol] = stock_results

            except Exception as e:
                print(f"Error analyzing {stock_symbol}: {str(e)}")
                continue

        # Create and save summary report
        self.create_summary_report(summary_data)

    def analyze_stock(self, csv_path, stock_symbol):
        """Run all models for a single stock."""
        stock_results = {}
        
        # Initialize models
        models = {
            'Prophet': ProphetTimeSeriesModel(csv_path),
            'SARIMA': SarimaTimeSeriesModel(csv_path),
            'Theta': ThetaTimeSeriesModel(csv_path)
        }

        for model_name, model in models.items():
            print(f"Running {model_name} model for {stock_symbol}...")
            try:
                # Set model-specific output directory
                model.output_dir = os.path.join(self.output_dir, model_name)
                
                # Create unified plotter
                plotter = TimeSeriesPlotter(output_dir=model.output_dir)
                
                # Run model
                model.prepare_data()
                model.train_model()
                forecast = model.make_predictions(periods=5)
                
                # Calculate MSE differently based on model type
                if model_name == 'Theta':
                    values = model.df['y'].values
                    if hasattr(model, 'best_params') and model.best_params is not None:
                        mse, rmse = model.calculate_mse(values, model.best_params)
                    else:
                        # Fallback if best_params isn't available
                        mse, rmse = float('inf'), float('inf')
                        print(f"Warning: No best_params found for Theta model on {stock_symbol}")
                else:
                    mse, rmse = model.calculate_mse(forecast)
                
                # Store results
                stock_results[model_name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'last_price': model.df['y'].iloc[-1],
                    'forecast_mean': forecast['yhat'].mean(),
                    'forecast_std': forecast['yhat'].std()
                }
                
                # Generate and save plots using the unified plotter
                plotter.plot_forecast(model.df, forecast, stock_symbol, model_name, mse)
                plotter.plot_focused_forecast(model.df, forecast, stock_symbol, model_name)
                plotter.plot_distribution(model.df, stock_symbol, model_name)
                
                print(f"Completed {model_name} model for {stock_symbol} with MSE: {mse:.4f}, RMSE: {rmse:.4f}")

            except Exception as e:
                print(f"Error in {model_name} model for {stock_symbol}: {str(e)}")
                continue

        return stock_results

    def create_summary_report(self, summary_data):
        """Create and save summary report."""
        # Check if summary_data is empty
        if not summary_data:
            print("No data to create summary report")
            return
            
        # Convert to DataFrame
        df_summary = pd.DataFrame(summary_data)
        
        # Sort by Stock and MSE
        df_summary = df_summary.sort_values(['Stock', 'MSE'])
        
        # Save summary to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.output_dir, f'model_comparison_summary_{timestamp}.csv')
        df_summary.to_csv(summary_path, index=False)
        
        # Print summary
        print("\nAnalysis Summary:")
        print("=" * 80)
        for stock in df_summary['Stock'].unique():
            stock_data = df_summary[df_summary['Stock'] == stock]
            print(f"\nResults for {stock}:")
            print("-" * 40)
            for _, row in stock_data.iterrows():
                print(f"{row['Model']}:")
                print(f"  MSE: {row['MSE']:.2f}")
                print(f"  RMSE: {row['RMSE']:.2f}")
                print(f"  Forecast Mean: ${row['Forecast Mean']:.2f}")
            print(f"Last Known Price: ${stock_data.iloc[0]['Last Price']:.2f}")

def main():
    # Set up paths
    data_dir = "Data"  # Directory containing price_collection.py output
    
    # Create dated output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("Output", timestamp)  # Directory for model outputs with timestamp
    
    # Initialize and run comparison
    comparison = StockModelComparison(data_dir, output_dir)
    comparison.run_analysis()

if __name__ == "__main__":
    main()