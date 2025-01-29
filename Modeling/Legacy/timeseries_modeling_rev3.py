import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly
from itertools import product
from sklearn.metrics import mean_squared_error
import csv
from tqdm import tqdm

class StockForecaster:
    def __init__(self, csv_path):
        """Initialize the StockForecaster with data path and default parameters."""
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.forecast = None
        self.hyperparameter_test_results = []
        self.current_best_mse = float('inf')  # Initialize with infinity
        
        # Define hyperparameter search space
        self.param_grid = {
            'changepoint_prior_scale': [0.25, 0.5, 0.6, 0.75, 0.9],
            'seasonality_prior_scale': [0.01, 1.0, 5],
            'holidays_prior_scale': [0.01, 5.0, 10.0],
            'seasonality_mode': ['multiplicative'],
            'interval_width': [0.6, 0.7, 0.8, 0.9, 0.95],
            'growth': ['linear'],
            'n_changepoints': [5, 25, 50],
            'daily_seasonality': [True],
            'weekly_seasonality': [True],
            'yearly_seasonality': [True]
        }

    def prepare_data(self):
        """Read and prepare the stock data for Prophet."""
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.rename(columns={'Date': 'ds', 'Price': 'y'})
        self.df['ds'] = pd.to_datetime(self.df['ds'])
        
        # If using logistic growth, we need to set cap and floor
        self.df['cap'] = self.df['y'].max() * 1.5
        self.df['floor'] = self.df['y'].min() * 0.5
        
        return self.df

    def calculate_mse(self, forecast_df):
        """Calculate Mean Squared Error for the forecast."""
        try:
            # Print initial dataframe information
            print("\nCalculate MSE Debug Info:")
            print("Forecast DF Head:")
            print(forecast_df.head())
            print("\nOriginal DF Head:")
            print(self.df.head())
            
            # Check for required columns
            print("\nChecking columns...")
            forecast_cols = forecast_df.columns.tolist()
            df_cols = self.df.columns.tolist()
            print(f"Forecast columns: {forecast_cols}")
            print(f"Original columns: {df_cols}")
            
            # Perform merge
            print("\nPerforming merge...")
            comparison_df = forecast_df.merge(
                self.df[['ds', 'y']], 
                on='ds', 
                how='inner'
            )
            
            print("\nComparison DF Head:")
            print(comparison_df.head())
            print(f"Comparison DF Shape: {comparison_df.shape}")
            
            # Calculate MSE
            print("\nCalculating MSE...")
            mse = mean_squared_error(
                comparison_df['y'],
                comparison_df['yhat']
            )
            
            print(f"MSE calculated successfully: {mse}")
            return mse
            
        except Exception as e:
            print(f"\nError in calculate_mse: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return None

    def plot_model_performance(self, mse, params):
        """Create a plot showing the model's forecast with the current best MSE."""
        if self.model is None or self.forecast is None:
            return
        
        # Create the main forecast plot
        fig = plot_plotly(self.model, self.forecast, xlabel='Date', ylabel='Stock Price ($)')
        
        # Update layout with MSE information
        fig.update_layout(
            title=f'Stock Price Forecast (MSE: {mse:.4f})',
            showlegend=True,
            annotations=[
                dict(
                    text=f"MSE: {mse:.4f}\nKey Parameters:\n" + \
                         "\n".join([f"{k}: {v}" for k, v in params.items() if k in ['changepoint_prior_scale', 'seasonality_mode', 'growth']]),
                    xref="paper",
                    yref="paper",
                    x=1,
                    y=0,
                    showarrow=False,
                    align="right",
                    bgcolor="rgba(255, 255, 255, 0.8)"
                )
            ]
        )
        
        fig.show()

    def test_hyperparameters(self, baseline_mse, output_file='hyperparameter_results.csv'):
        """Test different hyperparameter combinations and log results that beat the baseline."""
        if self.df is None:
            self.prepare_data()
            print("Data prepared successfully")

        self.current_best_mse = baseline_mse
        successful_improvements = 0
        total_tests = 0
        
        # Create CSV file and write header
        fieldnames = list(self.param_grid.keys()) + ['mse']
        
        try:
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Generate all combinations of hyperparameters
                param_combinations = [dict(zip(self.param_grid.keys(), v)) 
                                   for v in product(*self.param_grid.values())]
                
                print(f"\nStarting testing with baseline MSE: {baseline_mse}")
                print(f"Testing {len(param_combinations)} parameter combinations...")
                
                # Test each combination
                for params in tqdm(param_combinations, desc="Testing hyperparameters"):
                    total_tests += 1
                    try:
                        # Train model with current parameters
                        self.model = Prophet(**params)
                        self.model.fit(self.df)
                        
                        # Make predictions for the training period
                        future = self.model.make_future_dataframe(periods=7)
                        if params['growth'] == 'logistic':
                            future['cap'] = self.df['cap'].max()
                            future['floor'] = self.df['floor'].min()
                        
                        forecast = self.model.predict(future)
                        self.forecast = forecast  # Store for plotting
                        
                        # Calculate MSE
                        mse = self.calculate_mse(forecast)
                        
                        # Only proceed if MSE calculation was successful and beats current best
                        if mse is not None and mse < self.current_best_mse:
                            # Add MSE to parameters dictionary
                            result_dict = params.copy()
                            result_dict['mse'] = mse
                            
                            # Write results to CSV
                            writer.writerow(result_dict)
                            
                            # Store results in instance variable
                            self.hyperparameter_test_results.append(result_dict)
                            successful_improvements += 1
                            
                            print(f"\nNew best MSE found: {mse:.4f} (previous: {self.current_best_mse:.4f})")
                            self.current_best_mse = mse
                            
                            # Create plot for the new best model
                            self.plot_model_performance(mse, params)
                        
                    except Exception as e:
                        print(f"\nError with parameters {params}:")
                        print(f"Error details: {str(e)}")
                        continue

                print(f"\nTesting completed:")
                print(f"Total tests: {total_tests}")
                print(f"Improvements found: {successful_improvements}")
                print(f"Final best MSE: {self.current_best_mse:.4f}")
                print(f"Results saved to: {output_file}")

        except Exception as e:
            print(f"Error in file operations: {str(e)}")

        if len(self.hyperparameter_test_results) == 0:
            raise ValueError("No improvements over baseline were found.")

        return self.hyperparameter_test_results

    def get_best_parameters(self):
        """Return the hyperparameters that produced the lowest MSE."""
        if not self.hyperparameter_test_results:
            raise ValueError("No hyperparameter test results available.")
        
        best_result = min(self.hyperparameter_test_results, 
                         key=lambda x: x['mse'])
        return best_result

    def train_model(self, **kwargs):
        """Train the Prophet model with given parameters."""
        if self.df is None:
            self.prepare_data()
        
        self.model = Prophet(**kwargs)
        self.model.fit(self.df)
        return self.model

    def make_predictions(self, periods=3):
        """Generate predictions for the specified number of periods."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Make future dataframe
        future = self.model.make_future_dataframe(periods=periods)
        
        # Add cap and floor if using logistic growth
        if self.model.growth == 'logistic':
            future['cap'] = self.df['cap'].max()
            future['floor'] = self.df['floor'].min()
        
        # Generate forecast
        self.forecast = self.model.predict(future)
        return self.forecast

    def print_forecast_results(self, periods=3):
        """Print formatted forecast results."""
        if self.forecast is None:
            raise ValueError("No forecast available. Call make_predictions() first.")
        
        forecast_subset = self.forecast.tail(periods)
        
        print("\nForecast for the next {periods} days:")
        print("Date                      Predicted Price    Lower Bound    Upper Bound")
        print("-" * 75)
        
        for _, row in forecast_subset.iterrows():
            date = row['ds'].strftime('%Y-%m-%d')
            yhat = row['yhat']
            yhat_lower = row['yhat_lower']
            yhat_upper = row['yhat_upper']
            print(f"{date}           ${yhat:.2f}          ${yhat_lower:.2f}        ${yhat_upper:.2f}")

    def create_interactive_plots(self):
        """Generate and display interactive plots for the forecast."""
        if self.forecast is None or self.model is None:
            raise ValueError("No forecast available. Call make_predictions() first.")

        # Main forecast plot
        fig_forecast = plot_plotly(self.model, self.forecast, 
                                xlabel='Date', ylabel='Stock Price ($)')
        fig_forecast.update_layout(
            title='Stock Price Forecast',
            showlegend=True
        )
        fig_forecast.show()

        # Components plot
        fig_components = plot_components_plotly(self.model, self.forecast)
        fig_components.show()

        # Price change distribution plot
        returns = self.df['y'].pct_change().dropna()
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name='Daily Returns',
                showlegend=True
            )
        )
        fig_dist.update_layout(
            title='Distribution of Daily Price Changes',
            xaxis_title='Daily Return',
            yaxis_title='Frequency',
            bargap=0.1
        )
        fig_dist.show()

    def run_full_analysis(self, use_best_params=False):
        """Run the complete analysis pipeline."""
        try:
            if self.df is None:
                self.prepare_data()
            
            if use_best_params and self.hyperparameter_test_results:
                best_params = self.get_best_parameters()
                print("Using best parameters found:", best_params)
                # Remove MSE from parameters before training
                params = best_params.copy()
                params.pop('mse', None)
                
                # Initialize and train model with best parameters
                self.model = Prophet(**params)
                self.model.fit(self.df)
            else:
                # Use default parameters
                self.model = Prophet()
                self.model.fit(self.df)
                
            self.make_predictions()
            self.print_forecast_results()
            self.create_interactive_plots()
            return self.model, self.forecast
            
        except Exception as e:
            print(f"Error in run_full_analysis: {str(e)}")
            raise

def main():
    """Main entry point of the program."""
    try:
        # Initialize forecaster with your data path
        forecaster = StockForecaster('C:/Users/jpetty/Documents/Projects/dsWithDen/Data/stock_data/MSFT_prices_20250117.csv')
        
        # Load and print data sample
        forecaster.prepare_data()
        print("\nInitial data sample:")
        print(forecaster.df.head())
        print("\nData shape:", forecaster.df.shape)
        print("\nData columns:", forecaster.df.columns.tolist())

        # Set baseline MSE
        baseline_mse = 15  # Set your initial baseline MSE here
        print(f"\nStarting with baseline MSE: {baseline_mse}")

        # Continue with test...
        print("\nTesting hyperparameters...")
        forecaster.test_hyperparameters(baseline_mse, 'hyperparameter_results.csv')
        
        # Get and print best parameters
        best_params = forecaster.get_best_parameters()
        print("\nBest hyperparameters found:")
        print(f"Parameters: {best_params}")
        
        # Run full analysis with best parameters
        print("\nRunning full analysis with best parameters...")
        model, forecast = forecaster.run_full_analysis(use_best_params=True)
        
        return 0  # Success
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1  # Error

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)