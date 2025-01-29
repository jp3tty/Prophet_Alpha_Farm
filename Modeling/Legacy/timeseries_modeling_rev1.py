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
        
        # Define hyperparameter search space
        self.param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 1.0, 10.0, 50.0, 100.0],
            'holidays_prior_scale': [0.01, 1.0, 10.0, 50.0, 100.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'interval_width': [0.6, 0.7, 0.8, 0.9, 0.95],
            'growth': ['linear', 'logistic'],
            'n_changepoints': [10, 20, 25, 35, 50],
            'daily_seasonality': [True, False],
            'weekly_seasonality': [True, False],
            'yearly_seasonality': [True, False]
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

    def test_hyperparameters(self, output_file='hyperparameter_results.csv'):
        """Test different hyperparameter combinations and log results."""
        if self.df is None:
            self.prepare_data()
            print("Data prepared successfully")

        successful_tests = 0
        failed_tests = 0
        
        # Create CSV file and write header
        fieldnames = list(self.param_grid.keys()) + ['mse']
        
        try:
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Generate all combinations of hyperparameters
                param_combinations = [dict(zip(self.param_grid.keys(), v)) 
                                for v in product(*self.param_grid.values())]
                
                print(f"\nStarting testing of {len(param_combinations)} parameter combinations...")
                
                # Test each combination
                for params in tqdm(param_combinations, desc="Testing hyperparameters"):
                    try:
                        # Train model with current parameters
                        self.model = Prophet(**params)
                        self.model.fit(self.df)
                        
                        # Make predictions for the training period
                        future = self.model.make_future_dataframe(periods=0)
                        if params['growth'] == 'logistic':
                            future['cap'] = self.df['cap'].max()
                            future['floor'] = self.df['floor'].min()
                        
                        forecast = self.model.predict(future)
                        
                        # Calculate MSE
                        mse = self.calculate_mse(forecast)
                        
                        # Only proceed if MSE calculation was successful
                        if mse is not None:
                            # Add MSE to parameters dictionary
                            result_dict = params.copy()
                            result_dict['mse'] = mse
                            
                            # Write results to CSV
                            writer.writerow(result_dict)
                            
                            # Store results in instance variable
                            self.hyperparameter_test_results.append(result_dict)
                            successful_tests += 1
                            print(f"Successful test {successful_tests}: MSE = {mse}")
                        
                    except Exception as e:
                        failed_tests += 1
                        print(f"\nError with parameters {params}:")
                        print(f"Error details: {str(e)}")
                        continue

                print(f"\nTesting completed:")
                print(f"Successful tests: {successful_tests}")
                print(f"Failed tests: {failed_tests}")
                print(f"Results saved to: {output_file}")

        except Exception as e:
            print(f"Error in file operations: {str(e)}")

        if len(self.hyperparameter_test_results) == 0:
            raise ValueError("No successful hyperparameter combinations were found.")

        return self.hyperparameter_test_results
    
    def get_best_parameters(self):
        """Return the hyperparameters that produced the lowest MSE."""
        if not self.hyperparameter_test_results:
            raise ValueError("No hyperparameter test results available.")
        
        best_result = min(self.hyperparameter_test_results, 
                         key=lambda x: x['mse'])
        return best_result

    def run_full_analysis(self, use_best_params=False):
        """Run the complete analysis pipeline."""
        self.prepare_data()
        
        if use_best_params and self.hyperparameter_test_results:
            best_params = self.get_best_parameters()
            print("Using best parameters found:", best_params)
            # Remove MSE from parameters before training
            best_params.pop('mse', None)
            self.train_model(**best_params)
        else:
            self.train_model()
            
        self.make_predictions()
        self.print_forecast_results()
        self.create_interactive_plots()
        return self.model, self.forecast

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

        # Continue with test...
        print("\nTesting hyperparameters...")
        forecaster.test_hyperparameters('hyperparameter_results_rev1.csv')
        
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