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
import os
import pickle
import uuid

class StockForecaster:
    def __init__(self, csv_path):
        """Initialize the StockForecaster with data path and default parameters."""
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.forecast = None
        self.hyperparameter_test_results = []
        self.current_best_mse = float('inf')
        self.model_counter = 0
        
        # Extract stock name from csv path
        self.stock_name = os.path.splitext(os.path.basename(csv_path))[0].split('_')[0]
        
        # Use output directory passed from main() instead of creating new one
        self.output_dir = None  # Will be set when running the script
        
        # Rest of initialization remains the same
        self.param_grid = {
            'changepoint_prior_scale': [0.5, 1.2, 0.6, 0.75, 0.9, 1, 1.2],
            'seasonality_prior_scale': [2, 7, 3, 5, 6, 7],
            'holidays_prior_scale': [0.01],
            'seasonality_mode': ['multiplicative'],
            'interval_width': [0.05, 0.1, 0.15],
            'growth': ['linear'],
            'n_changepoints': [35, 70, 40, 45, 50, 55, 60, 65, 70],
            'daily_seasonality': [True],
            'weekly_seasonality': [False],
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
                # Perform merge
                comparison_df = forecast_df.merge(
                    self.df[['ds', 'y']], 
                    on='ds', 
                    how='inner'
                )
                
                # Calculate MSE
                mse = mean_squared_error(
                    comparison_df['y'],
                    comparison_df['yhat']
                )
                                
                print(f"MSE calculated successfully: {mse:.2f}%")
                return mse
                
            except Exception as e:
                print(f"\nError in calculate_mse: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                return None

    def _get_model_filename(self):
        """Generate a consistent filename for model files."""
        current_date = datetime.now().strftime("%Y%m%d")
        return f"{self.stock_name}_Champion_{current_date}"

    def save_model_plot(self, model_id, mse, params):
        """Save the model's forecast plot as PNG."""
        if self.model is None or self.forecast is None:
            return
        
        # Get filename using current date
        filename = self._get_model_filename()
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create key parameters string
        key_params = ", ".join([f"{k}: {v}" for k, v in params.items() 
                              if k in ['changepoint_prior_scale', 'seasonality_mode', 'growth']])
        
        # Create the plot using Prophet's plot_plotly but only get the first figure
        fig = plot_plotly(self.model, self.forecast, xlabel='Date', ylabel='Stock Price ($)')
        
        # Keep only the first trace group (main forecast plot)
        fig.data = fig.data[:4]  # This keeps only the main forecast traces
        
        # Update layout with stock name, date, and parameters in title
        fig.update_layout(
            title={
                'text': f'<span style="font-size: 24px">{self.stock_name} Stock Price Forecast - {current_date}</span><br>' +
                       f'<span style="font-size: 12px">Champion Model (MSE: {mse:.4f})</span><br>', # +
                    #    f'<span style="font-size: 8px">Parameters: {key_params}</span>',
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            showlegend=True
        )
            
        # Save new champion plot
        plot_path = os.path.join(self.output_dir, f'{filename}.png')
        fig.write_image(plot_path)
        print(f"Champion plot saved: {plot_path}")

    def save_model(self, model_id):
        """Save the Prophet model to disk."""
        if self.model is None:
            return
        
        # Use same filename as plot
        filename = self._get_model_filename()
        
        model_path = os.path.join(self.output_dir, f'{filename}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved: {model_path}")
        
        # Remove any existing champion files for this stock AFTER saving both new files
        for ext in ['.png', '.pkl']:
            existing_files = [f for f in os.listdir(self.output_dir) 
                            if f.startswith(f"{self.stock_name}_Champion_") 
                            and f.endswith(ext)
                            and not f.endswith(f"_{datetime.now().strftime('%Y%m%d')}{ext}")]  # Don't delete the files we just created
            for old_file in existing_files:
                os.remove(os.path.join(self.output_dir, old_file))

    def test_hyperparameters(self, baseline_mse, output_file='hyperparameter_results.csv'):
        """Test different hyperparameter combinations and log results that beat the baseline."""
        if self.df is None:
            self.prepare_data()
            print("Data prepared successfully")

        self.current_best_mse = baseline_mse
        successful_improvements = 0
        total_tests = 0
        
        # Create CSV file and write header
        fieldnames = ['model_id'] + list(self.param_grid.keys()) + ['mse']
        
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
                            # Generate unique model ID
                            model_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
                            
                            # Add model_id and MSE to parameters dictionary
                            result_dict = {'model_id': model_id}
                            result_dict.update(params)
                            result_dict['mse'] = mse
                            
                            # Write results to CSV
                            writer.writerow(result_dict)
                            
                            # Store results in instance variable
                            self.hyperparameter_test_results.append(result_dict)
                            successful_improvements += 1
                            
                            print(f"\nNew best MSE found: {mse:.4f} (previous: {self.current_best_mse:.4f})")
                            self.current_best_mse = mse
                            
                            # Save the model and its plot
                            self.save_model_plot(model_id, mse, params)
                            self.save_model(model_id)
                        
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

    def make_predictions(self, periods=5):  # Changed default to 5
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

    def print_forecast_results(self, periods=5):  # Changed default to 5
        """Print formatted forecast results."""
        if self.forecast is None:
            raise ValueError("No forecast available. Call make_predictions() first.")
        
        forecast_subset = self.forecast.tail(periods)
        
        print(f"\nForecast for the next {periods} days:")
        print("Date                      Predicted Price    Lower Bound    Upper Bound")
        print("-" * 75)
        
        for _, row in forecast_subset.iterrows():
            date = row['ds'].strftime('%Y-%m-%d')
            yhat = row['yhat']
            yhat_lower = row['yhat_lower']
            yhat_upper = row['yhat_upper']
            print(f"{date}           ${yhat:.2f}              ${yhat_lower:.2f}          ${yhat_upper:.2f}")

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
                # Remove model_id and MSE from parameters before training
                params = best_params.copy()
                params.pop('mse', None)
                params.pop('model_id', None)
                
                # Initialize and train model with best parameters
                self.model = Prophet(**params)
                self.model.fit(self.df)

            else:
                # Use default parameters
                self.model = Prophet()
                self.model.fit(self.df)
                
            # Changed to explicitly specify 5 periods
            self.make_predictions(periods=5)
            self.print_forecast_results(periods=5)
            self.create_interactive_plots()
            return self.model, self.forecast
            
        except Exception as e:
            print(f"Error in run_full_analysis: {str(e)}")
            raise

def main():
    """Main entry point of the program."""
    try:
        # Create single timestamped directory for this run
        output_base = 'forecast_output'
        if not os.path.exists(output_base):
            os.makedirs(output_base)
        
        timestamp = datetime.now().strftime("%m_%d_%Y_%H%M")
        output_dir = os.path.join(output_base, timestamp)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get all CSV files in the price_data directory
        price_data_dir = '../Data/price_data'
        csv_files = [f for f in os.listdir(price_data_dir) if f.endswith('.csv')]
        
        # Initialize list to store all results
        all_results = []
        
        print(f"\nFound {len(csv_files)} CSV files to process")
        
        # Process each CSV file
        for csv_file in csv_files:
            print(f"\nProcessing {csv_file}...")
            
            # Initialize forecaster for current file
            file_path = os.path.join(price_data_dir, csv_file)
            forecaster = StockForecaster(file_path)
            forecaster.output_dir = output_dir  # Set the output directory
            
            # Load and print data sample
            forecaster.prepare_data()
            print("\nData shape:", forecaster.df.shape)
            
            # Set baseline MSE
            baseline_mse = 15  # Set your initial baseline MSE here
            print(f"\nStarting with baseline MSE: {baseline_mse}")
            
            try:
                # Test hyperparameters
                print("\nTesting hyperparameters...")
                results = forecaster.test_hyperparameters(baseline_mse)
                
                # Get best parameters
                best_params = forecaster.get_best_parameters()
                
                # Add stock name to results
                stock_name = os.path.splitext(csv_file)[0].split('_')[0]
                best_params['stock'] = stock_name
                
                # Add to all results
                all_results.append(best_params)
                
                # Run full analysis with best parameters
                print("\nRunning full analysis with best parameters...")
                forecaster.run_full_analysis(use_best_params=True)
                
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")
                continue
        
        # Save consolidated results in the same timestamped directory
        if all_results:
            results_filename = 'forecast_results.csv'
            results_path = os.path.join(output_dir, results_filename)
            
            # Convert results to DataFrame and save
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(results_path, index=False)
            print(f"\nConsolidated results saved to: {results_path}")
            
        return 0  # Success
    
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        return 1  # Error
    

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)