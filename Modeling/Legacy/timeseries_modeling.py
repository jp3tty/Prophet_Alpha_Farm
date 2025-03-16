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
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

        # Add model storage for comparisons
        self.models = {
            'Prophet': None,
            'SARIMA': None,
            'Theta': None
        }
        self.forecasts = {}

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
            
            # Calculate RMSE (Root Mean Squared Error) for more interpretable results
            rmse = np.sqrt(mse)
                        
            print(f"MSE calculated successfully: ${mse:.2f} (RMSE: ±${rmse:.2f})")
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

    def test_hyperparameters(self, baseline_mse):
        """Test different hyperparameter combinations and log results that beat the baseline."""
        if self.df is None:
            self.prepare_data()
            print("Data prepared successfully")

        self.current_best_mse = baseline_mse
        successful_improvements = 0
        total_tests = 0
        
        # Create stock-specific filename
        output_file = os.path.join(self.output_dir, f'{self.stock_name}_hyperparameter_results.csv')
        
        # Create master results file if it doesn't exist
        master_file = os.path.join(self.output_dir, 'all_hyperparameter_results.csv')
        fieldnames = ['stock_name', 'model_id'] + list(self.param_grid.keys()) + ['mse']
        
        if not os.path.exists(master_file):
            with open(master_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
        
        try:
            # Open stock-specific file
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
                            
                            # Add model_id, stock_name, and MSE to parameters dictionary
                            result_dict = {
                                'stock_name': self.stock_name,
                                'model_id': model_id
                            }
                            result_dict.update(params)
                            result_dict['mse'] = mse
                            
                            # Write to stock-specific file
                            writer.writerow(result_dict)
                            
                            # Append to master file
                            with open(master_file, 'a', newline='') as masterfile:
                                master_writer = csv.DictWriter(masterfile, fieldnames=fieldnames)
                                master_writer.writerow(result_dict)
                            
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

    def train_sarima_model(self, periods=5):
        """Train and forecast using SARIMA model with parameter grid search."""
        try:
            # Ensure the dates are properly ordered and continuous
            self.df = self.df.sort_values('ds')
            
            # Create DatetimeIndex with explicit frequency
            date_index = pd.DatetimeIndex(self.df['ds']).to_period('D').to_timestamp()
            
            # Preprocess data with proper index
            data = pd.Series(
                np.log(self.df['y'].values),
                index=date_index,
                name='price'
            )
            
            # Print diagnostic information
            print("\nSARIMA Data Preparation:")
            print(f"Data shape: {data.shape}")
            print(f"Index frequency: {data.index.freq}")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            
            # More conservative parameter grid
            param_grid = {
                'p': [1],     # Autoregressive order
                'd': [1],     # Differencing order
                'q': [1],     # Moving average order
                'P': [0, 1],  # Seasonal autoregressive order
                'D': [0],     # Seasonal differencing order
                'Q': [0],     # Seasonal moving average order
                's': [5]      # Seasonal period (weekly)
            }
            
            best_aic = float('inf')
            best_params = None
            best_model = None
            
            # Grid search
            for p in param_grid['p']:
                for d in param_grid['d']:
                    for q in param_grid['q']:
                        for P in param_grid['P']:
                            for D in param_grid['D']:
                                for Q in param_grid['Q']:
                                    for s in param_grid['s']:
                                        try:
                                            # Initialize and train SARIMA model
                                            model = SARIMAX(
                                                data,
                                                order=(p, d, q),
                                                seasonal_order=(P, D, Q, s),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False
                                            )
                                            results = model.fit(disp=False)
                                            
                                            current_aic = results.aic
                                            if current_aic < best_aic:
                                                best_aic = current_aic
                                                best_params = {
                                                    'p': p, 'd': d, 'q': q,
                                                    'P': P, 'D': D, 'Q': Q, 's': s
                                                }
                                                best_model = results
                                                
                                        except Exception as e:
                                            continue
            
            if best_model is None:
                raise ValueError("No valid SARIMA model found with given parameters")
            
            print(f"\nBest SARIMA parameters found: {best_params}")
            print(f"AIC: {best_aic}")
            
            # Generate future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=periods,
                freq='D'
            )
            
            # Generate forecast and transform back
            forecast = best_model.forecast(periods)
            forecast_values = np.exp(forecast)
            
            # Store results
            self.models['SARIMA'] = best_model
            self.forecasts['SARIMA'] = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast_values,
                'model': 'SARIMA'
            })
            
            print("\nSARIMA Forecast Summary:")
            print(self.forecasts['SARIMA'][['ds', 'yhat']])
            
            return self.forecasts['SARIMA']
            
        except Exception as e:
            print(f"Error in SARIMA modeling: {str(e)}")
            return None

    def train_theta_model(self, periods=5):
        """Train and forecast using Theta model with parameter tuning."""
        try:
            # Define parameter grid with more conservative values
            param_grid = {
                'theta': [0.1, 0.2, 0.3, 0.4, 0.5],  # Reduced theta values for less aggressive trends
                'window_size': [5, 7, 10, 14],        # Different smoothing windows
                'decomposition_type': ['multiplicative']  # Stick to multiplicative for stock prices
            }
            
            best_mse = float('inf')
            best_params = None
            best_forecast = None
            
            values = self.df['y'].values
            n = len(values)
            
            # Use shorter training period for better recent trend capture
            train_size = min(252, n)  # Use last year of data or all if less
            values = values[-train_size:]
            n = len(values)
            
            # Grid search
            for theta in param_grid['theta']:
                for window in param_grid['window_size']:
                    for decomp_type in param_grid['decomposition_type']:
                        try:
                            # Calculate moving averages for long and short term trends
                            long_ma = pd.Series(values).rolling(window=window*2, center=True).mean()
                            short_ma = pd.Series(values).rolling(window=window, center=True).mean()
                            
                            # Fill NaN values
                            long_ma = long_ma.fillna(method='bfill').fillna(method='ffill')
                            short_ma = short_ma.fillna(method='bfill').fillna(method='ffill')
                            
                            # Calculate trend
                            trend = long_ma + theta * (short_ma - long_ma)
                            
                            # Calculate seasonal pattern
                            if decomp_type == 'multiplicative':
                                seasonal = values / trend
                                seasonal = pd.Series(seasonal).rolling(window=window, center=True).mean()
                                seasonal = seasonal.fillna(method='bfill').fillna(method='ffill')
                            
                            # Fit final trend
                            X = np.arange(n).reshape(-1, 1)
                            trend_model = np.poly1d(np.polyfit(X.flatten(), trend, 2))  # Use quadratic fit
                            
                            # Generate forecast components
                            future_X = np.arange(n, n + periods).reshape(-1, 1)
                            trend_forecast = trend_model(future_X.flatten())
                            
                            # Calculate recent average seasonal factor
                            recent_seasonal = seasonal[-window:].mean()
                            
                            # Generate final forecast
                            forecast = trend_forecast * recent_seasonal
                            
                            # Calculate MSE on recent data
                            recent_fitted = trend_model(X[-window:].flatten()) * seasonal[-window:]
                            recent_actual = values[-window:]
                            mse = mean_squared_error(recent_actual, recent_fitted)
                            
                            if mse < best_mse:
                                best_mse = mse
                                best_params = {
                                    'theta': theta,
                                    'window_size': window,
                                    'decomposition_type': decomp_type
                                }
                                best_forecast = forecast
                                
                        except Exception as e:
                            continue
            
            print(f"\nBest Theta parameters found: {best_params}")
            print(f"MSE: {best_mse}")
            
            # Calculate RMSE for comparison
            rmse = np.sqrt(best_mse)
            print(f"RMSE: ±${rmse:.2f}")
            
            # Create dates for forecast
            dates = pd.date_range(
                start=self.df['ds'].iloc[-1] + timedelta(days=1),
                periods=periods,
                freq='D'
            )
            
            # Ensure forecast doesn't deviate too much from last known price
            last_price = values[-1]
            max_deviation = 0.05  # 5% maximum deviation per day
            
            # Adjust forecasts if they deviate too much
            for i in range(len(best_forecast)):
                max_price = last_price * (1 + max_deviation * (i + 1))
                min_price = last_price * (1 - max_deviation * (i + 1))
                best_forecast[i] = np.clip(best_forecast[i], min_price, max_price)
            
            # Store results
            self.forecasts['Theta'] = pd.DataFrame({
                'ds': dates,
                'yhat': best_forecast,
                'model': 'Theta'
            })
            
            return self.forecasts['Theta']
            
        except Exception as e:
            print(f"Error in Theta modeling: {str(e)}")
            return None

    def compare_models(self, periods=5):
        """Compare forecasts from all models."""
        try:
            print("\nTraining models and generating forecasts...")
            
            # Use existing champion Prophet model if available, otherwise use default
            if self.model is not None:
                prophet_forecast = self.make_predictions(periods=periods)
            else:
                # Train new Prophet model only if no champion exists
                self.train_model()
                prophet_forecast = self.make_predictions(periods=periods)
            
            # Only keep the future predictions (last 5 days)
            self.forecasts['Prophet'] = prophet_forecast[-periods:][['ds', 'yhat']].copy()
            self.forecasts['Prophet']['model'] = 'Prophet'
            
            # Train SARIMA model
            self.train_sarima_model(periods=periods)
            
            # Train Theta model
            self.train_theta_model(periods=periods)
            
            # Print forecasts for each model
            for model_name, forecast in self.forecasts.items():
                print(f"\n{model_name} Forecast for the next {periods} days:")
                print("Date                      Predicted Price")
                print("-" * 45)
                
                # Only show the future predictions
                for _, row in forecast.tail(periods).iterrows():
                    if isinstance(row['ds'], pd.Timestamp):
                        date_str = row['ds'].strftime('%Y-%m-%d')
                    else:
                        date_str = row['ds']
                    print(f"{date_str}           ${row['yhat']:.2f}")
            
            # Create comparison plot
            self.plot_model_comparisons()
            
            return self.forecasts
            
        except Exception as e:
            print(f"Error in model comparison: {str(e)}")
            return None

    def plot_model_comparisons(self):
        """Create a comparison plot of all model forecasts."""
        try:
            fig = go.Figure()
            
            # Plot historical data for the last 30 days for better visualization
            historical_days = 30
            fig.add_trace(go.Scatter(
                x=self.df['ds'].tail(historical_days),
                y=self.df['y'].tail(historical_days),
                name='Historical Data',
                mode='lines'
            ))
            
            # Plot only the future forecasts from each model
            colors = {'Prophet': 'red', 'SARIMA': 'green', 'Theta': 'blue'}
            
            for name, forecast in self.forecasts.items():
                # Each forecast should already contain only the future predictions (5 days)
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    name=f'{name} Forecast',
                    mode='lines+markers',
                    line=dict(color=colors[name], dash='dash')
                ))
            
            fig.update_layout(
                title=f'{self.stock_name} Stock Price Forecasts - Model Comparison',
                xaxis_title='Date',
                yaxis_title='Stock Price ($)',
                showlegend=True
            )
            
            # Save the comparison plot with explicit error handling
            plot_path = os.path.join(self.output_dir, f'{self.stock_name}_model_comparison.png')
            try:
                fig.write_image(plot_path)
                print(f"Comparison plot saved: {plot_path}")
            except Exception as e:
                print(f"Error saving comparison plot: {str(e)}")
            
            # Show the plot
            fig.show()
            
        except Exception as e:
            print(f"Error in plot_model_comparisons: {str(e)}")

    def run_full_analysis(self, use_best_params=False):
        """Run the complete analysis pipeline."""
        try:
            if self.df is None:
                self.prepare_data()
            
            # If use_best_params is True and we have results, set up the champion model first
            if use_best_params and self.hyperparameter_test_results:
                best_params = self.get_best_parameters()
                print("\nUsing best Prophet parameters found:", best_params)
                
                # Clean parameters to only include Prophet-specific ones
                prophet_params = {
                    k: v for k, v in best_params.items() 
                    if k in [
                        'changepoint_prior_scale', 'seasonality_prior_scale',
                        'holidays_prior_scale', 'seasonality_mode', 'interval_width',
                        'growth', 'n_changepoints', 'daily_seasonality',
                        'weekly_seasonality', 'yearly_seasonality'
                    ]
                }
                
                # Initialize and fit Prophet model with cleaned parameters
                self.model = Prophet(**prophet_params)
                self.model.fit(self.df)
            
            # Run model comparison using the champion model
            self.compare_models(periods=5)
            
            # Print comparison table
            self.print_model_comparison_table()
            
            return self.model, self.forecasts
            
        except Exception as e:
            print(f"Error in run_full_analysis: {str(e)}")
            raise

    def print_model_comparison_table(self):
        """Print a formatted table comparing the performance of all models."""
        try:
            # Calculate metrics for each model
            metrics = {}
            last_known_price = self.df['y'].iloc[-1]
            
            for model_name, forecast in self.forecasts.items():
                # Calculate average forecast
                avg_forecast = forecast['yhat'].mean()
                
                # Calculate metrics
                metrics[model_name] = {
                    'First Day': forecast['yhat'].iloc[0],
                    'Last Day': forecast['yhat'].iloc[-1],
                    'Average': avg_forecast,
                    'Change': ((forecast['yhat'].iloc[-1] - last_known_price) / last_known_price) * 100
                }
            
            # Print the comparison table
            print("\n" + "="*80)
            print(f"Model Comparison for {self.stock_name}")
            print("="*80)
            
            # Print last known price
            print(f"Last Known Price: ${last_known_price:.2f}")
            print("-"*80)
            
            # Header
            print(f"{'Model':<10} {'First Day':>12} {'Last Day':>12} {'Average':>12} {'% Change':>12}")
            print("-"*80)
            
            # Print each model's metrics
            for model, stats in metrics.items():
                print(f"{model:<10} ${stats['First Day']:>11.2f} ${stats['Last Day']:>11.2f} "
                      f"${stats['Average']:>11.2f} {stats['Change']:>11.2f}%")
            
            print("="*80)
            
            # Print model-specific metrics if available
            if hasattr(self, 'current_best_mse'):
                print(f"\nProphet Best MSE: {self.current_best_mse:.2f} (RMSE: ±${np.sqrt(self.current_best_mse):.2f})")
            
            # Add any additional relevant metrics or notes
            print("\nNote: % Change represents the predicted price change from the last known price")
            print("      to the final forecast day.")
            
        except Exception as e:
            print(f"Error in model comparison table: {str(e)}")

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
        all_comparisons = []  # New list to store comparison metrics
        
        print(f"\nFound {len(csv_files)} CSV files to process")
        
        # Process each CSV file
        for csv_file in csv_files:
            print(f"\nProcessing {csv_file}...")
            
            # Initialize forecaster for current file
            file_path = os.path.join(price_data_dir, csv_file)
            forecaster = StockForecaster(file_path)
            forecaster.output_dir = output_dir
            
            # Load and print data sample
            forecaster.prepare_data()
            print("\nData shape:", forecaster.df.shape)
            
            # Set baseline MSE
            baseline_mse = 15
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
                
                # Collect comparison metrics for this stock
                metrics = {}
                last_known_price = forecaster.df['y'].iloc[-1]
                
                for model_name, forecast in forecaster.forecasts.items():
                    mse, rmse = forecaster.calculate_model_mse(model_name)
                    metrics[model_name] = {
                        'First Day': forecast['yhat'].iloc[0],
                        'Last Day': forecast['yhat'].iloc[-1],
                        'Average': forecast['yhat'].mean(),
                        'Change': ((forecast['yhat'].iloc[-1] - last_known_price) / last_known_price) * 100,
                        'MSE': mse if mse is not None else float('nan'),
                        'RMSE': rmse if rmse is not None else float('nan')
                    }
                
                all_comparisons.append({
                    'stock': stock_name,
                    'last_price': last_known_price,
                    'metrics': metrics
                })
                
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")
                continue
        
        # Save consolidated results
        if all_results:
            results_filename = 'forecast_results.csv'
            results_path = os.path.join(output_dir, results_filename)
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(results_path, index=False)
            print(f"\nConsolidated results saved to: {results_path}")
        
        # Print final comparison tables for all stocks
        print("\n" + "="*100)
        print("FINAL MODEL COMPARISONS")
        print("="*100)
        
        for comparison in all_comparisons:
            stock = comparison['stock']
            last_price = comparison['last_price']
            metrics = comparison['metrics']
            
            print(f"\nStock: {stock}")
            print("="*100)
            print(f"Last Known Price: ${last_price:.2f}")
            print("-"*100)
            
            # Header
            print(f"{'Model':<10} {'First Day':>12} {'Last Day':>12} {'Average':>12} {'% Change':>12} "
                  f"{'MSE':>12} {'RMSE':>12}")
            print("-"*100)
            
            # Print each model's metrics
            for model, stats in metrics.items():
                print(f"{model:<10} ${stats['First Day']:>11.2f} ${stats['Last Day']:>11.2f} "
                      f"${stats['Average']:>11.2f} {stats['Change']:>11.2f}% "
                      f"${stats['MSE']:>11.2f} ${stats['RMSE']:>11.2f}")
            
            print("="*100)
        
        print("\nNotes:")
        print("- % Change represents the predicted price change from the last known price to the final forecast day")
        print("- MSE (Mean Squared Error) represents the average squared difference between predictions and actual values")
        print("- RMSE (Root Mean Squared Error) represents the average error margin in dollars (±)")
        
        return 0  # Success
    
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        return 1  # Error
    

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)