"""
ThetaTimeSeriesModel - Theta Method Implementation

Implements the Theta method for time series forecasting, which decomposes the series
into trend and seasonal components using a combination of moving averages and polynomial fitting.

Key Features:
- Grid search for optimal parameters
- Multiplicative decomposition
- Polynomial trend fitting
- Seasonal pattern detection
- Price movement constraints

Parameters:
- theta: Theta parameter for trend adjustment
- window_size: Size of moving average windows
- decomposition_type: Type of decomposition
- trend_degree: Polynomial degree for trend
- seasonal_window: Window for seasonal pattern
"""

from base_timeseries import BaseTimeSeriesModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import os
from itertools import product
from tqdm import tqdm

class ThetaTimeSeriesModel:
    def __init__(self, csv_path):
        """Initialize the Theta model with data path."""
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.forecast = None
        self.output_dir = None
        
        # Extract stock name from csv path
        self.stock_name = os.path.splitext(os.path.basename(csv_path))[0].split('_')[0]
        
        # Enhanced parameter grid for Theta model
        self.param_grid = {
            'theta': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            'window_size': [5, 7, 10, 14, 21],
            'decomposition_type': ['multiplicative'],
            'trend_degree': [1, 2, 3],  # Polynomial degree for trend fitting
            'seasonal_window': [5, 7, 10]  # Window size for seasonal pattern
        }
        self.best_params = None
        self.best_mse = float('inf')

    def prepare_data(self):
        """Read and prepare the stock data."""
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.rename(columns={'Date': 'ds', 'Price': 'y'})
        self.df['ds'] = pd.to_datetime(self.df['ds'])
        print(f"Data range: {self.df['ds'].min()} to {self.df['ds'].max()}")
        return self.df

    def calculate_mse(self, values, params):
        """Calculate Mean Squared Error using the last window_size days of historical data."""
        try:
            window = params['window_size']
            theta = params['theta']
            trend_degree = params['trend_degree']
            seasonal_window = params['seasonal_window']
            
            # Calculate moving averages for the validation period
            long_ma = pd.Series(values).rolling(window=window*2, center=True).mean()
            short_ma = pd.Series(values).rolling(window=window, center=True).mean()
            
            # Fill NaN values
            long_ma = long_ma.fillna(method='bfill').fillna(method='ffill')
            short_ma = short_ma.fillna(method='bfill').fillna(method='ffill')
            
            # Calculate trend
            trend = long_ma + theta * (short_ma - long_ma)
            
            # Calculate seasonal pattern
            seasonal = values / trend
            seasonal = pd.Series(seasonal).rolling(window=seasonal_window, center=True).mean()
            seasonal = seasonal.fillna(method='bfill').fillna(method='ffill')
            
            # Fit trend model
            X = np.arange(len(values)).reshape(-1, 1)
            trend_model = np.poly1d(np.polyfit(X.flatten(), trend, trend_degree))
            
            # Generate fitted values
            fitted_trend = trend_model(X.flatten())
            fitted_values = fitted_trend * seasonal
            
            # Calculate MSE and RMSE
            mse = mean_squared_error(values, fitted_values)
            rmse = np.sqrt(mse)
            
            return mse, rmse
            
        except Exception as e:
            print(f"Error in calculate_mse: {str(e)}")
            return None, None

    def train_model(self, **kwargs):
        """Train and forecast using Theta model with parameter tuning."""
        if self.df is None:
            self.prepare_data()

        values = self.df['y'].values
        train_size = min(252, len(values))  # Use last year of data or all if less
        values = values[-train_size:]
        print(f"Training on {len(values)} days of data")

        # If parameters provided, use them directly
        if kwargs:
            params = kwargs
            self.model = self._fit_model(values, params)
            return self.model

        # Otherwise, perform grid search
        print("\nPerforming grid search for best parameters...")
        param_combinations = [dict(zip(self.param_grid.keys(), v)) 
                           for v in product(*self.param_grid.values())]
        
        for params in tqdm(param_combinations, desc="Testing parameters"):
            try:
                # Calculate MSE on validation data
                mse, rmse = self.calculate_mse(values, params)
                
                if mse is not None and mse < self.best_mse:
                    self.best_mse = mse
                    self.best_params = params
                    self.model = self._fit_model(values, params)
                    print(f"\nNew best MSE found: {mse:.4f}")
                    print(f"Parameters: {params}")
                    
            except Exception as e:
                print(f"\nError with parameters {params}:")
                print(f"Error details: {str(e)}")
                continue

        if self.best_params:
            print(f"\nBest parameters found:")
            print(f"Parameters: {self.best_params}")
            print(f"MSE: {self.best_mse:.4f}")
        else:
            raise ValueError("No valid parameters found during grid search")

        return self.model

    def _fit_model(self, values, params):
        """Helper method to fit the model with given parameters."""
        try:
            window = params['window_size']
            theta = params['theta']
            trend_degree = params['trend_degree']
            seasonal_window = params['seasonal_window']
            
            # Calculate moving averages
            long_ma = pd.Series(values).rolling(window=window*2, center=True).mean()
            short_ma = pd.Series(values).rolling(window=window, center=True).mean()
            
            # Fill NaN values
            long_ma = long_ma.fillna(method='bfill').fillna(method='ffill')
            short_ma = short_ma.fillna(method='bfill').fillna(method='ffill')
            
            # Calculate trend
            trend = long_ma + theta * (short_ma - long_ma)
            
            # Calculate seasonal pattern
            seasonal = values / trend
            seasonal = pd.Series(seasonal).rolling(window=seasonal_window, center=True).mean()
            seasonal = seasonal.fillna(method='bfill').fillna(method='ffill')
            
            # Fit trend model
            X = np.arange(len(values)).reshape(-1, 1)
            trend_model = np.poly1d(np.polyfit(X.flatten(), trend, trend_degree))
            
            # Generate forecast components
            future_X = np.arange(len(values), len(values) + 5).reshape(-1, 1)
            trend_forecast = trend_model(future_X.flatten())
            
            # Calculate recent average seasonal factor
            recent_seasonal = seasonal[-seasonal_window:].mean()
            
            # Generate final forecast
            forecast = trend_forecast * recent_seasonal
            
            return {
                'theta': theta,
                'window_size': window,
                'decomposition_type': params['decomposition_type'],
                'trend_degree': trend_degree,
                'seasonal_window': seasonal_window,
                'forecast': forecast,
                'trend_model': trend_model,
                'seasonal': seasonal
            }
            
        except Exception as e:
            print(f"Error fitting model: {str(e)}")
            return None

    def make_predictions(self, periods=5):
        """Generate predictions for the specified number of periods."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Create dates for forecast
        last_date = self.df['ds'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='B'  # Business days
        )

        # Get the forecast values from the training
        forecast_values = self.model['forecast']

        # Apply constraints to prevent unrealistic price movements
        last_price = self.df['y'].iloc[-1]
        max_deviation = 0.05  # 5% maximum deviation per day
        
        # Adjust forecasts if they deviate too much
        for i in range(len(forecast_values)):
            max_price = last_price * (1 + max_deviation * (i + 1))
            min_price = last_price * (1 - max_deviation * (i + 1))
            forecast_values[i] = np.clip(forecast_values[i], min_price, max_price)

        # Create forecast DataFrame
        self.forecast = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values
        })

        return self.forecast

    def plot_forecast(self, forecast_df):
        """Create and save forecast plot."""
        try:
            # Create full plot
            fig = go.Figure()

            # Plot historical data (last 60 days for better visualization)
            historical_data = self.df.tail(60)
            fig.add_trace(go.Scatter(
                x=historical_data['ds'],
                y=historical_data['y'],
                name='Historical Data',
                mode='lines'
            ))

            # Plot forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat'],
                name='Forecast',
                mode='lines+markers',
                line=dict(dash='dash')
            ))

            # Add last known price point
            last_known_price = self.df['y'].iloc[-1]
            last_known_date = self.df['ds'].iloc[-1]
            fig.add_trace(go.Scatter(
                x=[last_known_date],
                y=[last_known_price],
                name='Last Known Price',
                mode='markers',
                marker=dict(size=10, color='red')
            ))

            fig.update_layout(
                title=f'{self.stock_name} Stock Price Forecast - Theta Model<br>MSE: {self.best_mse:.4f}',
                xaxis_title='Date',
                yaxis_title='Stock Price ($)',
                showlegend=True
            )

            if self.output_dir:
                plot_path = os.path.join(
                    self.output_dir, 
                    f'{self.stock_name}_theta_forecast.png'
                )
                fig.write_image(plot_path)
                print(f"Plot saved: {plot_path}")

            # Create focused plot
            self.plot_focused_forecast(forecast_df)
            
            return fig
        except Exception as e:
            print(f"Error creating plot: {str(e)}")
            return None

    def plot_focused_forecast(self, forecast_df):
        """Create and save a focused plot of last 5 days and next 5 days."""
        try:
            fig = go.Figure()

            # Plot last 5 days of historical data
            last_5_days = self.df.tail(5)
            fig.add_trace(go.Scatter(
                x=last_5_days['ds'],
                y=last_5_days['y'],
                name='Last 5 Days',
                mode='lines+markers',
                line=dict(color='blue')
            ))

            # Plot next 5 days forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat'],
                name='Next 5 Days',
                mode='lines+markers',
                line=dict(dash='dash', color='red')
            ))

            # Add last known price point
            last_known_price = self.df['y'].iloc[-1]
            last_known_date = self.df['ds'].iloc[-1]
            fig.add_trace(go.Scatter(
                x=[last_known_date],
                y=[last_known_price],
                name='Last Known Price',
                mode='markers',
                marker=dict(size=10, color='red')
            ))

            fig.update_layout(
                title=f'{self.stock_name} Stock Price - Last 5 Days & Next 5 Days<br>Theta Model',
                xaxis_title='Date',
                yaxis_title='Stock Price ($)',
                showlegend=True,
                xaxis=dict(
                    tickangle=45,
                    tickformat='%Y-%m-%d'
                )
            )

            if self.output_dir:
                plot_path = os.path.join(
                    self.output_dir, 
                    f'{self.stock_name}_theta_focused_forecast.png'
                )
                fig.write_image(plot_path)
                print(f"Focused plot saved: {plot_path}")

            return fig
        except Exception as e:
            print(f"Error creating focused plot: {str(e)}")
            return None