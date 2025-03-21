"""
ProphetTimeSeriesModel - Facebook Prophet Implementation

Implements time series forecasting using Facebook's Prophet algorithm, which is particularly
effective for business time series with strong seasonal patterns and multiple seasonality.

Key Features:
- Grid search optimization for model parameters
- Automatic seasonality detection
- Holiday effect modeling
- Trend changepoint detection
- Uncertainty interval estimation

Parameters:
- changepoint_prior_scale: Controls trend flexibility
- seasonality_prior_scale: Controls seasonality strength
- holidays_prior_scale: Controls holiday effect strength
- seasonality_mode: 'multiplicative' or 'additive'
- interval_width: Width of prediction intervals
- growth: Type of trend growth
- n_changepoints: Number of trend changepoints
- daily/weekly/yearly_seasonality: Seasonality components
"""

from prophet import Prophet
from base_timeseries import BaseTimeSeriesModel
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import os
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly
import logging
from datetime import datetime, timedelta
from visualizations import TimeSeriesPlotter

# Suppress cmdstanpy logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

class ProphetTimeSeriesModel(BaseTimeSeriesModel):
    def __init__(self, csv_path):
        super().__init__(csv_path)
        # Define the exact parameters from prophet_EGO.py to ensure we try them first
        self.ego_params = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10,
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'yearly_seasonality': True
        }
        
        self.param_grid = {
            'changepoint_prior_scale': [0.05, 0.5, 0.1, 0.2],  # Include 0.05 from prophet_EGO
            'seasonality_prior_scale': [10, 2, 5, 7],  # Include 10 from prophet_EGO
            'holidays_prior_scale': [0.01],
            'seasonality_mode': ['multiplicative'],
            'interval_width': [0.05, 0.1, 0.15],
            'growth': ['linear'],
            'n_changepoints': [25, 35, 45],
            'daily_seasonality': [True],  # Match prophet_EGO
            'weekly_seasonality': [True],  # Match prophet_EGO
            'yearly_seasonality': [True]  # Match prophet_EGO
        }
        self.best_params = None
        self.best_mse = float('inf')
        self.baseline_mse = 15.0  # Static baseline MSE
        self.mse_history = []  # Track MSE history during grid search

    def train_model(self, **kwargs):
        """Train the Prophet model with optional grid search for hyperparameters."""
        if self.df is None:
            self.prepare_data()
            
        # If kwargs are provided, use them directly
        if kwargs:
            try:
                self.model = Prophet(**kwargs)
                self.model.fit(self.df)
                forecast_df = self.make_predictions()
                mse, rmse = self.calculate_mse(forecast_df, is_training=True)
                self.save_model('Prophet')
                return mse
            except Exception as e:
                return float('inf')

        # First, try the exact prophet_EGO.py parameters
        print(f"\nTrying prophet_EGO.py parameters first...")
        try:
            ego_model = Prophet(**self.ego_params)
            ego_model.fit(self.df)
            forecast_df = self.make_predictions(model=ego_model)
            mse, rmse = self.calculate_mse(forecast_df, is_training=True)
            
            print(f"EGO parameters MSE: {mse:.4f}, RMSE: {rmse:.4f}")
            
            if mse < self.best_mse and mse < self.baseline_mse:
                self.best_mse = mse
                self.best_params = self.ego_params
                self.model = ego_model
                print(f"prophet_EGO.py parameters beat baseline with MSE: {mse:.4f}")
                self.create_and_save_plots(forecast_df)
                self.save_model('Prophet')
                self.save_mse_history()
                return self.best_mse
        except Exception as e:
            print(f"Error with prophet_EGO.py parameters: {str(e)}")
        
        # Otherwise perform grid search
        all_params = [dict(zip(self.param_grid.keys(), v)) 
                     for v in product(*self.param_grid.values())]
        
        print(f"\nStarting grid search with {len(all_params)} parameter combinations...")
        print(f"Baseline MSE: {self.baseline_mse:.2f}")
        
        for params in tqdm(all_params, desc="Grid Search", position=0, leave=True, ncols=100):
            try:
                # Create and train a new Prophet model with current parameters
                current_model = Prophet(**params)
                current_model.fit(self.df)
                
                # Make predictions and calculate MSE
                forecast_df = self.make_predictions(model=current_model)
                mse, rmse = self.calculate_mse(forecast_df, is_training=True)
                
                # Record MSE history
                self.mse_history.append({
                    'timestamp': datetime.now(),
                    'mse': mse,
                    'rmse': rmse,
                    'params': params
                })
                
                if mse < self.best_mse and mse < self.baseline_mse:
                    self.best_mse = mse
                    self.best_params = params
                    self.model = current_model
                    print(f"\nNew best MSE found: {mse:.2f} (RMSE: {rmse:.2f})")
                    print(f"Parameters: {params}")
                    
                    # Create and save plots for the new best model
                    self.create_and_save_plots(forecast_df)
                    
            except Exception:
                continue
        
        if self.model is not None:
            self.save_model('Prophet')
            self.save_mse_history()
            print(f"\nFinal best MSE: {self.best_mse:.2f}")
            print(f"Best parameters: {self.best_params}")
            return self.best_mse
        else:
            print("\nNo model found that beats the baseline MSE of 15.0")
            raise Exception("No valid model found during grid search")

    def create_and_save_plots(self, forecast_df):
        """Create and save all plots for the current best model."""
        # Create the main forecast plot with rangeslider disabled
        fig_forecast = plot_plotly(self.model, forecast_df, xlabel='Date', ylabel='Stock Price ($)')
        
        # Remove the secondary plot (rangeslider) at the bottom
        fig_forecast.update_layout(
            title=f'{self.stock_name} Stock Price Forecast',
            showlegend=True,
            xaxis=dict(rangeslider=dict(visible=False))  # This disables the rangeslider/secondary plot
        )
        
        # Create the components plot
        fig_components = plot_components_plotly(self.model, forecast_df)
        fig_components.update_layout(title=f'{self.stock_name} - Model Components')
        
        # Create custom price change distribution plot
        returns = self.df['y'].pct_change().dropna()
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=returns, 
                                      nbinsx=50,
                                      name='Daily Returns',
                                      showlegend=True))
        fig_dist.update_layout(title=f'{self.stock_name} - Distribution of Daily Price Changes',
                              xaxis_title='Daily Return',
                              yaxis_title='Frequency',
                              bargap=0.1)
        
        # Create focused forecast plot
        fig_focused = self.plot_focused_forecast(forecast_df)
        
        # Save all plots
        if self.output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save forecast plot
            forecast_path = os.path.join(
                self.output_dir,
                f'{self.stock_name}_forecast_{timestamp}.png'
            )
            fig_forecast.write_image(forecast_path)
            
            # Save components plot
            components_path = os.path.join(
                self.output_dir,
                f'{self.stock_name}_components_{timestamp}.png'
            )
            fig_components.write_image(components_path)
            
            # Save distribution plot
            dist_path = os.path.join(
                self.output_dir,
                f'{self.stock_name}_distribution_{timestamp}.png'
            )
            fig_dist.write_image(dist_path)
            
            # Save focused forecast plot
            if fig_focused:
                focused_path = os.path.join(
                    self.output_dir,
                    f'{self.stock_name}_focused_forecast_{timestamp}.png'
                )
                fig_focused.write_image(focused_path)
                print(f"Focused plot saved: {focused_path}")
            
            print(f"Plots saved with timestamp: {timestamp}")

    def plot_forecast(self, forecast_df, output_dir=None):
        """
        Create and save a forecast plot for the stock price.
        
        Args:
            forecast_df (pd.DataFrame): Forecast DataFrame from Prophet.
            output_dir (str, optional): Directory to save the plot. Defaults to None.
        """
        try:
            # Use the unified plotter for creating plots
            plotter = TimeSeriesPlotter(output_dir=output_dir or self.output_dir)
            
            # Generate the main forecast plot
            fig = plotter.plot_forecast(
                df=self.df,
                forecast_df=forecast_df,
                stock_name=self.stock_name,
                model_name='Prophet',
                mse=self.mse
            )
            
            # Generate the focused forecast plot
            plotter.plot_focused_forecast(
                df=self.df,
                forecast_df=forecast_df, 
                stock_name=self.stock_name,
                model_name='Prophet'
            )
            
            # Also create and save the components plot
            if hasattr(self, 'model') and self.model is not None:
                # Generate components plot using Prophet's built-in function
                components_fig = plot_components_plotly(self.model, forecast_df)
                components_fig.update_layout(title=f'{self.stock_name} Stock Price Components - Prophet Model')
                plotter.plot_components(components_fig, self.stock_name, 'Prophet')
            
            # Create distribution plot
            plotter.plot_distribution(self.df, self.stock_name, 'Prophet')
                
            return fig
        except Exception as e:
            print(f"Error creating Prophet forecast plot: {str(e)}")
            return None
    
    def plot_focused_forecast(self, forecast_df, output_dir=None):
        """
        Create and save a focused forecast plot for the last 5 days and next 10 days.
        This method is kept for backwards compatibility but delegates to the TimeSeriesPlotter.
        
        Args:
            forecast_df (pd.DataFrame): Forecast DataFrame from Prophet.
            output_dir (str, optional): Directory to save the plot. Defaults to None.
        """
        try:
            # Use the unified plotter - note this is now called from plot_forecast
            plotter = TimeSeriesPlotter(output_dir=output_dir or self.output_dir)
            return plotter.plot_focused_forecast(
                df=self.df,
                forecast_df=forecast_df, 
                stock_name=self.stock_name,
                model_name='Prophet'
            )
        except Exception as e:
            print(f"Error creating Prophet focused forecast plot: {str(e)}")
            return None

    def save_mse_history(self):
        """Save the MSE history to a CSV file."""
        if self.mse_history and self.output_dir:
            df_history = pd.DataFrame(self.mse_history)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            history_path = os.path.join(
                self.output_dir,
                f'{self.stock_name}_mse_history_{timestamp}.csv'
            )
            df_history.to_csv(history_path, index=False)
            print(f"MSE history saved to: {history_path}")

    def make_predictions(self, periods=5, model=None):
        """Generate predictions for the specified number of periods."""
        if model is None:
            model = self.model
            
        if model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Store forecast if using the main model
        if model == self.model:
            self.forecast = forecast
            
        return forecast 