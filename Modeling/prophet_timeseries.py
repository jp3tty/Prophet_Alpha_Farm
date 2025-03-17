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

class ProphetTimeSeriesModel(BaseTimeSeriesModel):
    def __init__(self, csv_path):
        super().__init__(csv_path)
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
        self.best_params = None
        self.best_mse = float('inf')

    def train_model(self, **kwargs):
        """Train the Prophet model with optional grid search for hyperparameters."""
        # If kwargs are provided, use them directly
        if kwargs:
            self.model = Prophet(**kwargs)
            self.model.fit(self.df)
            forecast_df = self.make_predictions()
            mse, rmse = self.calculate_mse(forecast_df, is_training=True)
            print(f"Model trained with MSE: {mse:.2f}, RMSE: {rmse:.2f}")
            self.save_model('Prophet')
            return mse

        # Otherwise perform grid search
        all_params = [dict(zip(self.param_grid.keys(), v)) 
                     for v in product(*self.param_grid.values())]
        
        print(f"Grid searching through {len(all_params)} combinations...")
        
        for params in tqdm(all_params):
            try:
                m = Prophet(**params)
                m.fit(self.df)
                forecast_df = self.make_predictions(model=m)
                mse, rmse = self.calculate_mse(forecast_df, is_training=True)
                
                if mse < self.best_mse:
                    self.best_mse = mse
                    self.best_params = params
                    self.model = m
                    print(f"\nNew best MSE: {mse:.2f}, RMSE: {rmse:.2f}")
                    print(f"Parameters: {params}")
            except Exception as e:
                print(f"Error with parameters {params}: {str(e)}")
                continue
        
        if self.model is not None:
            print("\nBest model parameters:")
            print(self.best_params)
            print(f"Best MSE: {self.best_mse:.2f}")
            self.save_model('Prophet')
            return self.best_mse
        else:
            raise Exception("No valid model found during grid search")

    def make_predictions(self, periods=5):
        """Generate predictions for the specified number of periods."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        future = self.model.make_future_dataframe(periods=periods)
        self.forecast = self.model.predict(future)
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
                title=f'{self.stock_name} Stock Price Forecast - Prophet Model<br>MSE: {self.best_mse:.4f}',
                xaxis_title='Date',
                yaxis_title='Stock Price ($)',
                showlegend=True
            )

            if self.output_dir:
                plot_path = os.path.join(
                    self.output_dir, 
                    f'{self.stock_name}_prophet_forecast.png'
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
                title=f'{self.stock_name} Stock Price - Last 5 Days & Next 5 Days<br>Prophet Model',
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
                    f'{self.stock_name}_prophet_focused_forecast.png'
                )
                fig.write_image(plot_path)
                print(f"Focused plot saved: {plot_path}")

            return fig
        except Exception as e:
            print(f"Error creating focused plot: {str(e)}")
            return None 