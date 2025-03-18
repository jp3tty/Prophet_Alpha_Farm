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
import logging

# Suppress cmdstanpy logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

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
        self.baseline_mse = 15.0  # Static baseline MSE

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
                
                if mse < self.best_mse and mse < self.baseline_mse:
                    self.best_mse = mse
                    self.best_params = params
                    self.model = current_model
                    print(f"\nNew best MSE found: {mse:.2f} (RMSE: {rmse:.2f})")
                    print(f"Parameters: {params}")
                    
                    # Create and save plots for the new best model
                    self.plot_forecast(forecast_df)
                    self.plot_focused_forecast(forecast_df)
                    
                    # Create a plot specifically showing the MSE improvement
                    fig = go.Figure()
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=mse,
                        title={'text': f"Current Best MSE: {mse:.2f}"},
                        gauge={
                            'axis': {'range': [0, self.baseline_mse]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, self.baseline_mse], 'color': "lightgray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': mse
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        title=f'{self.stock_name} - MSE Progress (Baseline: {self.baseline_mse:.2f})',
                        height=300
                    )
                    
                    if self.output_dir:
                        mse_plot_path = os.path.join(
                            self.output_dir,
                            f'{self.stock_name}_mse_progress.png'
                        )
                        fig.write_image(mse_plot_path)
                        print(f"MSE progress plot saved: {mse_plot_path}")
                    
            except Exception:
                continue
        
        if self.model is not None:
            self.save_model('Prophet')
            print(f"\nFinal best MSE: {self.best_mse:.2f}")
            print(f"Best parameters: {self.best_params}")
            return self.best_mse
        else:
            print("\nNo model found that beats the baseline MSE of 15.0")
            raise Exception("No valid model found during grid search")

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

    def plot_forecast(self, forecast_df):
        """Create and save forecast plot."""
        try:
            # Create full plot
            fig = go.Figure()

            # Plot historical data (last 14 days for visualization)
            historical_data = self.df.tail(14)
            fig.add_trace(go.Scatter(
                x=historical_data['ds'],
                y=historical_data['y'],
                name='Historical Data',
                mode='lines',
                line=dict(color='blue', width=2)
            ))

            # Get only the future predictions (next 5 days)
            future_data = forecast_df[forecast_df['ds'] > self.df['ds'].iloc[-1]]
            fig.add_trace(go.Scatter(
                x=future_data['ds'],
                y=future_data['yhat'],
                name='Forecast',
                mode='lines',
                line=dict(color='red', dash='dot', width=2)
            ))

            # Add last known price point
            last_known_price = self.df['y'].iloc[-1]
            last_known_date = self.df['ds'].iloc[-1]
            fig.add_trace(go.Scatter(
                x=[last_known_date],
                y=[last_known_price],
                name='Last Known Price',
                mode='markers',
                marker=dict(color='black', size=10, symbol='circle')
            ))

            # Calculate x-axis range
            min_date = historical_data['ds'].min()
            # Add 2 days to the last prediction date
            max_date = future_data['ds'].max() + pd.Timedelta(days=2)

            # Update layout
            fig.update_layout(
                title=f'{self.stock_name} Stock Price Forecast',
                xaxis_title='Date',
                yaxis_title='Stock Price ($)',
                showlegend=True,
                xaxis=dict(
                    type='date',
                    tickformat='%Y-%m-%d',
                    tickangle=45,
                    range=[min_date, max_date]  # Set explicit range
                ),
                yaxis=dict(
                    tickprefix='$',
                    tickformat='.2f'
                ),
                hovermode='x unified'
            )

            # Save plot if output directory exists
            if self.output_dir:
                plot_path = os.path.join(
                    self.output_dir, 
                    f'{self.stock_name}_{self.__class__.__name__}_forecast.png'
                )
                fig.write_image(plot_path)
                print(f"Plot saved: {plot_path}")

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