"""
BaseTimeSeriesModel - Abstract Base Class for Time Series Models

This module provides the foundation for all time series models in the system. It defines
the common interface and shared functionality that all specific time series models must implement.

Key Components:
- Abstract base class with required method definitions
- Common data preparation and validation
- Shared plotting and evaluation utilities
- Standardized interface for model training and prediction

Required Methods:
- prepare_data(): Load and preprocess time series data
- train_model(): Train the time series model
- make_predictions(): Generate future predictions
- calculate_mse(): Evaluate model performance
- plot_forecast(): Visualize model results
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os

class BaseTimeSeriesModel:
    def __init__(self, csv_path):
        """Initialize the base time series model with data path."""
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.forecast = None
        
        # Extract stock name from csv path
        self.stock_name = os.path.splitext(os.path.basename(csv_path))[0].split('_')[0]
        self.output_dir = None

    def prepare_data(self):
        """Read and prepare the stock data."""
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.rename(columns={'Date': 'ds', 'Price': 'y'})
        self.df['ds'] = pd.to_datetime(self.df['ds'])
        return self.df

    def calculate_mse(self, forecast_df):
        """Calculate Mean Squared Error for the forecast."""
        try:
            comparison_df = forecast_df.merge(
                self.df[['ds', 'y']], 
                on='ds', 
                how='inner'
            )
            
            mse = np.mean((comparison_df['y'] - comparison_df['yhat'])**2)
            rmse = np.sqrt(mse)
            
            return mse, rmse
            
        except Exception as e:
            print(f"Error in calculate_mse: {str(e)}")
            return None, None

    def plot_forecast(self, forecast_df):
        """Create and save forecast plot."""
        try:
            fig = go.Figure()

            # Plot historical data
            fig.add_trace(go.Scatter(
                x=self.df['ds'],
                y=self.df['y'],
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

            fig.update_layout(
                title=f'{self.stock_name} Stock Price Forecast - {self.__class__.__name__}',
                xaxis_title='Date',
                yaxis_title='Stock Price ($)',
                showlegend=True
            )

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