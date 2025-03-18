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
import pickle

class BaseTimeSeriesModel:
    def __init__(self, csv_path):
        """Initialize the base time series model with data path."""
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.forecast = None
        
        self.stock_name = os.path.splitext(os.path.basename(csv_path))[0].split('_')[0]
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(csv_path)), 'forecast_output')
        os.makedirs(self.output_dir, exist_ok=True)

    def save_model(self, model_type):
        """Save the trained model to a pickle file."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.stock_name}_{model_type}_Model_{timestamp}.pkl"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        return filepath

    def prepare_data(self, max_training_days=None):
        """Read and prepare the stock data."""
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.rename(columns={'Date': 'ds', 'Price': 'y'})
        self.df['ds'] = pd.to_datetime(self.df['ds'])
        self.df = self.df.sort_values('ds')
        
        if max_training_days is not None and max_training_days < len(self.df):
            self.df = self.df.tail(max_training_days)
        
        return self.df

    def calculate_mse(self, forecast_df, is_training=False):
        """Calculate Mean Squared Error for the forecast."""
        try:
            if is_training:
                last_30_days = self.df.tail(30)
                comparison_df = forecast_df[forecast_df['ds'].isin(last_30_days['ds'])]
            else:
                comparison_df = forecast_df.merge(
                    self.df[['ds', 'y']], 
                    on='ds', 
                    how='inner'
                )
                
            if len(comparison_df) == 0:
                return float('inf'), float('inf')
            
            squared_diff = (comparison_df['yhat'].values - comparison_df['y'].values) ** 2
            squared_diff = squared_diff[~np.isnan(squared_diff)]
            
            if len(squared_diff) == 0:
                return float('inf'), float('inf')
            
            mse = np.mean(squared_diff)
            rmse = np.sqrt(mse)
            
            return mse, rmse
            
        except Exception as e:
            return float('inf'), float('inf')

    def plot_forecast(self, forecast_df):
        """Create and save forecast plot."""
        try:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=self.df['ds'],
                y=self.df['y'],
                name='Historical Data',
                mode='lines'
            ))

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

            return fig
        except Exception as e:
            return None