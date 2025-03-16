"""
SarimaTimeSeriesModel - SARIMA Implementation

Implements the Seasonal Autoregressive Integrated Moving Average (SARIMA) model,
which is particularly effective for time series with both trend and seasonal components.

Key Features:
- Grid search for optimal SARIMA parameters
- Automatic differencing for stationarity
- Seasonal pattern modeling
- Log transformation for multiplicative effects
- AIC-based model selection

Parameters:
- p: Autoregressive order
- d: Differencing order
- q: Moving average order
- P: Seasonal autoregressive order
- D: Seasonal differencing order
- Q: Seasonal moving average order
- s: Seasonal period
"""

from statsmodels.tsa.statespace.sarimax import SARIMAX
from base_timeseries import BaseTimeSeriesModel
import pandas as pd
import numpy as np
from datetime import timedelta
import os
import plotly.graph_objects as go

class SarimaTimeSeriesModel(BaseTimeSeriesModel):
    def __init__(self, csv_path):
        super().__init__(csv_path)
        self.param_grid = {
            'p': [1],     # Autoregressive order
            'd': [1],     # Differencing order
            'q': [1],     # Moving average order
            'P': [0, 1],  # Seasonal autoregressive order
            'D': [0],     # Seasonal differencing order
            'Q': [0],     # Seasonal moving average order
            's': [5]      # Seasonal period (weekly)
        }

    def train_model(self, periods=5):
        """Train and forecast using SARIMA model with parameter grid search."""
        if self.df is None:
            self.prepare_data()

        # Create time series without explicit frequency
        dates = pd.to_datetime(self.df['ds'])
        data = pd.Series(
            np.log(self.df['y'].values),
            index=dates,
            name='price'
        )

        best_aic = float('inf')
        best_model = None

        # Grid search
        for p in self.param_grid['p']:
            for d in self.param_grid['d']:
                for q in self.param_grid['q']:
                    for P in self.param_grid['P']:
                        for D in self.param_grid['D']:
                            for Q in self.param_grid['Q']:
                                for s in self.param_grid['s']:
                                    try:
                                        model = SARIMAX(
                                            data,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, s),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False
                                        )
                                        results = model.fit(disp=False)
                                        
                                        if results.aic < best_aic:
                                            best_aic = results.aic
                                            best_model = results
                                            
                                    except Exception as e:
                                        print(f"Error with parameters p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q}, s={s}: {str(e)}")
                                        continue

        if best_model is None:
            raise ValueError("No valid model found with the given parameters")

        self.model = best_model
        return self.model

    def make_predictions(self, periods=5):
        """Generate predictions for the specified number of periods."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Generate future dates using business day frequency
        last_date = self.df['ds'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq='B'  # Business days
        )

        # Generate forecast and transform back from log scale
        forecast_values = np.exp(self.model.forecast(periods))

        # Create forecast DataFrame
        self.forecast = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values
        })

        return self.forecast

    def calculate_mse(self, forecast_df):
        """Calculate Mean Squared Error for the forecast."""
        try:
            # Get the last 30 days of actual data for comparison
            last_30_days = self.df.tail(30)
            
            # Get model predictions for the last 30 days
            predictions = np.exp(self.model.predict(start=len(self.df)-30, end=len(self.df)-1))
            
            # Calculate MSE and RMSE
            mse = np.mean((last_30_days['y'].values - predictions)**2)
            rmse = np.sqrt(mse)
            
            return mse, rmse
            
        except Exception as e:
            print(f"Error in calculate_mse: {str(e)}")
            return None, None

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
                title=f'{self.stock_name} Stock Price Forecast - SARIMA Model<br>MSE: {self.mse:.4f}',
                xaxis_title='Date',
                yaxis_title='Stock Price ($)',
                showlegend=True
            )

            if self.output_dir:
                plot_path = os.path.join(
                    self.output_dir, 
                    f'{self.stock_name}_sarima_forecast.png'
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
                title=f'{self.stock_name} Stock Price - Last 5 Days & Next 5 Days<br>SARIMA Model',
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
                    f'{self.stock_name}_sarima_focused_forecast.png'
                )
                fig.write_image(plot_path)
                print(f"Focused plot saved: {plot_path}")

            return fig
        except Exception as e:
            print(f"Error creating focused plot: {str(e)}")
            return None 