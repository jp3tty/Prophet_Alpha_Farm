"""
Time Series Visualization Module

This module provides a unified plotting class for time series forecasting models,
supporting Prophet, SARIMA, and Theta models with consistent styling and functionality.

Key Features:
- Standard forecast plots for full historical data and forecasts
- Focused forecast plots with 12-day window (last 5 days + 10 forecast days + 2 extra days)
- Confidence interval visualization
- Consistent styling across all models
- Automatic plot saving
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

class TimeSeriesPlotter:
    """
    Unified plotting class for time series models.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the plotter.
        
        Args:
            output_dir (str, optional): Directory to save plots. Defaults to None.
        """
        self.output_dir = output_dir
    
    def plot_forecast(self, df, forecast_df, stock_name, model_name, mse=None):
        """
        Create and save a full forecast plot with historical data and forecast.
        
        Args:
            df (pd.DataFrame): Historical data with 'ds' and 'y' columns
            forecast_df (pd.DataFrame): Forecast data with 'ds', 'yhat', and optionally 'yhat_lower'/'yhat_upper'
            stock_name (str): Stock symbol
            model_name (str): Model name (Prophet, SARIMA, Theta)
            mse (float, optional): MSE value to display in title. Defaults to None.
            
        Returns:
            plotly.graph_objects.Figure: The created figure
        """
        try:
            # Create full plot
            fig = go.Figure()

            # Plot historical data (last 60 days for better visualization)
            historical_data = df.tail(60)
            fig.add_trace(go.Scatter(
                x=historical_data['ds'],
                y=historical_data['y'],
                name='Historical Data',
                mode='markers',
                marker=dict(color='black', size=6)
            ))
            
            # Get all forecast data including historical fitted values
            # First, ensure we're working with the correct forecast format
            if 'yhat' not in forecast_df.columns:
                print("Warning: forecast_df does not contain 'yhat' column")
                return None
                
            # Sort by date to ensure continuous line
            sorted_forecast = forecast_df.sort_values('ds')
            
            # Plot complete model line (both fitted values and forecast)
            fig.add_trace(go.Scatter(
                x=sorted_forecast['ds'],
                y=sorted_forecast['yhat'],
                name='Model Fit & Forecast',
                mode='lines',
                line=dict(color='blue', width=2)
            ))
            
            # Add confidence intervals if available
            if 'yhat_lower' in sorted_forecast.columns and 'yhat_upper' in sorted_forecast.columns:
                # Filter out any NaN values
                ci_data = sorted_forecast.dropna(subset=['yhat_lower', 'yhat_upper'])
                
                if not ci_data.empty:
                    fig.add_trace(go.Scatter(
                        x=ci_data['ds'],
                        y=ci_data['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=ci_data['ds'],
                        y=ci_data['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(68, 68, 255, 0.2)',
                        name='95% Confidence Interval'
                    ))

            # Add last known price point
            last_known_price = df['y'].iloc[-1]
            last_known_date = df['ds'].iloc[-1]
            fig.add_trace(go.Scatter(
                x=[last_known_date],
                y=[last_known_price],
                name='Last Known Price',
                mode='markers',
                marker=dict(size=10, color='red')
            ))

            # Add MSE to title if provided
            title = f'{stock_name} Stock Price Forecast<br>{model_name} Model'
            if mse is not None:
                title = f'{stock_name} Stock Price Forecast<br>{model_name} Model (MSE: {mse:.4f})'
                
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Stock Price ($)',
                showlegend=True,
                xaxis=dict(
                    tickangle=45,
                    tickformat='%Y-%m-%d'
                )
            )

            if self.output_dir:
                # Create directory if it doesn't exist
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                    
                # Save plot
                model_prefix = model_name.lower()
                plot_path = os.path.join(
                    self.output_dir, 
                    f'{stock_name}_{model_prefix}_forecast.png'
                )
                fig.write_image(plot_path)
                print(f"Plot saved: {plot_path}")

            return fig
        except Exception as e:
            print(f"Error creating forecast plot: {str(e)}")
            return None

    def plot_focused_forecast(self, df, forecast_df, stock_name, model_name, model=None):
        """
        Create and save a focused plot of last 5 days and next 10 days, plus 2 extra days.
        
        Args:
            df (pd.DataFrame): Historical data with 'ds' and 'y' columns
            forecast_df (pd.DataFrame): Forecast data with 'ds', 'yhat', and optionally 'yhat_lower'/'yhat_upper'
            stock_name (str): Stock symbol
            model_name (str): Model name (Prophet, SARIMA, Theta)
            model (object, optional): Model object for SARIMA to generate forecasts directly. Defaults to None.
            
        Returns:
            plotly.graph_objects.Figure: The created figure
        """
        try:
            fig = go.Figure()

            # Get last 5 days of historical data
            last_5_days = df.tail(5)
            
            # Plot historical data as black dots
            fig.add_trace(go.Scatter(
                x=last_5_days['ds'],
                y=last_5_days['y'],
                name='Historical Data',
                mode='markers',
                marker=dict(color='black', size=8)
            ))

            # Last known date from historical data
            last_date = df['ds'].iloc[-1]
            
            # Process based on if we have a SARIMA model object
            is_sarima = model_name.upper() == 'SARIMA' and model is not None
            
            if is_sarima:
                # For SARIMA with model: Get fitted values + forecasts directly from model
                fitted_values = None
                if hasattr(model, 'results') and model.results is not None:
                    # Try to get fitted values from the fitted model
                    try:
                        fitted_values = model.results.fittedvalues[-5:]
                    except:
                        pass
                
                # If we couldn't get fitted values from results, use the model to predict on historical data
                if fitted_values is None:
                    try:
                        # Use the last 5 days indices to predict historical values
                        fitted_values = model.predict(start=len(df)-5, end=len(df)-1)
                    except:
                        # If that fails, use the forecast values for those dates (less accurate)
                        fitted_indices = forecast_df[forecast_df['ds'].isin(last_5_days['ds'])]
                        if not fitted_indices.empty:
                            fitted_values = fitted_indices['yhat'].values
                        else:
                            # Default to the actual values if we can't get predictions
                            fitted_values = last_5_days['y'].values
                
                # Generate a fresh 10-day forecast with confidence intervals
                forecast_values = model.forecast(10)
                
                # Get prediction intervals
                pred_intervals = model.get_forecast(10).conf_int(alpha=0.05)
                lower_bound = pred_intervals.iloc[:, 0]
                upper_bound = pred_intervals.iloc[:, 1]
                
                # Get 10 days of business days starting from the day after the last historical date
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=10,
                    freq='B'  # Business days
                )
                
                # For historical fitted values, approximate confidence intervals
                residuals = df['y'].values - model.fittedvalues
                std_error = np.std(residuals)
                historical_lower = fitted_values - 1.96 * std_error
                historical_upper = fitted_values + 1.96 * std_error
                
                # Combine historical fitted values with forecast
                all_dates = list(last_5_days['ds']) + list(forecast_dates)
                all_values = list(fitted_values) + list(forecast_values)
                all_lower = list(historical_lower) + list(lower_bound.values)
                all_upper = list(historical_upper) + list(upper_bound.values)
                
                # Create a DataFrame with all dates and values for the blue line
                line_data = pd.DataFrame({
                    'ds': all_dates,
                    'yhat': all_values,
                    'yhat_lower': all_lower,
                    'yhat_upper': all_upper
                })
            else:
                # For Prophet and Theta models: Use the provided forecast DataFrame
                # Extract fitted values for historical period
                historical_fitted = forecast_df[forecast_df['ds'].isin(last_5_days['ds'])].copy()
                
                # Extract forecast values for future dates (10 days)
                future_forecast = forecast_df[forecast_df['ds'] > last_date].head(10).copy()
                
                # If we don't have enough forecast days, pad with additional days
                if len(future_forecast) < 10:
                    missing_days = 10 - len(future_forecast)
                    last_forecast_date = future_forecast['ds'].iloc[-1] if len(future_forecast) > 0 else last_date
                    
                    # Generate missing dates
                    missing_dates = pd.date_range(
                        start=last_forecast_date + pd.Timedelta(days=1),
                        periods=missing_days,
                        freq='B'  # Business days
                    )
                    
                    # Create DataFrame with missing dates
                    missing_df = pd.DataFrame({
                        'ds': missing_dates,
                        'yhat': [future_forecast['yhat'].iloc[-1]] * missing_days if len(future_forecast) > 0 else [np.nan] * missing_days,
                        'yhat_lower': [future_forecast['yhat_lower'].iloc[-1]] * missing_days if 'yhat_lower' in future_forecast.columns and len(future_forecast) > 0 else [np.nan] * missing_days,
                        'yhat_upper': [future_forecast['yhat_upper'].iloc[-1]] * missing_days if 'yhat_upper' in future_forecast.columns and len(future_forecast) > 0 else [np.nan] * missing_days
                    })
                    
                    # Add the missing days
                    future_forecast = pd.concat([future_forecast, missing_df], ignore_index=True)
                
                # Combine historical and future data for a continuous line
                line_data = pd.concat([historical_fitted, future_forecast], ignore_index=True)
                
                # If we don't have historical fitted values, use actual values
                if len(historical_fitted) == 0:
                    # If no confidence intervals in forecast_df, create approximations
                    has_ci = 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns
                    
                    historical_as_fitted = pd.DataFrame({
                        'ds': last_5_days['ds'],
                        'yhat': last_5_days['y'],
                        'yhat_lower': last_5_days['y'] * 0.95 if has_ci else None,
                        'yhat_upper': last_5_days['y'] * 1.05 if has_ci else None
                    })
                    line_data = pd.concat([historical_as_fitted, future_forecast], ignore_index=True)
            
            # Sort by date to ensure continuous line
            line_data = line_data.sort_values('ds')
            
            # Add 2 extra days with no data
            last_forecast_date = line_data['ds'].iloc[-1]
            extra_dates = pd.date_range(
                start=last_forecast_date + pd.Timedelta(days=1),
                periods=2,
                freq='B'  # Business days
            )
            
            # Plot the continuous blue line (historical fitted + forecast)
            fig.add_trace(go.Scatter(
                x=line_data['ds'],
                y=line_data['yhat'],
                name='Model Fit & Forecast',
                mode='lines',
                line=dict(color='blue', width=2)
            ))
            
            # Add confidence intervals if available
            if 'yhat_lower' in line_data.columns and 'yhat_upper' in line_data.columns:
                # Filter out any NaN values that might exist
                ci_data = line_data.dropna(subset=['yhat_lower', 'yhat_upper'])
                
                if not ci_data.empty:
                    fig.add_trace(go.Scatter(
                        x=ci_data['ds'],
                        y=ci_data['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=ci_data['ds'],
                        y=ci_data['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(68, 68, 255, 0.2)',
                        name='95% Confidence Interval'
                    ))

            # Add last known price point highlighted
            last_known_price = df['y'].iloc[-1]
            last_known_date = df['ds'].iloc[-1]
            fig.add_trace(go.Scatter(
                x=[last_known_date],
                y=[last_known_price],
                name='Last Known Price',
                mode='markers',
                marker=dict(size=10, color='red')
            ))

            # Set the x-axis range to include the 2 extra days
            all_viz_dates = list(last_5_days['ds']) + list(line_data['ds']) + list(extra_dates)
            min_date = min(all_viz_dates)
            max_date = max(all_viz_dates)

            fig.update_layout(
                title=f'{stock_name} Stock Price - 12-Day Window<br>{model_name} Model',
                xaxis_title='Date',
                yaxis_title='Stock Price ($)',
                showlegend=True,
                xaxis=dict(
                    range=[min_date, max_date],
                    tickangle=45,
                    tickformat='%Y-%m-%d'
                )
            )

            if self.output_dir:
                # Create directory if it doesn't exist
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                    
                # Save plot
                model_prefix = model_name.lower()
                plot_path = os.path.join(
                    self.output_dir, 
                    f'{stock_name}_{model_prefix}_focused_forecast.png'
                )
                fig.write_image(plot_path)
                print(f"Focused plot saved: {plot_path}")
                
                # Also save with timestamp for historical reference
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plot_path_with_timestamp = os.path.join(
                    self.output_dir, 
                    f'{stock_name}_focused_forecast_{timestamp}.png'
                )
                fig.write_image(plot_path_with_timestamp)

            return fig
        except Exception as e:
            print(f"Error creating focused plot: {str(e)}")
            return None
            
    def plot_components(self, components_fig, stock_name, model_name):
        """
        Save the components plot (for Prophet model)
        
        Args:
            components_fig (plotly.graph_objects.Figure): The components figure from Prophet
            stock_name (str): Stock symbol
            model_name (str): Model name (should be 'Prophet')
            
        Returns:
            plotly.graph_objects.Figure: The input figure
        """
        if self.output_dir and components_fig:
            # Create directory if it doesn't exist
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                
            # Save plot with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            components_path = os.path.join(
                self.output_dir,
                f'{stock_name}_{model_name.lower()}_components_{timestamp}.png'
            )
            components_fig.write_image(components_path)
            print(f"Components plot saved: {components_path}")
            
            # Also save without timestamp (latest version)
            latest_path = os.path.join(
                self.output_dir,
                f'{stock_name}_{model_name.lower()}_components.png'
            )
            components_fig.write_image(latest_path)
            
        return components_fig
        
    def plot_distribution(self, df, stock_name, model_name):
        """
        Create and save a distribution plot of daily price changes.
        
        Args:
            df (pd.DataFrame): Historical data with 'ds' and 'y' columns
            stock_name (str): Stock symbol
            model_name (str): Model name
            
        Returns:
            plotly.graph_objects.Figure: The created figure
        """
        try:
            # Calculate daily returns
            daily_returns = df['y'].pct_change().dropna() * 100
            
            # Create histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=daily_returns,
                nbinsx=50,
                marker_color='blue',
                opacity=0.7
            ))
            
            fig.update_layout(
                title=f'{stock_name} - Distribution of Daily Price Changes',
                xaxis_title='Daily Return (%)',
                yaxis_title='Frequency',
                bargap=0.1
            )
            
            if self.output_dir:
                # Create directory if it doesn't exist
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                    
                # Save plot with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                dist_path = os.path.join(
                    self.output_dir,
                    f'{stock_name}_{model_name.lower()}_distribution_{timestamp}.png'
                )
                fig.write_image(dist_path)
                print(f"Distribution plot saved: {dist_path}")
                
                # Also save without timestamp (latest version)
                latest_path = os.path.join(
                    self.output_dir,
                    f'{stock_name}_{model_name.lower()}_distribution.png'
                )
                fig.write_image(latest_path)
                
            return fig
        except Exception as e:
            print(f"Error creating distribution plot: {str(e)}")
            return None
    
    def plot_grid_search_results(self, mse_history, stock_name, model_name):
        """
        Create and save a visualization of grid search results (for SARIMA).
        
        Args:
            mse_history (list): List of dictionaries with grid search results
            stock_name (str): Stock symbol
            model_name (str): Model name
            
        Returns:
            plotly.graph_objects.Figure: The created figure
        """
        try:
            # Convert to DataFrame
            df_history = pd.DataFrame(mse_history)
            
            # Sort by MSE (best models first)
            df_history = df_history.sort_values('mse')
            
            # Create a summary column that encodes the model parameters
            if model_name.upper() == 'SARIMA':
                df_history['model'] = df_history.apply(
                    lambda row: f"SARIMA({row['p']},{row['d']},{row['q']})x({row['P']},{row['D']},{row['Q']},{row['s']})",
                    axis=1
                )
            
            # Create a visualization of the top 10 models by MSE
            top_models = df_history.head(10)
            
            fig = go.Figure()
            
            # Add MSE bars
            fig.add_trace(go.Bar(
                x=top_models['model'],
                y=top_models['mse'],
                name='MSE',
                marker_color='blue'
            ))
            
            # Add AIC line on secondary axis if available
            if 'aic' in top_models.columns:
                fig.add_trace(go.Scatter(
                    x=top_models['model'],
                    y=top_models['aic'],
                    name='AIC',
                    mode='markers+lines',
                    marker=dict(color='red'),
                    yaxis='y2'
                ))
                
                # Update layout with dual y-axis
                fig.update_layout(
                    title=f'{stock_name} - {model_name} Grid Search Results (Top 10 Models)',
                    xaxis=dict(title='Model Parameters', tickangle=45),
                    yaxis=dict(title='MSE', side='left'),
                    yaxis2=dict(title='AIC', side='right', overlaying='y'),
                    legend=dict(x=0.7, y=1),
                    barmode='group',
                    height=600,
                    width=1000
                )
            else:
                # Update layout without secondary axis
                fig.update_layout(
                    title=f'{stock_name} - {model_name} Grid Search Results (Top 10 Models)',
                    xaxis=dict(title='Model Parameters', tickangle=45),
                    yaxis=dict(title='MSE'),
                    barmode='group',
                    height=600,
                    width=1000
                )
            
            if self.output_dir:
                # Create directory if it doesn't exist
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                    
                # Save plot with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                viz_path = os.path.join(
                    self.output_dir,
                    f'{stock_name}_{model_name.lower()}_grid_search_viz_{timestamp}.png'
                )
                fig.write_image(viz_path)
                print(f"Grid search visualization saved to: {viz_path}")
                
            return fig
        except Exception as e:
            print(f"Error creating grid search visualization: {str(e)}")
            return None 