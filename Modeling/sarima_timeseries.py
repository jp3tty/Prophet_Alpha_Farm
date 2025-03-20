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
        ts = pd.Series(self.df['y'].values, index=self.df['ds'])
        
        best_aic = float('inf')
        best_mse = float('inf')
        best_params = None
        self.mse_history = []
        
        total_combinations = (
            len(self.param_grid['p']) * len(self.param_grid['d']) * len(self.param_grid['q']) *
            len(self.param_grid['P']) * len(self.param_grid['D']) * len(self.param_grid['Q']) *
            len(self.param_grid['s'])
        )
        
        print(f"Running grid search with {total_combinations} parameter combinations...")
        combination_count = 0
        
        # Grid search through parameters
        for p in self.param_grid['p']:
            for d in self.param_grid['d']:
                for q in self.param_grid['q']:
                    for P in self.param_grid['P']:
                        for D in self.param_grid['D']:
                            for Q in self.param_grid['Q']:
                                for s in self.param_grid['s']:
                                    combination_count += 1
                                    param_combo = f"SARIMA{(p,d,q)}x{(P,D,Q,s)}"
                                    print(f"\nTrying {param_combo} - combination {combination_count}/{total_combinations}")
                                    
                                    try:
                                        model = SARIMAX(ts,
                                                      order=(p, d, q),
                                                      seasonal_order=(P, D, Q, s),
                                                      enforce_stationarity=False,
                                                      enforce_invertibility=False)
                                        results = model.fit(disp=False)
                                        
                                        # Make predictions for evaluation
                                        forecast = self.make_predictions(model=results)
                                        mse, rmse = self.calculate_mse(forecast, is_training=True)
                                        
                                        # Track this model's performance
                                        model_record = {
                                            'p': p, 'd': d, 'q': q, 
                                            'P': P, 'D': D, 'Q': Q, 's': s,
                                            'aic': results.aic,
                                            'mse': mse,
                                            'rmse': rmse
                                        }
                                        self.mse_history.append(model_record)
                                        
                                        # Use both AIC and MSE for model selection, with MSE as primary metric
                                        is_better = False
                                        reason = ""
                                        
                                        if mse < best_mse:
                                            is_better = True
                                            reason = "better MSE"
                                            best_mse = mse
                                            
                                        # If MSE is similar, use AIC as tiebreaker
                                        elif abs(mse - best_mse) < 0.01 and results.aic < best_aic:
                                            is_better = True
                                            reason = "similar MSE but better AIC"
                                            
                                        if is_better:
                                            best_aic = results.aic
                                            best_params = {
                                                'order': (p, d, q),
                                                'seasonal_order': (P, D, Q, s)
                                            }
                                            self.model = results
                                            print(f"✅ New best model ({reason}):")
                                            print(f"   Parameters: {param_combo}")
                                            print(f"   MSE: {mse:.4f}, RMSE: {rmse:.4f}, AIC: {results.aic:.2f}")
                                        else:
                                            print(f"   MSE: {mse:.4f}, RMSE: {rmse:.4f}, AIC: {results.aic:.2f} - Not better than current best")
                                            
                                    except Exception as e:
                                        print(f"❌ Error with {param_combo}: {str(e)}")
                                        continue
        
        if self.model is not None:
            print("\n==== Best model found ====")
            print(f"Parameters: SARIMA{best_params['order']}x{best_params['seasonal_order']}")
            print(f"Best MSE: {best_mse:.4f}")
            print(f"Best AIC: {best_aic:.2f}")
            self.save_model('SARIMA')
            return best_aic
        else:
            raise Exception("No valid model found during grid search")

    def make_predictions(self, periods=5, model=None):
        """Generate predictions for the specified number of periods."""
        if model is None:
            model = self.model
            
        if model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Generate future dates using business day frequency
        last_date = self.df['ds'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq='B'  # Business days
        )

        # Generate forecast
        forecast_values = model.forecast(periods)

        # Create forecast DataFrame with proper index
        self.forecast = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values
        })

        # Include historical data in the forecast for MSE calculation
        historical_dates = self.df['ds'].tolist()
        historical_fitted = model.fittedvalues
        
        # Create a complete forecast that includes both historical fitted values and future forecasts
        complete_forecast = pd.DataFrame({
            'ds': historical_dates + future_dates.tolist(),
            'yhat': np.concatenate([historical_fitted, forecast_values])
        })
        
        self.forecast = complete_forecast
        
        print(f"DEBUG - SARIMA: Generated forecast with {len(complete_forecast)} points")
        
        return self.forecast

    def calculate_mse(self, forecast_df, is_training=False):
        """Calculate Mean Squared Error for the forecast."""
        try:
            # Use the consistent approach similar to Prophet model
            comparison_df = forecast_df.merge(
                self.df[['ds', 'y']], 
                on='ds', 
                how='inner'
            )
            
            if len(comparison_df) == 0:
                print("No matching dates found for MSE calculation")
                return float('inf'), float('inf')
            
            squared_diff = (comparison_df['yhat'].values - comparison_df['y'].values) ** 2
            squared_diff = squared_diff[~np.isnan(squared_diff)]
            
            if len(squared_diff) == 0:
                print("No valid squared differences for MSE calculation")
                return float('inf'), float('inf')
            
            mse = np.mean(squared_diff)
            rmse = np.sqrt(mse)
            
            print(f"DEBUG - SARIMA: Number of points in comparison: {len(comparison_df)}, MSE: {mse:.4f}")
            
            if is_training:
                self.mse = mse
                self.rmse = rmse
            
            return mse, rmse
            
        except Exception as e:
            print(f"Error in calculate_mse: {str(e)}")
            return float('inf'), float('inf')

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
                mode='markers',
                marker=dict(color='black', size=6)
            ))

            # Plot forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat'],
                name='Forecast',
                mode='lines',
                line=dict(color='blue', width=2)
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
        """Create and save a focused plot of last 5 days and next 10 days, plus 2 extra days."""
        try:
            fig = go.Figure()
            
            # Get last 5 days of historical data
            last_5_days = self.df.tail(5)
            
            # Plot historical data as black dots
            fig.add_trace(go.Scatter(
                x=last_5_days['ds'],
                y=last_5_days['y'],
                name='Historical Data',
                mode='markers',
                marker=dict(color='black', size=8)
            ))

            # Last known date from historical data
            last_date = self.df['ds'].iloc[-1]
            
            # Get 10 days of business days for the forecast
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=10,
                freq='B'  # Business days
            )
            
            # Create a complete blue line that includes both historical fitted values and forecasts
            if self.model is not None:
                # For historical dates: get fitted values from the model for the last 5 days
                fitted_values = None
                if hasattr(self, 'results') and self.results is not None:
                    # Try to get fitted values from the fitted model
                    try:
                        # Get fitted values for the historical period
                        fitted_values = self.results.fittedvalues[-5:]
                    except:
                        pass
                
                # If we couldn't get fitted values from results, use the model to predict on historical data
                if fitted_values is None:
                    try:
                        # Use the last 5 days indices to predict historical values
                        fitted_values = self.model.predict(start=len(self.df)-5, end=len(self.df)-1)
                    except:
                        # If that fails, use the forecast values for those dates (less accurate)
                        fitted_indices = forecast_df[forecast_df['ds'].isin(last_5_days['ds'])]
                        if not fitted_indices.empty:
                            fitted_values = fitted_indices['yhat'].values
                        else:
                            # Default to the actual values if we can't get predictions
                            fitted_values = last_5_days['y'].values
                
                # Generate a fresh 10-day forecast
                forecast_values = self.model.forecast(10)
                
                # Combine historical fitted values with forecast
                all_dates = list(last_5_days['ds']) + list(forecast_dates)
                all_values = list(fitted_values) + list(forecast_values)
                
                # Create a DataFrame with all dates and values for the blue line
                line_data = pd.DataFrame({
                    'ds': all_dates,
                    'yhat': all_values
                })
            else:
                # Use the provided forecast DataFrame
                # Extract fitted values for historical period
                historical_fitted = forecast_df[forecast_df['ds'].isin(last_5_days['ds'])].copy()
                
                # Extract forecast values for future dates
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
                        'yhat': [future_forecast['yhat'].iloc[-1]] * missing_days if len(future_forecast) > 0 else [np.nan] * missing_days
                    })
                    
                    # Add the missing days
                    future_forecast = pd.concat([future_forecast, missing_df], ignore_index=True)
                
                # Combine historical and future data for a continuous line
                line_data = pd.concat([historical_fitted, future_forecast], ignore_index=True)
                
                # If we don't have historical fitted values, use actual values
                if len(historical_fitted) == 0:
                    historical_as_fitted = pd.DataFrame({
                        'ds': last_5_days['ds'],
                        'yhat': last_5_days['y']
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

            # Add last known price point highlighted
            last_known_price = self.df['y'].iloc[-1]
            last_known_date = self.df['ds'].iloc[-1]
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
                title=f'{self.stock_name} Stock Price - 12-Day Window<br>SARIMA Model',
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

    def save_mse_history(self):
        """Save the MSE history to a CSV file and create visualization."""
        if hasattr(self, 'mse_history') and self.mse_history and self.output_dir:
            # Convert to DataFrame
            df_history = pd.DataFrame(self.mse_history)
            
            # Sort by MSE (best models first)
            df_history = df_history.sort_values('mse')
            
            # Create a summary column that encodes the model parameters
            df_history['model'] = df_history.apply(
                lambda row: f"SARIMA({row['p']},{row['d']},{row['q']})x({row['P']},{row['D']},{row['Q']},{row['s']})",
                axis=1
            )
            
            # Save to CSV
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join(
                self.output_dir,
                f'{self.stock_name}_sarima_grid_search_{timestamp}.csv'
            )
            df_history.to_csv(csv_path, index=False)
            print(f"Grid search results saved to: {csv_path}")
            
            try:
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
                
                # Add AIC line on secondary axis
                fig.add_trace(go.Scatter(
                    x=top_models['model'],
                    y=top_models['aic'],
                    name='AIC',
                    mode='markers+lines',
                    marker=dict(color='red'),
                    yaxis='y2'
                ))
                
                # Update layout
                fig.update_layout(
                    title=f'{self.stock_name} - SARIMA Grid Search Results (Top 10 Models)',
                    xaxis=dict(title='Model Parameters', tickangle=45),
                    yaxis=dict(title='MSE', side='left'),
                    yaxis2=dict(title='AIC', side='right', overlaying='y'),
                    legend=dict(x=0.7, y=1),
                    barmode='group',
                    height=600,
                    width=1000
                )
                
                # Save the visualization
                viz_path = os.path.join(
                    self.output_dir,
                    f'{self.stock_name}_sarima_grid_search_viz_{timestamp}.png'
                )
                fig.write_image(viz_path)
                print(f"Grid search visualization saved to: {viz_path}")
                
            except Exception as e:
                print(f"Error creating grid search visualization: {str(e)}")
                
            return csv_path
        else:
            print("No MSE history available to save.")
            return None 