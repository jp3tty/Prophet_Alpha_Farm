"""
Test script to validate that our visualization module works correctly with all three models.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from visualizations import TimeSeriesPlotter
from prophet_timeseries import ProphetTimeSeriesModel
from sarima_timeseries import SarimaTimeSeriesModel
from theta_timeseries import ThetaTimeSeriesModel

def test_visualizations():
    """
    Test the TimeSeriesPlotter class with sample data for all three models.
    """
    # Create output directory
    output_dir = os.path.join('Output_Testing', 'Viz_Test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample data
    dates = pd.date_range(start='2022-01-01', periods=100, freq='B')
    y = np.sin(np.linspace(0, 5, 100)) * 10 + 50 + np.random.normal(0, 1, 100)
    
    # Create DataFrame with 'ds' and 'y' columns
    df = pd.DataFrame({
        'ds': dates,
        'y': y
    })
    
    # Create sample forecast data (10 days into future)
    future_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=10, freq='B')
    forecast_values = np.sin(np.linspace(5, 6, 10)) * 10 + 50
    lower_bound = forecast_values - 2
    upper_bound = forecast_values + 2
    
    # Add historical fitted values
    fitted_values = y + np.random.normal(0, 0.5, 100)
    
    # Create complete forecast DataFrame
    forecast_df = pd.DataFrame({
        'ds': list(dates) + list(future_dates),
        'yhat': list(fitted_values) + list(forecast_values),
        'yhat_lower': list(fitted_values - 1) + list(lower_bound),
        'yhat_upper': list(fitted_values + 1) + list(upper_bound)
    })
    
    # Initialize plotter
    plotter = TimeSeriesPlotter(output_dir=output_dir)
    
    # Test with each model name
    for model_name in ['Prophet', 'SARIMA', 'Theta']:
        print(f"Testing {model_name} visualization...")
        
        # Plot main forecast
        plotter.plot_forecast(
            df=df,
            forecast_df=forecast_df,
            stock_name='TEST',
            model_name=model_name,
            mse=1.234
        )
        
        # Plot focused forecast
        plotter.plot_focused_forecast(
            df=df,
            forecast_df=forecast_df,
            stock_name='TEST',
            model_name=model_name
        )
        
        # Plot distribution
        plotter.plot_distribution(df, 'TEST', model_name)
        
    print("All visualizations completed successfully!")

def test_with_real_models():
    """
    Test the integration of the visualization module with actual model implementations.
    """
    print("\nStarting integration tests...")
    
    # Create output directories
    test_data_dir = os.path.join('..', 'Data', 'Test')
    os.makedirs(test_data_dir, exist_ok=True)
    
    output_dir = os.path.join('Output_Testing', 'Viz_Integration_Test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample stock data
    dates = pd.date_range(start='2022-01-01', periods=200, freq='B')
    y = np.sin(np.linspace(0, 10, 200)) * 5 + 50 + np.random.normal(0, 1, 200) + np.linspace(0, 10, 200)
    
    # Create DataFrame with 'ds' and 'y' columns
    df = pd.DataFrame({
        'ds': dates,
        'y': y
    })
    
    # Save data to CSV
    test_csv_path = os.path.join(test_data_dir, 'TEST_stock_data.csv')
    df.to_csv(test_csv_path, index=False)
    print(f"Sample data saved to {test_csv_path}")
    
    # Test Prophet model
    try:
        print("Testing Prophet model integration...")
        prophet_model = ProphetTimeSeriesModel(test_csv_path)
        prophet_model.output_dir = output_dir
        prophet_model.prepare_data()  # Prepare the data
        prophet_model.train_model()  # Train the model
        prophet_forecast = prophet_model.make_predictions(periods=10)
        prophet_model.plot_forecast(prophet_forecast)
        print("Prophet model test complete!")
    except Exception as e:
        print(f"Prophet model test failed: {str(e)}")
    
    # Test SARIMA model
    try:
        print("Testing SARIMA model integration...")
        sarima_model = SarimaTimeSeriesModel(test_csv_path)
        sarima_model.output_dir = output_dir
        sarima_model.prepare_data()  # Prepare the data
        
        # Use a simple model for testing - SARIMA uses grid search by default
        sarima_model.train_model(use_grid_search=False)
        sarima_forecast = sarima_model.make_predictions(periods=10)
        sarima_model.plot_forecast(sarima_forecast)
        print("SARIMA model test complete!")
    except Exception as e:
        print(f"SARIMA model test failed: {str(e)}")
    
    # Test Theta model
    try:
        print("Testing Theta model integration...")
        theta_model = ThetaTimeSeriesModel(test_csv_path)
        theta_model.output_dir = output_dir
        theta_model.prepare_data()  # Prepare the data
        
        # Simple model training
        theta_model.train_model()  # Use default parameters
        theta_forecast = theta_model.make_predictions(periods=10)
        theta_model.plot_forecast(theta_forecast)
        print("Theta model test complete!")
    except Exception as e:
        print(f"Theta model test failed: {str(e)}")
    
    print("Integration tests completed!")

if __name__ == "__main__":
    print("Starting visualization tests...")
    test_visualizations()
    print("\nStarting model integration tests...")
    test_with_real_models()
    print("\nAll tests completed!") 