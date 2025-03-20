from sarima_timeseries import SarimaTimeSeriesModel
import os
import pandas as pd
import numpy as np

def test_sarima_grid_search():
    """Test function specifically focused on validating the SARIMA grid search functionality."""
    # Set up paths
    data_dir = "Data"
    output_dir = "Output/SARIMA"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Use a specific stock for testing (EGO has shown good results previously)
    csv_file = "EGO_prices.csv"
    csv_path = os.path.join(data_dir, csv_file)
    
    print(f"\n==== Testing SARIMA model grid search on {csv_file} ====\n")
    
    # Initialize model
    model = SarimaTimeSeriesModel(csv_path)
    model.output_dir = output_dir
    
    # Extend the parameter grid for more thorough testing
    model.param_grid = {
        'p': [0, 1, 2],     # Autoregressive order
        'd': [0, 1],         # Differencing order
        'q': [0, 1],         # Moving average order
        'P': [0, 1],         # Seasonal autoregressive order
        'D': [0, 1],         # Seasonal differencing order
        'Q': [0, 1],         # Seasonal moving average order
        's': [5, 7]          # Seasonal period (weekly, biweekly)
    }
    
    # Track models and their performance
    model_records = []
    
    try:
        # Prepare data
        model.prepare_data()
        print("Data prepared successfully")
        
        # Extend train_model to track model performance
        original_train_model = model.train_model
        
        def train_model_with_tracking(*args, **kwargs):
            """Wrapper around train_model to track models and MSE values"""
            if not hasattr(model, 'mse_history'):
                model.mse_history = []
            
            # Call the original train method
            result = original_train_model(*args, **kwargs)
            
            # Return the result
            return result
            
        # Replace the method
        model.train_model = train_model_with_tracking
        
        # Train model
        print("\nStarting grid search for optimal SARIMA parameters...")
        best_aic = model.train_model()
        print("\nModel grid search completed successfully")
        
        # Save the MSE history
        print("\nSaving grid search results...")
        model.save_mse_history()
        
        # Make predictions with the best model
        forecast = model.make_predictions(periods=10)
        print("\nForecast for next 10 days:")
        print(forecast.tail(10))
        
        # Calculate final MSE
        mse, rmse = model.calculate_mse(forecast)
        print(f"\nFinal MSE: {mse:.4f}")
        print(f"Final RMSE: {rmse:.4f}")
        
        # Create and save forecast plots
        model.plot_forecast(forecast)
        model.plot_focused_forecast(forecast)
        
        print("\nSARIMA model testing completed successfully!")
        
    except Exception as e:
        print(f"Error during SARIMA grid search test: {str(e)}")

if __name__ == "__main__":
    test_sarima_grid_search() 