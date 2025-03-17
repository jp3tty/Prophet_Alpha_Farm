from prophet_timeseries import ProphetTimeSeriesModel
import os

def test_prophet_model(max_training_days=90):  # Default to 90 days of training data
    """Test Prophet model with specified amount of training data.
    
    Args:
        max_training_days: Number of days of historical data to use for training
    """
    # Set up paths
    data_dir = "Data"
    output_dir = "Output/Prophet"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_prices.csv')]
    
    for csv_file in csv_files:
        print(f"\nTesting Prophet model on {csv_file}")
        print(f"Using {max_training_days} days for training")
        
        # Initialize model
        csv_path = os.path.join(data_dir, csv_file)
        model = ProphetTimeSeriesModel(csv_path)
        model.output_dir = output_dir
        
        try:
            # Prepare data with specified training window
            model.prepare_data(max_training_days=max_training_days)
            print("Data prepared successfully")
            
            # Train model
            model.train_model()
            print("Model trained successfully")
            
            # Make predictions
            forecast = model.make_predictions(periods=5)
            print("\nForecast for next 5 days:")
            print(forecast.tail())
            
            # Calculate training MSE (using last 30 days)
            train_mse, train_rmse = model.calculate_mse(forecast, is_training=True)
            print(f"\nTraining Metrics (last 30 days):")
            print(f"MSE: {train_mse:.2f}")
            print(f"RMSE: ${train_rmse:.2f}")
            
            # Calculate prediction MSE (if we have actual data to compare)
            pred_mse, pred_rmse = model.calculate_mse(forecast, is_training=False)
            print(f"\nPrediction Metrics:")
            print(f"MSE: {pred_mse:.2f}")
            print(f"RMSE: ${pred_rmse:.2f}")
            
            # Create plot
            model.plot_forecast(forecast)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

if __name__ == "__main__":
    # You can change the training window here
    test_prophet_model(max_training_days=90)  # Using 90 days by default 