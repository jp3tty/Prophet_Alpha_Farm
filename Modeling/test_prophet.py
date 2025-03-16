from prophet_timeseries import ProphetTimeSeriesModel
import os

def test_prophet_model():
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
        
        # Initialize model
        csv_path = os.path.join(data_dir, csv_file)
        model = ProphetTimeSeriesModel(csv_path)
        model.output_dir = output_dir
        
        try:
            # Prepare data
            model.prepare_data()
            print("Data prepared successfully")
            
            # Train model
            model.train_model()
            print("Model trained successfully")
            
            # Make predictions
            forecast = model.make_predictions(periods=5)
            print("\nForecast for next 5 days:")
            print(forecast)
            
            # Calculate MSE
            mse, rmse = model.calculate_mse(forecast)
            print(f"\nMSE: {mse:.2f}")
            print(f"RMSE: {rmse:.2f}")
            
            # Create plot
            model.plot_forecast(forecast)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

if __name__ == "__main__":
    test_prophet_model() 