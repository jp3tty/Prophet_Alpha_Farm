"""
Test Module for Theta Model Implementation

Provides testing functionality for the Theta time series model, including parameter
validation and performance evaluation.

Key Features:
- Model initialization testing
- Data preparation validation
- Parameter grid testing
- MSE calculation verification
- Visualization testing

Test Components:
- Data loading and preprocessing
- Model training and prediction
- Performance evaluation
- Plot generation
"""

from theta_timeseries import ThetaTimeSeriesModel
import os

def test_theta_model():
    # Set up paths
    data_dir = "Data"  # Updated path to Data directory
    output_dir = "Output/Theta"  # Updated path to Output directory
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_prices.csv')]
    
    for csv_file in csv_files:
        print(f"\nTesting Theta model on {csv_file}")
        
        # Initialize model
        csv_path = os.path.join(data_dir, csv_file)
        model = ThetaTimeSeriesModel(csv_path)
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
            
            # Calculate MSE using the best parameters found during training
            values = model.df['y'].values
            mse, rmse = model.calculate_mse(values, model.best_params)
            print(f"\nMSE: {mse:.2f}")
            print(f"RMSE: {rmse:.2f}")
            
            # Create plot
            model.plot_forecast(forecast)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

if __name__ == "__main__":
    test_theta_model() 