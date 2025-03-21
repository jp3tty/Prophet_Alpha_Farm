import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
import glob

class StockPredictor:
    def __init__(self, model_path, model_type='Prophet', mse_value='N/A'):
        """Initialize predictor with path to saved model."""
        self.model_path = model_path
        self.model = None
        self.model_type = model_type
        self.mse_value = mse_value
        
    def load_model(self):
        """Load the saved model."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def make_predictions(self, days=5):
        """Generate predictions for specified number of days."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.model_type == 'Prophet':
            # Create future dataframe for Prophet
            future = self.model.make_future_dataframe(periods=days)
            
            # Make predictions
            forecast = self.model.predict(future)
            
            # Get only the future predictions
            future_forecast = forecast.tail(days)
            
            # Format results
            results = pd.DataFrame({
                'Date': future_forecast['ds'],
                'Predicted_Price': future_forecast['yhat'],
                'Lower_Bound': future_forecast['yhat_lower'],
                'Upper_Bound': future_forecast['yhat_upper']
            })
            
            return results
        else:
            raise ValueError(f"Prediction not implemented for model type: {self.model_type}")
    
    def print_predictions(self, predictions):
        """Print formatted prediction results."""
        print("\nStock Price Predictions:")
        print("-" * 80)
        print("Date           Predicted Price    Lower Bound    Upper Bound")
        print("-" * 80)
        
        for _, row in predictions.iterrows():
            date = row['Date'].strftime('%Y-%m-%d')
            print(f"{date}    ${row['Predicted_Price']:.2f}         ${row['Lower_Bound']:.2f}      ${row['Upper_Bound']:.2f}")

def main():
    try:
        # Get the absolute path to the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the Output directory with timestamped folders (using absolute path)
        project_root = os.path.dirname(script_dir)  # Parent directory of Predictions
        forecast_dir = os.path.join(project_root, "Output")
        
        print(f"Looking for model files in: {forecast_dir}")
        
        # Find the most recent timestamped directory
        timestamp_dirs = [d for d in os.listdir(forecast_dir) 
                         if os.path.isdir(os.path.join(forecast_dir, d)) 
                         and d[0].isdigit()]  # Filter directories that start with numbers (timestamps)
        
        if not timestamp_dirs:
            raise FileNotFoundError("No timestamped directories found in Output directory")
            
        # Sort directories by name (timestamp) in descending order
        latest_timestamp_dir = sorted(timestamp_dirs, reverse=True)[0]
        forecast_dir = os.path.join(forecast_dir, latest_timestamp_dir)
        print(f"Using latest output directory: {forecast_dir}")
        
        # Load MSE values from the model comparison summary file
        comparison_files = glob.glob(os.path.join(forecast_dir, "model_comparison_summary_*.csv"))
        mse_values = {}
        
        if comparison_files:
            # Get the latest comparison file
            latest_comparison = sorted(comparison_files)[-1]
            print(f"Found model comparison file: {latest_comparison}")
            
            # Read the comparison data
            comparison_df = pd.read_csv(latest_comparison)
            
            # Extract MSE values by stock and model
            for _, row in comparison_df.iterrows():
                stock = row['Stock']
                model = row['Model']
                mse = row['MSE']
                
                # Round MSE to 2 decimal places
                if isinstance(mse, (int, float)):
                    mse = round(mse, 2)
                
                # Store in dictionary with key format 'STOCK_MODEL'
                mse_values[f"{stock}_{model}"] = mse
                
            print(f"Loaded MSE values for {len(mse_values)} stock/model combinations")
        else:
            print("No model comparison file found, MSE values will be shown as N/A")
        
        # We'll focus on Prophet models only, which are in the Prophet subdirectory
        prophet_dir = os.path.join(forecast_dir, "Prophet")
        if not os.path.exists(prophet_dir):
            raise FileNotFoundError(f"Prophet directory not found in {forecast_dir}")
        
        # Find the latest .pkl file for each stock prefix
        latest_models = {}
        
        # Find all Prophet model pickle files
        for file in os.listdir(prophet_dir):
            if file.endswith('.pkl'):
                # Extract stock prefix (e.g., 'AAPL' from 'AAPL_Prophet_Model_20250321_081855.pkl')
                stock_prefix = file.split('_')[0]
                full_path = os.path.join(prophet_dir, file)
                
                # Update if this is the first or a later model for this stock
                if stock_prefix not in latest_models or \
                   os.path.getctime(full_path) > os.path.getctime(latest_models[stock_prefix]):
                    latest_models[stock_prefix] = full_path
        
        if not latest_models:
            raise FileNotFoundError(f"No .pkl files found in directory {prophet_dir}")
            
        # Create output directory for predictions if it doesn't exist
        predictions_dir = os.path.join(script_dir, "predictions_output")
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Lists to collect data for the new format
        stocks = []
        models = []
        mses = []
        all_dates = []
        all_predictions_by_stock = {}
        
        # Process predictions for each stock's latest model
        for stock_prefix, model_path in latest_models.items():
            print(f"\nProcessing predictions for {stock_prefix}:")
            
            # Get MSE value if available
            mse_key = f"{stock_prefix}_Prophet"
            mse_value = mse_values.get(mse_key, "N/A")
            
            predictor = StockPredictor(model_path, model_type='Prophet', mse_value=mse_value)
            predictor.load_model()
            predictions = predictor.make_predictions(days=5)
            
            # Round prediction values to 2 decimal places
            predictions['Predicted_Price'] = predictions['Predicted_Price'].round(2)
            predictions['Lower_Bound'] = predictions['Lower_Bound'].round(2)
            predictions['Upper_Bound'] = predictions['Upper_Bound'].round(2)
            
            # Store the dates if we haven't yet
            if not all_dates:
                all_dates = predictions['Date'].dt.strftime('%Y-%m-%d').tolist()
            
            # Get the stock information
            stocks.append(stock_prefix)
            models.append('Prophet')
            mses.append(mse_value)
            
            # Store predictions by date for this stock
            all_predictions_by_stock[stock_prefix] = predictions['Predicted_Price'].tolist()
            
            # Print results
            predictor.print_predictions(predictions)
        
        # Create a new DataFrame with the desired format
        new_format_data = []
        
        for i, stock in enumerate(stocks):
            row_data = {
                'Stock': stock,
                'Model': models[i],
                'MSE': mses[i]
            }
            
            # Add the predictions for each date
            for j, date in enumerate(all_dates):
                row_data[date] = all_predictions_by_stock[stock][j]
                
            new_format_data.append(row_data)
        
        # Convert to DataFrame
        new_format_df = pd.DataFrame(new_format_data)
        
        # Save to CSV with current date in filename
        current_date = datetime.now().strftime('%Y%m%d')
        output_file = os.path.join(predictions_dir, f"{current_date}_predictions.csv")
        
        # Format the DataFrame for CSV output
        # Create a copy of the DataFrame for CSV output to avoid modifying the display DataFrame
        csv_df = new_format_df.copy()
        
        # Save to CSV with formatting applied
        csv_df.to_csv(output_file, index=False, float_format='%.2f')
        print(f"\nAll predictions saved to {output_file}")
        
        # Also print the table to console
        print("\nPrediction Summary Table:")
        print("-" * 100)
        print(new_format_df.to_string(index=False, float_format=lambda x: f"${x:.2f}" if isinstance(x, float) else x))
        print("-" * 100)
        
        return 0
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)