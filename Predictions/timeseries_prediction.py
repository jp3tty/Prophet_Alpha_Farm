import pandas as pd
import pickle
from datetime import datetime, timedelta
import os

class StockPredictor:
    def __init__(self, model_path):
        """Initialize predictor with path to saved model."""
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """Load the saved Prophet model."""
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
            
        # Create future dataframe
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
    
    def print_predictions(self, predictions):
        """Print formatted prediction results."""
        print("\nStock Price Predictions:")
        print("-" * 80)
        print("Date           Predicted Price    Lower Bound    Upper Bound")
        print("-" * 80)
        
        for _, row in predictions.iterrows():
            date = row['Date'].strftime('%Y-%m-%d')
            print(f"{date}    ${row['Predicted_Price']:,.2f}         ${row['Lower_Bound']:,.2f}      ${row['Upper_Bound']:,.2f}")

def main():
    try:
        # Path to your saved model
        # Get the latest .pkl file for each stock prefix from the forecast output directory
        forecast_dir = "../Modeling/forecast_output"
        latest_models = {}
        
        # Walk through all directories
        for root, dirs, files in os.walk(forecast_dir):
            for file in files:
                if file.endswith('.pkl'):
                    # Extract stock prefix (e.g., 'EGO' from 'EGO_Model13.pkl')
                    stock_prefix = file.split('_')[0]
                    full_path = os.path.join(root, file)
                    
                    # Update if this is the first or a later model for this stock
                    if stock_prefix not in latest_models or \
                       os.path.getctime(full_path) > os.path.getctime(latest_models[stock_prefix]):
                        latest_models[stock_prefix] = full_path
        
        if not latest_models:
            raise FileNotFoundError("No .pkl files found in forecast output directory")
            
        # Create output directory for predictions if it doesn't exist
        predictions_dir = "predictions_output"
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Initialize a dictionary to store all predictions
        all_predictions = {}
        first_dates = None
        
        # Process predictions for each stock's latest model
        for stock_prefix, model_path in latest_models.items():
            print(f"\nProcessing predictions for {stock_prefix}:")
            
            predictor = StockPredictor(model_path)
            predictor.load_model()
            predictions = predictor.make_predictions(days=5)
            
            # Store the dates if this is the first stock
            if first_dates is None:
                first_dates = predictions['Date']
                all_predictions['Date'] = first_dates
            
            # Add this stock's predictions to the combined dictionary
            all_predictions[stock_prefix] = predictions['Predicted_Price']
            
            # Print results
            predictor.print_predictions(predictions)
        
        # Create combined DataFrame
        combined_df = pd.DataFrame(all_predictions)
        
        # Save to CSV with current date in filename
        current_date = datetime.now().strftime('%Y%m%d')
        output_file = os.path.join(predictions_dir, f"{current_date}_predictions.csv")
        combined_df.to_csv(output_file, index=False)
        print(f"\nAll predictions saved to {output_file}")
        
        return 0
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)