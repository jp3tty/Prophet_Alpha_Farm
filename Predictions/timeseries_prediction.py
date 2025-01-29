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
        model_path = "../Modeling/forecast_output/prophet_model_20250122_133107_7cd63455.pkl"  # Update with your model filename
        
        # Initialize predictor
        predictor = StockPredictor(model_path)
        
        # Load model
        predictor.load_model()
        
        # Make predictions
        predictions = predictor.make_predictions(days=5)
        
        # Print results
        predictor.print_predictions(predictions)
        
        return 0
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)