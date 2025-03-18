from prophet_timeseries import ProphetTimeSeriesModel
import os
from tqdm import tqdm

def test_prophet_model(max_training_days=90):
    """Test Prophet model with specified amount of training data."""
    data_dir = "Data"
    output_dir = "Output/Prophet"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_prices.csv')]
    total_files = len(csv_files)
    
    # Create progress bar for overall progress
    pbar = tqdm(csv_files, desc="Processing stocks", unit="stock")
    
    for csv_file in pbar:
        pbar.set_description(f"Processing {csv_file}")
        
        csv_path = os.path.join(data_dir, csv_file)
        model = ProphetTimeSeriesModel(csv_path)
        model.output_dir = output_dir
        
        try:
            model.prepare_data(max_training_days=max_training_days)
            model.train_model()
            forecast = model.make_predictions(periods=5)
            
            # Calculate metrics
            train_mse, train_rmse = model.calculate_mse(forecast, is_training=True)
            pred_mse, pred_rmse = model.calculate_mse(forecast, is_training=False)
            
            # Create plot silently
            model.plot_forecast(forecast)
            
            # Update progress bar with metrics
            pbar.set_postfix({
                'Train RMSE': f'${train_rmse:.2f}',
                'Pred RMSE': f'${pred_rmse:.2f}'
            })
            
        except Exception as e:
            pbar.set_postfix({'Error': str(e)})
            continue

if __name__ == "__main__":
    test_prophet_model(max_training_days=90) 