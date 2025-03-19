from prophet_timeseries import ProphetTimeSeriesModel
import os
from tqdm import tqdm

def test_prophet_model():
    """Test Prophet model using all available historical data and grid search for optimal parameters."""
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
        model.baseline_mse = 15.0  # Confirm baseline MSE is set
        
        try:
            # Prepare data with all available history
            model.prepare_data()
            
            # This will perform grid search using the param_grid defined in ProphetTimeSeriesModel
            # It will try to find parameters that yield MSE < 15.0
            best_mse = model.train_model()
            
            if model.model is not None:
                # If a suitable model was found with MSE < 15.0
                forecast = model.make_predictions(periods=5)
                
                # Calculate metrics
                train_mse, train_rmse = model.calculate_mse(forecast, is_training=True)
                pred_mse, pred_rmse = model.calculate_mse(forecast, is_training=False)
                
                # Create plot silently
                model.plot_forecast(forecast)
                
                # Update progress bar with metrics
                pbar.set_postfix({
                    'Best MSE': f'${best_mse:.2f}',
                    'Train RMSE': f'${train_rmse:.2f}',
                    'Pred RMSE': f'${pred_rmse:.2f}'
                })
                
                print(f"\nResults for {model.stock_name}:")
                print(f"Best parameters: {model.best_params}")
                print(f"Best MSE: {best_mse:.4f}")
            else:
                print(f"\nNo model found for {csv_file} that beats baseline MSE of 15.0")
            
        except Exception as e:
            pbar.set_postfix({'Error': str(e)})
            print(f"\nError processing {csv_file}: {str(e)}")
            continue

if __name__ == "__main__":
    test_prophet_model() 