from prophet_timeseries import ProphetTimeSeriesModel
from sarima_timeseries import SarimaTimeSeriesModel
from theta_timeseries import ThetaTimeSeriesModel

def run_comparison(csv_path, output_dir):
    models = [
        ProphetTimeSeriesModel(csv_path),
        SarimaTimeSeriesModel(csv_path),
        ThetaTimeSeriesModel(csv_path)
    ]

    results = {}
    for model in models:
        model.output_dir = output_dir
        model.prepare_data()
        model.train_model()
        forecast = model.make_predictions(periods=5)
        mse, rmse = model.calculate_mse(forecast)
        model.plot_forecast(forecast)
        
        results[model.__class__.__name__] = {
            'forecast': forecast,
            'mse': mse,
            'rmse': rmse
        }

    return results 