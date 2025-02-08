# Prophet_Alpha_Farm

## Overview
This project evaluates Meta's Prophet library for stock price prediction. It implements an automated pipeline that collects stock price data, trains a Prophet model through hyperparameter optimization, and generates five-day price predictions.

## Project Structure
The repository is organized into four main directories:

### Data
Contains the CSV files storing historical stock price data collected from Yahoo Finance (yfinance).

### Program Workflow

1. **Data Collection** (`Explorations/price_collection.py`)
   - Prompts user to input stock symbols for analysis
   - Fetches 1 year of historical price data from Yahoo Finance for each symbol
   - Saves individual stock data as CSV files in `Data/price_data/` directory
   - Example output: `AAPL_prices.csv`, `GOOGL_prices.csv`

2. **Model Training** (`Modeling/timeseries_modeling.py`)
   - Reads stock price CSVs from `Data/price_data/`
   - For each stock:
     - Performs extensive hyperparameter optimization using grid search
     - Evaluates models using Mean Squared Error (MSE)
     - Saves the best performing model as a pickle file in `forecast_output/{timestamp}/`
     - Generates visualization plots of the champion model's predictions
   - Creates a consolidated results CSV with best parameters for each stock

3. **Price Prediction** (`Predictions/timeseries_prediction.py`)
   - Loads the latest champion model for each stock from `forecast_output/`
   - Generates 5-day price predictions including:
     - Predicted price
     - Upper and lower confidence bounds
   - Combines predictions for all stocks into a single CSV file
   - Saves results in `predictions_output/` with current date

### Output Files
- **Data Collection**: `Data/price_data/{SYMBOL}_prices.csv`
- **Model Training**: 
  - `forecast_output/{timestamp}/{SYMBOL}_Champion_{date}.pkl` (model file)
  - `forecast_output/{timestamp}/{SYMBOL}_Champion_{date}.png` (forecast plot)
  - `forecast_output/{timestamp}/forecast_results.csv` (consolidated results)
- **Predictions**: `predictions_output/{date}_predictions.csv`