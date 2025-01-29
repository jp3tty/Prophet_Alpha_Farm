# Prophet_Alpha_Farm

## Overview
This project evaluates Meta's Prophet library for stock price prediction. It implements an automated pipeline that collects stock price data, trains a Prophet model through hyperparameter optimization, and generates five-day price predictions.

## Project Structure
The repository is organize into four main directories:

### Data
Contains the CSV files storing historical stock price data collected from Yahoo Finance (yfinance).

### Explorations
Contains price_collection.py, which:
- Interfaces with the yfinance API
- Collects historical stock price data
- Saves the data to CSV format in the Data directory

### Modeling
Contains timeseries_modeling.py, which:
- Implements a hyperparameter grid search
- Trains multiple Prophet models with different parameter combinations
- Identifies and saves the best performing model based on evaluation metrics

### Predictions
Contains timeseries_predictions.py, which:
- Loads the optimized Prophet model
- Generates 5-day price predictions for specified stocks
- Outputs prediction results