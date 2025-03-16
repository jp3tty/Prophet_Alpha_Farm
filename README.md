# Time Series Analysis System

A comprehensive system for time series forecasting using multiple models including Prophet, SARIMA, and Theta methods.

## System Schema

```
Modeling/
├── base_timeseries.py      # Abstract base class for all time series models
├── prophet_timeseries.py   # Facebook Prophet implementation
├── sarima_timeseries.py    # SARIMA model implementation
├── theta_timeseries.py     # Theta method implementation
├── main.py                # Main execution and orchestration
└── test_theta.py          # Test module for Theta model

Data/                      # Input data directory
└── *_prices.csv          # Stock price data files

Output/                    # Output directory
├── Prophet/              # Prophet model outputs
├── SARIMA/               # SARIMA model outputs
└── Theta/                # Theta model outputs
```

## Model Components

### Base Model (`base_timeseries.py`)
- Abstract base class defining the common interface
- Shared functionality for data preparation and validation
- Standardized methods for model training and prediction
- Common plotting and evaluation utilities

### Prophet Model (`prophet_timeseries.py`)
- Facebook Prophet implementation
- Grid search optimization
- Automatic seasonality detection
- Holiday effect modeling
- Trend changepoint detection
- Uncertainty interval estimation

### SARIMA Model (`sarima_timeseries.py`)
- Seasonal ARIMA implementation
- Grid search for optimal parameters
- Automatic differencing
- Seasonal pattern modeling
- Log transformation support
- AIC-based model selection

### Theta Model (`theta_timeseries.py`)
- Theta method implementation
- Grid search optimization
- Multiplicative decomposition
- Polynomial trend fitting
- Seasonal pattern detection
- Price movement constraints

## Main Execution (`main.py`)
- Orchestrates model execution
- Parallel processing support
- Comprehensive model comparison
- Performance metrics calculation
- Visualization generation
- CSV report generation

## Performance Metrics
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- Last Known Price
- Forecast Mean
- Forecast Standard Deviation

## Usage

### 1. Data Collection
First, collect historical stock price data using the data collection script:
```bash
python Explorations/price_collection.py
```
- The script will prompt you to enter stock symbols (e.g., AAPL, GOOGL)
- It will fetch 1 year of historical price data from Yahoo Finance
- Data will be saved as CSV files in the `Data/` directory
- Example output: `AAPL_prices.csv`, `GOOGL_prices.csv`

### 2. Model Execution
Run the main analysis script:
```bash
python Modeling/main.py
```
This will:
- Process all stock price files in the `Data/` directory
- Train and evaluate all models (Prophet, SARIMA, Theta)
- Generate forecasts and performance metrics
- Create visualization plots

### 3. View Results
Check the `Output/` directory for:
- Model-specific forecasts
- Performance metrics
- Visualization plots
- Comparison reports

## Dependencies
- pandas
- numpy
- prophet
- statsmodels
- plotly
- tqdm
- yfinance (for data collection)

## Output Structure
Each model generates:
1. Full forecast plot (60 days historical + 5 days forecast)
2. Focused plot (last 5 days + next 5 days)
3. Performance metrics
4. Forecast values

## Notes
- All models implement grid search for optimal parameters
- Price movement constraints are applied to prevent unrealistic forecasts
- Parallel processing is used for efficient model execution
- Comprehensive error handling and logging