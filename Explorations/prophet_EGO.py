import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly
import os
import numpy as np

# Read and prepare the data
def prepare_data(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Rename columns to Prophet's required names
    df = df.rename(columns={'Date': 'ds', 'Price': 'y'})
    
    # Convert to datetime
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Get stock name from CSV filename
    stock_name = os.path.splitext(os.path.basename(csv_path))[0].split('_')[0]
    
    return df, stock_name

# Train the Prophet model
def train_prophet_model(df):
    # Initialize and train the model
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,  # How flexible the trend is
        seasonality_prior_scale=10,    # How much seasonality to fit
    )
    
    model.fit(df)
    return model

# Make predictions
def make_predictions(model, periods=3):
    # Create future dates dataframe
    future_dates = model.make_future_dataframe(periods=periods)
    
    # Make predictions
    forecast = model.predict(future_dates)
    
    return forecast

# Calculate MSE and RMSE
def calculate_mse(forecast, df):
    # Merge forecast with actual data
    comparison_df = forecast.merge(
        df[['ds', 'y']], 
        on='ds', 
        how='inner'
    )
    
    # Calculate squared differences
    squared_diff = (comparison_df['yhat'].values - comparison_df['y'].values) ** 2
    
    # Calculate MSE and RMSE
    mse = np.mean(squared_diff)
    rmse = np.sqrt(mse)
    
    return mse, rmse

# Function to print the results
def print_forecast_results(forecast, periods=3):
    # Get the last few actual dates and the forecast dates
    forecast_subset = forecast.tail(periods)
    
    print("\nForecast for the next 3 days:")
    print("Date                      Predicted Price    Lower Bound    Upper Bound")
    print("-" * 75)
    
    for _, row in forecast_subset.iterrows():
        date = row['ds'].strftime('%Y-%m-%d')
        yhat = row['yhat']
        yhat_lower = row['yhat_lower']
        yhat_upper = row['yhat_upper']
        print(f"{date}           ${yhat:.2f}          ${yhat_lower:.2f}        ${yhat_upper:.2f}")

# Function to create interactive plots
def create_interactive_plots(model, forecast, df, stock_name):
    # Create the main forecast plot
    fig_forecast = plot_plotly(model, forecast, xlabel='Date', ylabel='Stock Price ($)')
    fig_forecast.update_layout(title=f'{stock_name} Stock Price Forecast',
                             showlegend=True)
    fig_forecast.show()
    
    # Create the components plot (trends, yearly seasonality, etc.)
    fig_components = plot_components_plotly(model, forecast)
    fig_components.update_layout(title=f'{stock_name} - Model Components')
    fig_components.show()
    
    # Create custom price change distribution plot
    returns = df['y'].pct_change().dropna()
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=returns, 
                                  nbinsx=50,
                                  name='Daily Returns',
                                  showlegend=True))
    fig_dist.update_layout(title=f'{stock_name} - Distribution of Daily Price Changes',
                          xaxis_title='Daily Return',
                          yaxis_title='Frequency',
                          bargap=0.1)
    fig_dist.show()

# Main execution
def main():
    # Get the path to the CSV file in the Modeling/Data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level to project root
    csv_path = os.path.join(project_root, 'Modeling', 'Data', 'EGO_prices.csv')
    
    # Read your CSV file
    df, stock_name = prepare_data(csv_path)
    
    # Train the model
    model = train_prophet_model(df)
    
    # Make predictions
    forecast = make_predictions(model, periods=3)
    
    # Calculate and print MSE
    mse, rmse = calculate_mse(forecast, df)
    print(f"\nModel Performance for {stock_name}:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Print the numerical results
    print_forecast_results(forecast)
    
    # Create and display interactive plots
    create_interactive_plots(model, forecast, df, stock_name)
    
    return model, forecast

# Run the forecasting
if __name__ == "__main__":
    model, forecast = main()