import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly

# Read and prepare the data
def prepare_data(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Rename columns to Prophet's required names
    df = df.rename(columns={'Date': 'ds', 'Price': 'y'})
    
    # Convert to datetime
    df['ds'] = pd.to_datetime(df['ds'])
    
    return df

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
def create_interactive_plots(model, forecast, df):
    # Create the main forecast plot
    fig_forecast = plot_plotly(model, forecast, xlabel='Date', ylabel='Stock Price ($)')
    fig_forecast.update_layout(title='Microsoft Stock Price Forecast',
                             showlegend=True)
    fig_forecast.show()
    
    # Create the components plot (trends, yearly seasonality, etc.)
    fig_components = plot_components_plotly(model, forecast)
    fig_components.show()
    
    # Create custom price change distribution plot
    returns = df['y'].pct_change().dropna()
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=returns, 
                                  nbinsx=50,
                                  name='Daily Returns',
                                  showlegend=True))
    fig_dist.update_layout(title='Distribution of Daily Price Changes',
                          xaxis_title='Daily Return',
                          yaxis_title='Frequency',
                          bargap=0.1)
    fig_dist.show()

# Main execution
def main():
    # Read your CSV file
    df = prepare_data('EGO_prices_20250115.csv')
    
    # Train the model
    model = train_prophet_model(df)
    
    # Make predictions
    forecast = make_predictions(model, periods=3)
    
    # Print the numerical results
    print_forecast_results(forecast)
    
    # Create and display interactive plots
    create_interactive_plots(model, forecast, df)
    
    return model, forecast

# Run the forecasting
if __name__ == "__main__":
    model, forecast = main()