import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Read and prepare data
df = pd.read_csv('price_data\MSFT_prices_20250108.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 1. Descriptive Statistics
desc_stats = df['Price'].describe()
skewness = df['Price'].skew()
kurtosis = df['Price'].kurtosis()
daily_returns = df['Price'].pct_change()

# Calculate rolling statistics
rolling_mean = df['Price'].rolling(window=20).mean()
rolling_std = df['Price'].rolling(window=20).std()

# 2. Time Series Decomposition
decomposition = seasonal_decompose(df['Price'], period=21)  # 21 trading days = 1 month

# 3. Stationarity Test (Augmented Dickey-Fuller)
adf_result = adfuller(df['Price'].dropna())

# 4. Calculate additional metrics
volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
autocorr = df['Price'].autocorr()

# Print results
print("\nDescriptive Statistics:")
print("======================")
print(f"Mean: ${desc_stats['mean']:.2f}")
print(f"Median: ${desc_stats['50%']:.2f}")
print(f"Standard Deviation: ${desc_stats['std']:.2f}")
print(f"Minimum: ${desc_stats['min']:.2f}")
print(f"Maximum: ${desc_stats['max']:.2f}")
print(f"\nSkewness: {skewness:.3f}")
print(f"Kurtosis: {kurtosis:.3f}")
print(f"Annualized Volatility: {volatility:.3f}")
print(f"Autocorrelation: {autocorr:.3f}")

print("\nAugmented Dickey-Fuller Test:")
print("============================")
print(f"ADF Statistic: {adf_result[0]:.3f}")
print(f"p-value: {adf_result[1]:.3f}")

# Create visualizations
plt.figure(figsize=(15, 10))

# Original price series with rolling statistics
plt.subplot(311)
plt.plot(df.index, df['Price'], label='Original')
plt.plot(df.index, rolling_mean, label='Rolling Mean (20 days)')
plt.plot(df.index, rolling_std + rolling_mean, label='Mean + Std')
plt.plot(df.index, rolling_mean - rolling_std, label='Mean - Std')
plt.title('MSFT Price with Rolling Statistics')
plt.legend()

# Decomposition plots
plt.subplot(312)
plt.plot(decomposition.trend)
plt.title('Trend')

plt.subplot(313)
plt.plot(decomposition.seasonal)
plt.title('Seasonal')

plt.tight_layout()
plt.show()

# Distribution analysis
plt.figure(figsize=(15, 5))

plt.subplot(121)
sns.histplot(df['Price'], kde=True)
plt.title('Price Distribution')

plt.subplot(122)
sns.histplot(daily_returns.dropna(), kde=True)
plt.title('Daily Returns Distribution')

plt.tight_layout()
plt.show()

# Calculate quartiles and IQR
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1

print("\nQuartile Analysis:")
print("=================")
print(f"Q1 (25th percentile): ${Q1:.2f}")
print(f"Q3 (75th percentile): ${Q3:.2f}")
print(f"IQR: ${IQR:.2f}")

# Calculate month-over-month changes
monthly_returns = df['Price'].resample('M').last().pct_change()

print("\nMonthly Statistics:")
print("=================")
print(f"Average Monthly Return: {monthly_returns.mean()*100:.2f}%")
print(f"Monthly Return Std Dev: {monthly_returns.std()*100:.2f}%")