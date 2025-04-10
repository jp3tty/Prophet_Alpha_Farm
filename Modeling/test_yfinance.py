import yfinance as yf

# Test with AAPL
ticker = yf.Ticker("AAPL")
data = ticker.history(period="1d")
print("AAPL data:")
print(data) 