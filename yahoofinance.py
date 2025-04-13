import yfinance as yf
import pandas as pd

# Define tickers and parameters
tickers = ['AAPL', 'NVDA']
period = '30d'           # Yahoo limit for 60m interval
interval = '60m'

# Download data
data = yf.download(
    tickers,
    period=period,
    interval=interval,
    group_by='ticker',
    auto_adjust=True
)

# Save to CSV
data.to_csv('hourly_stock_data.csv')

print("Hourly stock data saved to 'hourly_stock_data.csv'")
