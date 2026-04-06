import yfinance as yf
import pandas as pd

df = pd.read_csv('SPY.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
print(df.head)

df.to_csv('spy_processed.csv')
