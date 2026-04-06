import pandas as pd
import yfinance as yf
import os

def load_data():
    """
    Phase 1: Fetch and Clean Data
    """
    file_path = 'data/SPY.csv'
    
    # Check if the CSV already exists to avoid re-downloading, 
    # otherwise fetch it using yfinance as per instructions.
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        print("Fetching data from yfinance...")
        spy = yf.Ticker("SPY")
        df = spy.history(period="max")
        df.reset_index(inplace=True)
        # Save raw data for future use
        df.to_csv(file_path, index=False)

    # Clean Data
    # 1. Format Date
    # Rename 'Date' if yfinance returned 'Datetime' or similar
    if 'Date' not in df.columns and 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
        
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
    df.set_index('Date', inplace=True)
    
    # 2. Keep specific columns
    cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Ensure all columns exist (yfinance sometimes capitalizes them)
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols]
    
    # 3. Sort by date
    df.sort_index(inplace=True)
    
    # 4. Remove NaNs
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    df = load_data()
    print("--- First 5 Rows ---")
    print(df.head())
    print("\n--- Last 5 Rows ---")
    print(df.tail())