import pandas as pd
import yfinance as yf
import os

def load_data():
    """
    Phase 1: Fetch and Clean Data
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, 'data', 'SPY.csv')
    
    # Check if the CSV already exists to avoid re-downloading, 
    # otherwise fetch it using yfinance as per instructions.
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        date_col = 'Date' if 'Date' in df.columns else 'Datetime'
        last_date = pd.to_datetime(df[date_col], utc=True).max()
        
        # Check if local data is up to date (accounting for weekends)
        today = pd.Timestamp.today(tz='UTC').normalize()
        
        if today.weekday() == 5:    # Saturday
            expected_date = today - pd.Timedelta(days=1)
        elif today.weekday() == 6:  # Sunday
            expected_date = today - pd.Timedelta(days=2)
        elif today.weekday() == 0:  # Monday
            expected_date = today - pd.Timedelta(days=3)
        else:
            expected_date = today - pd.Timedelta(days=1)

        if last_date >= expected_date:
            print("Local data is up to date. Skipping yfinance fetch.")
        else:
            print(f"Fetching new data from yfinance since {last_date.strftime('%Y-%m-%d')}...")
            
            spy = yf.Ticker("SPY")
            new_df = spy.history(start=last_date.strftime('%Y-%m-%d'))
            if not new_df.empty:
                new_df.reset_index(inplace=True)
                if 'Datetime' in new_df.columns:
                    new_df.rename(columns={'Datetime': 'Date'}, inplace=True)
                if 'Date' not in df.columns and 'Datetime' in df.columns:
                    df.rename(columns={'Datetime': 'Date'}, inplace=True)
                
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                new_df['Date'] = pd.to_datetime(new_df['Date'], utc=True)
                
                df = pd.concat([df, new_df], ignore_index=True)
                df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
                df.to_csv(file_path, index=False)
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