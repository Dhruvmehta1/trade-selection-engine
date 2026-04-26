import sys
import os
import pandas as pd
from features.feature_engineering import create_features
# Add the parent directory (project root) to sys.path to fix ModuleNotFoundError
# This allows Python to find the 'features' folder when running this script directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)



def momentum_strategy(df, window=20):
    """
    Applies the momentum breakout strategy to the DataFrame.
    """
    # 1. Calculate the N-day highest high and lowest low
    # We use .shift(1) so today's Close is compared to the PREVIOUS N days.
    # Without .shift(1), today's High would be included, making breakouts impossible.
    high_col = f'{window}_day_high'
    low_col = f'{window}_day_low'
    
    df[high_col] = df['High'].rolling(window=window).max().shift(1)
    df[low_col] = df['Low'].rolling(window=window).min().shift(1)
    
    # 2. Create the Signal column (default is 0: Hold)
    df['Signal'] = 0
    
    # 3. Generate BUY Signals (1)
    buy_condition = df['Close'] > df[high_col]
    df.loc[buy_condition, 'Signal'] = 1
    
    # 4. Generate SELL Signals (-1)
    sell_condition = df['Close'] < df[low_col]
    df.loc[sell_condition, 'Signal'] = -1    
    
    return df

if __name__ == '__main__':
    # Test block: only runs if this file is executed directly
    try:
        # Load the raw processed data
        df = pd.read_csv('../data/spy_processed.csv', index_col='Date', parse_dates=True)
        
        # 1. Push data through the feature engineering pipeline
        df = create_features(df)
        
        # 2. Push the updated data through the momentum strategy pipeline
        df = momentum_strategy(df)
        
        # Check the results
        print("Strategy applied successfully!\n")
        print(df[['Close', '20_day_high', '20_day_low', 'Signal']].tail(15))
        
        print("\nSignal Distribution:")
        print(df['Signal'].value_counts())
        
    except FileNotFoundError:
        print("Error: Could not find '../data/spy_processed.csv'.")
        print("Make sure you are running this script from inside the 'strategies' folder.")