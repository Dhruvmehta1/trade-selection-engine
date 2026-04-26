import pandas as pd
import numpy as np
import os
import sys

def create_labels(df):
    """
    Phase 4: Label Generation
    Computes the trade outcome (5-day future return) and assigns a binary label.
    """
    # Step 4.1: Compute return over the next 5 days
    # We use shift(-5) to grab the closing price 5 days in the future.
    # Future Return = (Close in 5 days - Today's Close) / Today's Close
    df['Future_Return_5d'] = (df['Close'].shift(-5) - df['Close']) / df['Close']
    
    # Step 4.2: Assign Label
    # 1 if the future 5-day return is strictly greater than 0, else 0
    df['Label'] = np.where(df['Future_Return_5d'] > 0, 1, 0)
    
    # CRITICAL: The last 5 rows of our dataset cannot look 5 days into the future.
    # They will have NaN for 'Future_Return_5d'. We must drop them so they don't break the ML model.
    df.dropna(subset=['Future_Return_5d'], inplace=True)
    
    # Convert Label to integer to ensure clean ML inputs
    df['Label'] = df['Label'].astype(int)
    
    return df

if __name__ == "__main__":
    # Ensure Python can find the other project folders
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    from data.data_loader import load_data
    from features.feature_engineering import create_features
    from strategies.momentum import momentum_strategy
    
    print("Running pipeline up to Phase 4...")
    df = load_data()
    df = create_features(df)
    df = momentum_strategy(df)
    
    print("\n--- PHASE 4: Generating Labels ---")
    df = create_labels(df)
    
    print("\n--- Checkpoint: Alignment & Future Return ---")
    # Show how today's signal aligns with the future outcome
    print(df[['Close', 'Signal', 'Future_Return_5d', 'Label']].tail(15))
    
    print("\n--- Checkpoint: Label Distribution (All Days) ---")
    print(df['Label'].value_counts())
    
    print("\n--- Checkpoint: Label Distribution (Only on BUY Signals) ---")
    buy_signals = df[df['Signal'] == 1]
    if len(buy_signals) > 0:
        print(buy_signals['Label'].value_counts())
    else:
        print("No BUY signals found in this dataset.")