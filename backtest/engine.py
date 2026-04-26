import pandas as pd
import matplotlib.pyplot as plt

def run_backtest(df, X_test, ml_signals, transaction_cost=0.001, delay=1):
    """
    Phase 7: BACKTEST ENGINE
    Simulates trading the Baseline vs ML-Filtered strategy and plots the equity curves.
    Accepts 'delay' parameter for Phase 9 stress testing.
    """
    print("\n--- PHASE 7: Running Backtest Engine ---")
    
    # 1. Isolate the exact dates used in the testing phase
    test_dates = X_test.index
    backtest_df = df.loc[test_dates].copy()
    
    # 2. Add our ML Signals to this dataframe
    backtest_df['ML_Signal'] = ml_signals
    
    # 3. Define Positions (Shifted by 'delay' to prevent look-ahead bias and simulate execution lag)
    # Baseline takes all Momentum signals (1 for Buy, -1 for Sell)
    backtest_df['Baseline_Position'] = backtest_df['Signal'].shift(delay).fillna(0)
    
    # ML Strategy only takes the trade if Baseline says BUY (1) AND ML says GOOD (1)
    # If ML says 0, the position becomes 0 (Hold/Cash)
    ml_position = (backtest_df['Signal'] == 1) & (backtest_df['ML_Signal'] == 1)
    backtest_df['ML_Position'] = ml_position.astype(int).shift(delay).fillna(0)
    
    # --- Step 7.1 & 7.2: Calculate Daily Returns ---
    backtest_df['Baseline_Return'] = backtest_df['Baseline_Position'] * backtest_df['Daily_Return']
    backtest_df['ML_Return'] = backtest_df['ML_Position'] * backtest_df['Daily_Return']
    
    # --- Step 7.3: Apply Transaction Costs (0.1% per trade) ---
    # We pay a cost every time our position changes
    baseline_trades = backtest_df['Baseline_Position'].diff().abs()
    ml_trades = backtest_df['ML_Position'].diff().abs()
    
    backtest_df['Baseline_Return'] -= (baseline_trades * transaction_cost)
    backtest_df['ML_Return'] -= (ml_trades * transaction_cost)
    
    # --- Step 7.4: Calculate Cumulative Equity Curve ---
    # Start with 1.0 (representing 100% or $1) and compound the returns
    backtest_df['Baseline_Equity'] = (1 + backtest_df['Baseline_Return']).cumprod()
    backtest_df['ML_Equity'] = (1 + backtest_df['ML_Return']).cumprod()
    
    # --- CHECKPOINT: Plot Baseline vs ML ---
    plot_equity_curves(backtest_df)
    
    return backtest_df

def plot_equity_curves(backtest_df):
    """Helper function to plot the backtest results"""
    plt.figure(figsize=(12, 6))
    
    plt.plot(backtest_df.index, backtest_df['Baseline_Equity'], label='Baseline (Mechanical Only)', color='gray')
    plt.plot(backtest_df.index, backtest_df['ML_Equity'], label='ML Filtered Strategy', color='blue')
    
    plt.title("Phase 7: Backtest Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns (1.0 = 100%)")
    plt.legend()
    plt.grid(True)
    
    # Save the plot so we can view it
    plt.savefig("backtest_results.png")
    print("-> Plot saved as 'backtest_results.png'")