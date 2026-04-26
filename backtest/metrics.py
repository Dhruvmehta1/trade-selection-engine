import numpy as np
import pandas as pd

def compute_metrics(returns):
    """
    Step 8.1: Compute performance metrics for a given series of daily returns.
    """
    if len(returns) == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    # Total Return
    cum_ret = (1 + returns).cumprod()
    total_return = cum_ret.iloc[-1] - 1

    # Annualized Sharpe Ratio (assuming 252 trading days)
    mean_ret = returns.mean()
    std_ret = returns.std()
    if std_ret > 0:
        sharpe = (mean_ret / std_ret) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max Drawdown
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min()

    # Win Rate (Percentage of profitable active trading days)
    # We only count days where a return was generated (position != 0)
    active_returns = returns[returns != 0]
    if len(active_returns) > 0:
        win_rate = (active_returns > 0).mean()
    else:
        win_rate = 0.0

    return total_return, sharpe, max_dd, win_rate

def compare_strategies(backtest_df):
    """
    Phase 8: METRICS
    Step 8.2 & Checkpoint: Computes and compares metrics for Baseline vs ML strategy.
    """
    print("\n--- PHASE 8: Strategy Metrics Comparison ---")
    
    # Calculate metrics for both strategies
    base_ret, base_sharpe, base_mdd, base_win = compute_metrics(backtest_df['Baseline_Return'])
    ml_ret, ml_sharpe, ml_mdd, ml_win = compute_metrics(backtest_df['ML_Return'])
    
    # Print Comparison Table
    print("-" * 50)
    print(f"{'Metric':<20} | {'Baseline':<12} | {'ML Strategy':<12}")
    print("-" * 50)
    print(f"{'Total Return':<20} | {base_ret*100:>11.2f}% | {ml_ret*100:>11.2f}%")
    print(f"{'Sharpe Ratio':<20} | {base_sharpe:>12.2f} | {ml_sharpe:>12.2f}")
    print(f"{'Max Drawdown':<20} | {base_mdd*100:>11.2f}% | {ml_mdd*100:>11.2f}%")
    print(f"{'Win Rate':<20} | {base_win*100:>11.2f}% | {ml_win*100:>11.2f}%")
    print("-" * 50)
    
    return {
        'Baseline': {'Total Return': base_ret, 'Sharpe': base_sharpe, 'Max DD': base_mdd, 'Win Rate': base_win},
        'ML': {'Total Return': ml_ret, 'Sharpe': ml_sharpe, 'Max DD': ml_mdd, 'Win Rate': ml_win}
    }