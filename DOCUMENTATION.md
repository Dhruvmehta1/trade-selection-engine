# System Documentation

## Data Pipeline

### Data Source
- Yahoo Finance (via yfinance)

### Assets
- SPY (or any liquid ETF)

### Frequency
- Daily OHLCV data

---

## Feature Engineering

Features are generated BEFORE signal evaluation to avoid leakage.

Examples:

### Price-based
- Returns (1d, 5d, 10d)
- Rolling mean
- Rolling std

### Technical Indicators
- RSI (14)
- MACD
- Bollinger Bands

### Volatility
- ATR
- Rolling volatility

---

## Strategy Layer

### Momentum Strategy
Signal:
- Buy when price > 20-day high
- Sell when price < 20-day low

### Mean Reversion
Signal:
- Buy when RSI < 30
- Sell when RSI > 70

Each signal generates:
- entry date
- exit condition
- return

---

## Label Generation

For each trade:
- Label = 1 if trade return > 0
- Label = 0 otherwise

This becomes the ML target.

---

## ML Model

### Model Type
- LightGBM / RandomForest

### Input
- Features at trade entry

### Output
- Probability of profitable trade

### Decision Rule
- Take trade if probability > threshold (e.g., 0.55)

---

## Backtesting Engine

Tracks:
- Trade execution
- Position sizing (fixed)
- PnL
- Equity curve

Includes:
- Transaction cost (default: 0.1%)
- Slippage simulation (optional)

---

## Evaluation Metrics

- Sharpe Ratio
- Max Drawdown
- CAGR
- Win Rate
- Trade Count
- Turnover

---

## Robustness Testing

### 1. Transaction Cost Stress
Test:
- 0.1%
- 0.2%
- 0.5%

---

### 2. Execution Delay
Simulate:
- 1-day delay in trade execution

---

### 3. Parameter Sensitivity
Vary:
- RSI thresholds
- Lookback windows

---

## Validation Method

Walk-forward validation:

- Train on past data
- Validate on next period
- Roll forward

NO random split.

---

## Output

- Performance comparison plots
- Trade logs
- Metrics table
- Robustness summary
