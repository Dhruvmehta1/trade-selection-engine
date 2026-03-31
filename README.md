# Trade Selection Engine  
### Evaluating whether machine learning adds real edge to trading strategies

---

## Overview

This project explores whether machine learning can improve rule-based trading strategies by acting as a **decision filter rather than a predictor**.

Instead of predicting market direction, the system:

1. Generates trade signals using simple mechanical strategies  
2. Uses ML to decide whether to take or skip each trade  
3. Evaluates performance under realistic conditions  

The goal is not to maximize backtest returns, but to determine whether any improvement is **real, robust, and survives real-world constraints**.

---

## Core Question

> Does machine learning provide real edge in filtering trades, or does it introduce overfitting that breaks under realistic conditions?

---

## System Design

The system is structured into four layers:

### 1. Strategy Layer (Signal Generation)
Rule-based strategies generate trade signals.

Examples:
- Momentum (breakout)
- Mean Reversion (RSI)
- Volatility-based filters

---

### 2. ML Decision Layer
A classification model evaluates each trade at entry.

The model does NOT predict price.  
It predicts:

→ Take trade  
→ Skip trade  

---

### 3. Evaluation Layer
Performance is compared between:

- Baseline strategy (all trades)  
- ML-filtered strategy  

Metrics:
- Sharpe Ratio  
- Max Drawdown  
- Win Rate  
- Turnover  
- PnL  

---

### 4. Robustness Engine
The system is stress-tested under realistic conditions:

- Increased transaction costs  
- Execution delays  
- Parameter perturbations  

The objective is to determine whether improvements persist under **real-world friction**.

---

## Key Principles

- No data leakage  
- Strict time-aware validation  
- Realistic backtesting assumptions  
- Focus on robustness over raw returns  

---

## Expected Outcome

A clear understanding of:

- When ML improves trade selection  
- When ML fails or overfits  
- Whether the improvement is stable or fragile
