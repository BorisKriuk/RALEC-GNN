# CGECD — Correlation Graph Eigenvalue Crisis Detector

**Spectral Features of Dynamic Correlation Networks for Market Crash Early Warning**

A machine learning system that extracts spectral features (eigenvalues, spectral gap, absorption ratio) from rolling correlation matrices of 25 ETFs and combines them with traditional financial features (volatility, drawdown, momentum) in a Random Forest ensemble to predict large (>3%) 3-day S&P 500 moves.

## Key Results

| Task | AUC-ROC | Model |
|------|---------|-------|
| UP moves (>3%) | 0.832 | CGECD (RF + Vol-LR ensemble) |
| DOWN moves (>3%) | 0.675 | CGECD (RF + Vol-LR ensemble) |

- **85 features**: 45 spectral + 40 traditional
- **Walk-forward validation**: 20 folds, 3-year expanding window, 6-month test, 5-day gap
- **Lambda_2** (second eigenvalue) is #2 predictor for crashes via SHAP analysis
- **Spectral gap** provides earliest crash warning at day -14

## Project Structure

```
algorithm.py          — Core CGECD model (data loading, feature engineering, walk-forward)
config.py             — Configuration dataclass (hyperparameters, asset universe)
metrics.py            — Evaluation metrics (AUC, precision, recall, bootstrap CI)
benchmarks.py         — Baseline models (HAR-RV, GARCH, Absorption Ratio, Turbulence)
visualizations.py     — Plotting utilities
run.py                — Main experiment runner (model comparison + ablation)
backtester.py         — Trading strategy backtester module
app.py                — Flask web dashboard (CORS enabled, port 80)
generate_paper_figures.py — Publication-quality figure generation + SHAP
investigate_v2.py     — Feature investigation (thresholds, SHAP on all 85 features)
investigate_v3.py     — Extended analysis (interaction rules, backtest, spectral explanation)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
echo "EODHD_API_KEY=your_key" > .env

# Run full experiment (model comparison + ablation)
python run.py

# Generate paper figures + SHAP analysis
python generate_paper_figures.py

# Run backtest
python backtester.py

# Launch web dashboard
sudo python app.py   # port 80 requires sudo
```

## Web Dashboard

```bash
sudo python app.py
# Open http://localhost in browser
```

Features:
- Equity curve and performance metrics
- Signal monitor with latest feature values
- Configuration panel (sizing, holding period, thresholds)
- Full trade log

## Asset Universe

25 ETFs covering: S&P 500, Nasdaq, Russell 2000, 9 GICS sectors, international equity (EAFE, EM, Europe, Japan), fixed income (Treasury, IG, HY), commodities (Gold, Oil), USD, REITs.

## Data Source

[EODHD API](https://eodhd.com/) — daily OHLCV data, 15-year history.
