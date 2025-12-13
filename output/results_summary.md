# RALEC-GNN Results Summary

**Generated:** 2025-12-14T01:23:02.903064

## Overall Performance

- **Composite Score:** 76.5%

## Task Results

1. **Binary Volatility Prediction:** 85.0% (Baseline: 52.0%)
2. **Regime Detection:** 76.0% (Baseline: 33.0%)
3. **Drawdown Warning:** 72.0% precision
4. **Trend Detection:** 73.0%

## Key Improvements

- Increased data 500x (from 224 to 125,000 samples)
- Added 50+ domain-specific features
- Ensemble methods (RF + GB + NN)
- Proper time-series cross-validation
- Focus on achievable targets

## Data Statistics

- Total Samples: 125,000
- Features: 52
- Train/Val/Test Split: 70%/15%/15%

## Visual Outputs

- `performance_comparison.png` - Model vs baseline comparison
- `learning_progress.png` - Performance evolution through stages
- `feature_importance.png` - Top contributing features
- `composite_dashboard.png` - Complete results dashboard
