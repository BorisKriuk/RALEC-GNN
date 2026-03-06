# CGECD — Correlation Graph Eigenvalue Crisis Detector

Spectral analysis of dynamic correlation networks to detect market regime shifts and predict extreme moves.

## Two-Task Evaluation

| Task | Target | Horizon |
|------|--------|---------|
| Rally Detection | S&P 500 up >3% | 10 days |
| Crash Detection | Max drawdown >7% | 10 days |

**Combined metric:** BCD-AUC = sqrt(Rally AUC x Crash AUC)

## Key Results

- **206 features** (179 spectral + 27 traditional)
- **25 assets** in correlation universe
- **8-fold walk-forward** validation (3yr train, 6mo test, 10d gap)

## Models

| Model | Type |
|-------|------|
| CGECD Combined (Ours) | Spectral + Traditional -> RF |
| Spectral Only RF | Spectral -> RF (ablation) |
| Traditional RF | Traditional -> RF (benchmark) |
| Turbulence RF | Kritzman & Li (2010) -> RF |
| HAR-RV LR | Corsi (2009) -> LR |
| SMA Vol LR | Rolling vol -> LR |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
echo "EODHD_API_KEY=your_key" > .env

# Run main experiment
python run.py

# Generate paper figures + SHAP
python generate_paper_figures.py

# Run backtester
python backtester.py

# Start dashboard (port 8080)
python app.py
```

## Files

| File | Description |
|------|-------------|
| `run.py` | Main experiment — 6 models x 2 tasks |
| `algorithm.py` | CGECD model, spectral features, walk-forward |
| `benchmarks.py` | SOTA benchmark models |
| `config.py` | Configuration |
| `metrics.py` | Evaluation metrics |
| `generate_paper_figures.py` | Publication figures + SHAP analysis |
| `backtester.py` | Walk-forward trading backtest |
| `app.py` | Flask dashboard with CORS |
