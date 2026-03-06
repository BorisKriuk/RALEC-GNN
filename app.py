#!/usr/bin/env python3
"""
CGECD Web Dashboard — Flask + CORS
====================================

Visual interface for backtest results, model performance, and signal monitoring.

Usage:
    python app.py              # Start on port 8080
    python app.py 80           # Start on port 80 (requires sudo)

Endpoints:
    GET /                      — Dashboard HTML
    GET /api/results           — Model comparison results
    GET /api/backtest          — Backtest results + equity curve
    GET /api/figures           — List of generated figures
    GET /api/figures/<name>    — Serve a specific figure
    POST /api/run-backtest     — Run backtest with custom parameters
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
from pathlib import Path

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "paper_figures"
BACKTEST_DIR = RESULTS_DIR / "backtest"


# ─────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────

@app.route("/api/results")
def api_results():
    """Return model comparison results."""
    csv_path = RESULTS_DIR / "experiment_results.csv"
    bcd_path = RESULTS_DIR / "bcd_auc_results.csv"

    data = {}
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        data["experiment_results"] = df.to_dict(orient="records")
    if bcd_path.exists():
        import pandas as pd
        df = pd.read_csv(bcd_path)
        data["bcd_auc_results"] = df.to_dict(orient="records")

    if not data:
        return jsonify({"error": "No results found. Run `python run.py` first."}), 404
    return jsonify(data)


@app.route("/api/backtest")
def api_backtest():
    """Return backtest results."""
    bt_path = BACKTEST_DIR / "backtest_results.json"
    if not bt_path.exists():
        return jsonify({"error": "No backtest results. Run `python backtester.py` first."}), 404
    with open(bt_path) as f:
        return jsonify(json.load(f))


@app.route("/api/figures")
def api_figures():
    """List available figures."""
    if not FIGURES_DIR.exists():
        return jsonify({"error": "No figures found. Run `python generate_paper_figures.py` first."}), 404
    files = sorted([f.name for f in FIGURES_DIR.iterdir() if f.suffix in (".png", ".pdf")])
    return jsonify({"figures": files})


@app.route("/api/figures/<name>")
def api_figure(name):
    """Serve a specific figure."""
    path = FIGURES_DIR / name
    if not path.exists():
        return jsonify({"error": f"Figure {name} not found"}), 404
    return send_file(path)


@app.route("/api/run-backtest", methods=["POST"])
def api_run_backtest():
    """Run backtest with custom parameters."""
    try:
        params = request.get_json(silent=True) or {}
        hold_days = params.get("hold_days", 10)
        up_threshold = params.get("up_threshold", 0.4)
        down_threshold = params.get("down_threshold", 0.4)

        from config import Config
        from algorithm import (
            load_data, build_spectral_features, build_traditional_features,
            compute_all_targets,
        )
        from backtester import run_backtest

        cfg = Config()
        prices, returns = load_data(cfg)
        spectral = build_spectral_features(returns, cfg)
        traditional = build_traditional_features(prices, returns)
        combined = __import__("pandas").concat([spectral, traditional], axis=1)
        all_targets = compute_all_targets(prices)

        result = run_backtest(
            combined, prices,
            all_targets["up_3pct_10d"],
            all_targets["drawdown_7pct_10d"],
            cfg,
            hold_days=hold_days,
            up_threshold=up_threshold,
            down_threshold=down_threshold,
        )

        with open(BACKTEST_DIR / "backtest_results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────
# DASHBOARD HTML
# ─────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CGECD Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #0f1419; color: #e1e8ed; }
.header { background: linear-gradient(135deg, #1a1a2e, #16213e);
           padding: 24px 40px; border-bottom: 2px solid #e74c3c; }
.header h1 { font-size: 24px; color: #fff; }
.header p { color: #8899a6; font-size: 14px; margin-top: 4px; }
.container { max-width: 1400px; margin: 0 auto; padding: 24px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
.card { background: #192734; border-radius: 12px; padding: 20px;
        border: 1px solid #2d3f50; }
.card h2 { color: #1da1f2; font-size: 16px; margin-bottom: 12px;
           border-bottom: 1px solid #2d3f50; padding-bottom: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 8px 6px; color: #8899a6; border-bottom: 1px solid #2d3f50; }
td { padding: 6px; border-bottom: 1px solid #1e2d3d; }
.ours { color: #e74c3c; font-weight: bold; }
.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.stat { text-align: center; padding: 12px; background: #1e2d3d; border-radius: 8px; }
.stat .value { font-size: 22px; font-weight: bold; color: #1da1f2; }
.stat .label { font-size: 11px; color: #8899a6; margin-top: 4px; }
.positive { color: #2ecc71; }
.negative { color: #e74c3c; }
.figures-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }
.fig-card { background: #192734; border-radius: 12px; overflow: hidden;
            border: 1px solid #2d3f50; }
.fig-card img { width: 100%; display: block; }
.fig-card .caption { padding: 10px; font-size: 12px; color: #8899a6; }
.tabs { display: flex; gap: 8px; margin-bottom: 20px; }
.tab { padding: 8px 20px; background: #192734; border: 1px solid #2d3f50;
       border-radius: 8px; cursor: pointer; font-size: 13px; color: #8899a6; }
.tab.active { background: #e74c3c; color: #fff; border-color: #e74c3c; }
.section { display: none; }
.section.active { display: block; }
#loading { text-align: center; padding: 60px; color: #8899a6; }
</style>
</head>
<body>
<div class="header">
  <h1>CGECD — Correlation Graph Eigenvalue Crisis Detector</h1>
  <p>Two-Task Evaluation: Rally Detection + Crash Detection | BCD-AUC Ranking</p>
</div>
<div class="container">
  <div class="tabs">
    <div class="tab active" onclick="showTab('overview')">Overview</div>
    <div class="tab" onclick="showTab('backtest')">Backtest</div>
    <div class="tab" onclick="showTab('figures')">Figures</div>
  </div>

  <div id="overview" class="section active">
    <div id="loading">Loading results...</div>
    <div id="results-content" style="display:none">
      <div class="grid">
        <div class="card"><h2>BCD-AUC Ranking</h2><div id="bcd-table"></div></div>
        <div class="card"><h2>Model Performance</h2><div id="model-table"></div></div>
      </div>
    </div>
  </div>

  <div id="backtest" class="section">
    <div id="bt-loading">Loading backtest...</div>
    <div id="bt-content" style="display:none">
      <div class="stat-grid" id="bt-stats"></div>
      <div class="grid" style="margin-top: 20px">
        <div class="card"><h2>Recent Trades</h2><div id="trades-table"></div></div>
        <div class="card"><h2>Equity Curve</h2><div id="equity-chart"></div></div>
      </div>
    </div>
  </div>

  <div id="figures" class="section">
    <div class="figures-grid" id="figures-grid"></div>
  </div>
</div>

<script>
function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById(name).classList.add('active');
}

// Load results
fetch('/api/results').then(r => r.json()).then(data => {
  document.getElementById('loading').style.display = 'none';
  document.getElementById('results-content').style.display = 'block';

  if (data.bcd_auc_results) {
    let html = '<table><tr><th>Model</th><th>Rally</th><th>Crash</th><th>BCD-AUC</th></tr>';
    data.bcd_auc_results.sort((a, b) => b.BCD_AUC - a.BCD_AUC);
    data.bcd_auc_results.forEach(r => {
      const cls = r.Role === 'ours' ? ' class="ours"' : '';
      html += `<tr${cls}><td>${r.Model}</td><td>${r.Rally_AUC.toFixed(3)}</td>` +
              `<td>${r.Crash_AUC.toFixed(3)}</td><td>${r.BCD_AUC.toFixed(3)}</td></tr>`;
    });
    html += '</table>';
    document.getElementById('bcd-table').innerHTML = html;
  }

  if (data.experiment_results) {
    let html = '<table><tr><th>Task</th><th>Model</th><th>AUC</th><th>AvgP</th><th>F1</th></tr>';
    data.experiment_results.sort((a, b) => b.AUC_ROC - a.AUC_ROC);
    data.experiment_results.forEach(r => {
      const cls = r.Role === 'ours' ? ' class="ours"' : '';
      html += `<tr${cls}><td>${r.Task}</td><td>${r.Model}</td>` +
              `<td>${r.AUC_ROC.toFixed(3)}</td><td>${r.Avg_Precision.toFixed(3)}</td>` +
              `<td>${r.F1.toFixed(3)}</td></tr>`;
    });
    html += '</table>';
    document.getElementById('model-table').innerHTML = html;
  }
}).catch(() => {
  document.getElementById('loading').innerHTML = 'No results found. Run <code>python run.py</code> first.';
});

// Load backtest
fetch('/api/backtest').then(r => r.json()).then(data => {
  document.getElementById('bt-loading').style.display = 'none';
  document.getElementById('bt-content').style.display = 'block';

  if (data.stats) {
    const s = data.stats;
    const retClass = s.total_return >= 0 ? 'positive' : 'negative';
    const ddClass = 'negative';
    document.getElementById('bt-stats').innerHTML = `
      <div class="stat"><div class="value ${retClass}">${(s.total_return * 100).toFixed(1)}%</div><div class="label">Total Return</div></div>
      <div class="stat"><div class="value">${s.sharpe}</div><div class="label">Sharpe Ratio</div></div>
      <div class="stat"><div class="value ${ddClass}">${(s.max_drawdown * 100).toFixed(1)}%</div><div class="label">Max Drawdown</div></div>
      <div class="stat"><div class="value">${(s.win_rate * 100).toFixed(0)}%</div><div class="label">Win Rate</div></div>
      <div class="stat"><div class="value">${s.n_trades}</div><div class="label">Trades (${s.n_long}L / ${s.n_short}S)</div></div>
      <div class="stat"><div class="value">${s.buy_hold_return !== null ? (s.buy_hold_return * 100).toFixed(1) + '%' : 'N/A'}</div><div class="label">Buy & Hold</div></div>
    `;
  }

  if (data.trades) {
    let html = '<table><tr><th>Entry</th><th>Exit</th><th>Dir</th><th>Prob</th><th>PnL</th></tr>';
    data.trades.slice(-20).reverse().forEach(t => {
      const cls = t.pnl >= 0 ? 'positive' : 'negative';
      html += `<tr><td>${t.entry}</td><td>${t.exit}</td><td>${t.direction}</td>` +
              `<td>${t.probability.toFixed(2)}</td><td class="${cls}">${(t.pnl * 100).toFixed(2)}%</td></tr>`;
    });
    html += '</table>';
    document.getElementById('trades-table').innerHTML = html;
  }

  if (data.equity) {
    // Simple SVG equity curve
    const eq = data.equity;
    const w = 560, h = 200, pad = 30;
    const minV = Math.min(0, ...eq), maxV = Math.max(0, ...eq);
    const range = maxV - minV || 1;
    const xScale = (w - 2 * pad) / (eq.length - 1);
    const yScale = (h - 2 * pad) / range;
    let path = '';
    eq.forEach((v, i) => {
      const x = pad + i * xScale;
      const y = h - pad - (v - minV) * yScale;
      path += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1);
    });
    const zeroY = h - pad - (0 - minV) * yScale;
    document.getElementById('equity-chart').innerHTML = `
      <svg width="${w}" height="${h}" style="background:#1e2d3d;border-radius:8px">
        <line x1="${pad}" y1="${zeroY}" x2="${w-pad}" y2="${zeroY}" stroke="#555" stroke-dasharray="4"/>
        <path d="${path}" fill="none" stroke="#1da1f2" stroke-width="2"/>
      </svg>`;
  }
}).catch(() => {
  document.getElementById('bt-loading').innerHTML = 'No backtest results. Run <code>python backtester.py</code> first.';
});

// Load figures
fetch('/api/figures').then(r => r.json()).then(data => {
  if (data.figures) {
    const grid = document.getElementById('figures-grid');
    data.figures.filter(f => f.endsWith('.png')).forEach(f => {
      grid.innerHTML += `<div class="fig-card">
        <img src="/api/figures/${f}" loading="lazy" alt="${f}">
        <div class="caption">${f.replace('.png','').replace(/_/g,' ')}</div>
      </div>`;
    });
  }
}).catch(() => {});
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"CGECD Dashboard: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
