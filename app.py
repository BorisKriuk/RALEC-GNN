#!/usr/bin/env python3
"""
CGECD Web Interface — Dashboard, Signal Monitor, Configuration
================================================================
Flask app with CORS enabled on port 80.

Endpoints:
  GET  /                  — Dashboard HTML
  GET  /api/backtest      — Run backtest with current config
  GET  /api/metrics       — Get latest backtest metrics
  GET  /api/trades        — Get trade log
  GET  /api/equity        — Get equity curve data
  GET  /api/signals       — Get current signal rules
  POST /api/signals       — Update signal rules
  GET  /api/features      — Get latest feature values
  GET  /api/status        — Health check
"""

import warnings
warnings.filterwarnings("ignore")

import json
import threading
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS

from config import Config
from algorithm import (
    load_data,
    build_spectral_features,
    build_traditional_features,
)
from backtester import Backtester, Strategy, SignalRule, DEFAULT_RULES

app = Flask(__name__)
CORS(app)

# ── Global state ────────────────────────────────────────────────
_state = {
    "loaded": False,
    "loading": False,
    "backtest_result": None,
    "config": None,
    "rules": [r.to_dict() for r in DEFAULT_RULES],
    "sizing_method": "binary",
    "holding_period": 3,
    "min_rf_prob": 0.3,
    "features_latest": {},
    "error": None,
}

_data_cache = {
    "prices": None,
    "returns": None,
    "spectral": None,
    "traditional": None,
    "combined": None,
    "target_up": None,
    "target_down": None,
}


def _load_data_background():
    """Load data in background thread."""
    try:
        _state["loading"] = True
        config = Config()
        _state["config"] = config

        prices, returns = load_data(config)
        spectral = build_spectral_features(returns, config)
        traditional = build_traditional_features(prices, returns)
        combined = pd.concat([spectral, traditional], axis=1)

        market = prices["SP500"] if "SP500" in prices.columns else prices.iloc[:, 0]
        fut_3d = market.pct_change(3).shift(-3)

        _data_cache["prices"] = prices
        _data_cache["returns"] = returns
        _data_cache["spectral"] = spectral
        _data_cache["traditional"] = traditional
        _data_cache["combined"] = combined
        _data_cache["target_up"] = (fut_3d > 0.03).astype(int)
        _data_cache["target_down"] = (fut_3d < -0.03).astype(int)

        # Latest feature values
        last_row = combined.iloc[-1]
        _state["features_latest"] = {
            col: round(float(last_row[col]), 6) if not np.isnan(last_row[col]) else None
            for col in combined.columns
        }

        _state["loaded"] = True
        _state["loading"] = False
        _state["error"] = None
        print("Data loaded successfully.")
    except Exception as e:
        _state["error"] = str(e)
        _state["loading"] = False
        print(f"Data load error: {e}")


def _parse_rules_from_state():
    """Convert state rules back to SignalRule objects."""
    rules = []
    for r in _state["rules"]:
        rules.append(SignalRule(
            name=r["name"],
            direction=r["direction"],
            feature_conditions=[(c[0], c[1], c[2]) for c in r["conditions"]],
            min_rf_prob=r.get("min_prob", _state["min_rf_prob"]),
        ))
    return rules


def _run_backtest():
    """Run backtest with current configuration."""
    if not _state["loaded"]:
        return {"error": "Data not loaded yet"}

    rules = _parse_rules_from_state()
    strategy = Strategy(
        rules=rules,
        holding_period=_state["holding_period"],
        sizing_method=_state["sizing_method"],
    )
    backtester = Backtester(_state["config"], strategy)
    result = backtester.run(
        _data_cache["combined"],
        _data_cache["prices"],
        _data_cache["target_up"],
        _data_cache["target_down"],
    )
    _state["backtest_result"] = result
    return result


# =================================================================
# HTML TEMPLATE
# =================================================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CGECD Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f1923; color: #e0e6ed; }
  .header { background: #1a2332; padding: 20px 30px; border-bottom: 2px solid #2ecc71;
            display: flex; justify-content: space-between; align-items: center; }
  .header h1 { font-size: 24px; color: #2ecc71; }
  .header .status { font-size: 14px; color: #7f8c8d; }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px; }
  .card { background: #1a2332; border-radius: 8px; padding: 20px; border: 1px solid #2c3e50; }
  .card h2 { font-size: 16px; color: #3498db; margin-bottom: 15px; text-transform: uppercase;
             letter-spacing: 1px; }
  .metric { display: flex; justify-content: space-between; padding: 8px 0;
            border-bottom: 1px solid #2c3e50; }
  .metric:last-child { border-bottom: none; }
  .metric .label { color: #7f8c8d; font-size: 13px; }
  .metric .value { font-weight: 700; font-size: 15px; }
  .metric .value.positive { color: #2ecc71; }
  .metric .value.negative { color: #e74c3c; }
  .metric .value.neutral { color: #f39c12; }
  .btn { background: #2ecc71; color: #0f1923; border: none; padding: 10px 24px;
         border-radius: 6px; cursor: pointer; font-weight: 700; font-size: 14px;
         transition: background 0.2s; }
  .btn:hover { background: #27ae60; }
  .btn:disabled { background: #7f8c8d; cursor: not-allowed; }
  .btn-secondary { background: #3498db; }
  .btn-secondary:hover { background: #2980b9; }
  canvas { width: 100% !important; height: 300px !important; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  table th { background: #2c3e50; padding: 8px 10px; text-align: left; font-weight: 600; }
  table td { padding: 6px 10px; border-bottom: 1px solid #2c3e50; }
  table tr:hover { background: #1e2d3d; }
  .signal-card { padding: 12px; margin-bottom: 10px; border-radius: 6px; border-left: 4px solid; }
  .signal-long { border-color: #2ecc71; background: rgba(46,204,113,0.1); }
  .signal-short { border-color: #e74c3c; background: rgba(231,76,60,0.1); }
  .signal-card .name { font-weight: 700; margin-bottom: 4px; }
  .signal-card .conds { font-size: 12px; color: #7f8c8d; }
  .config-row { display: flex; gap: 15px; align-items: center; margin-bottom: 12px; }
  .config-row label { color: #7f8c8d; font-size: 13px; min-width: 120px; }
  .config-row select, .config-row input {
    background: #0f1923; border: 1px solid #2c3e50; color: #e0e6ed;
    padding: 6px 10px; border-radius: 4px; font-size: 13px; }
  .loading { text-align: center; padding: 40px; color: #7f8c8d; }
  .spinner { display: inline-block; width: 30px; height: 30px; border: 3px solid #2c3e50;
             border-top: 3px solid #2ecc71; border-radius: 50%;
             animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .feature-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .feature-item { background: #0f1923; padding: 6px 10px; border-radius: 4px; font-size: 12px; }
  .feature-item .fname { color: #7f8c8d; }
  .feature-item .fval { font-weight: 700; color: #e0e6ed; }
  .tabs { display: flex; gap: 0; margin-bottom: 20px; }
  .tab { padding: 10px 24px; background: #1a2332; border: 1px solid #2c3e50;
         cursor: pointer; font-size: 14px; color: #7f8c8d; transition: all 0.2s; }
  .tab:first-child { border-radius: 6px 0 0 6px; }
  .tab:last-child { border-radius: 0 6px 6px 0; }
  .tab.active { background: #2ecc71; color: #0f1923; font-weight: 700; border-color: #2ecc71; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>

<div class="header">
  <h1>CGECD Dashboard</h1>
  <div class="status" id="status">Loading data...</div>
</div>

<div class="container">
  <div class="tabs">
    <div class="tab active" onclick="switchTab('dashboard')">Dashboard</div>
    <div class="tab" onclick="switchTab('signals')">Signal Monitor</div>
    <div class="tab" onclick="switchTab('config')">Configuration</div>
    <div class="tab" onclick="switchTab('trades')">Trade Log</div>
  </div>

  <!-- DASHBOARD TAB -->
  <div class="tab-content active" id="tab-dashboard">
    <div class="grid-3" id="metrics-cards">
      <div class="card">
        <h2>Performance</h2>
        <div id="perf-metrics"><div class="loading"><div class="spinner"></div><p>Loading...</p></div></div>
      </div>
      <div class="card">
        <h2>Trade Stats</h2>
        <div id="trade-metrics"><div class="loading"><div class="spinner"></div></div></div>
      </div>
      <div class="card">
        <h2>Risk</h2>
        <div id="risk-metrics"><div class="loading"><div class="spinner"></div></div></div>
      </div>
    </div>
    <div class="card" style="margin-bottom:20px">
      <h2>Equity Curve</h2>
      <canvas id="equityChart"></canvas>
    </div>
    <div class="card">
      <h2>Trade PnL Distribution</h2>
      <canvas id="pnlChart"></canvas>
    </div>
  </div>

  <!-- SIGNALS TAB -->
  <div class="tab-content" id="tab-signals">
    <div class="grid">
      <div class="card">
        <h2>Active Signal Rules</h2>
        <div id="signal-rules"></div>
      </div>
      <div class="card">
        <h2>Latest Feature Values</h2>
        <div id="feature-values" class="feature-grid"></div>
      </div>
    </div>
  </div>

  <!-- CONFIG TAB -->
  <div class="tab-content" id="tab-config">
    <div class="card">
      <h2>Strategy Configuration</h2>
      <div class="config-row">
        <label>Position Sizing:</label>
        <select id="cfg-sizing">
          <option value="binary">Binary (0/1)</option>
          <option value="linear">Linear (probability-scaled)</option>
        </select>
      </div>
      <div class="config-row">
        <label>Holding Period:</label>
        <input type="number" id="cfg-holding" value="3" min="1" max="20">
        <span style="color:#7f8c8d;font-size:12px">days</span>
      </div>
      <div class="config-row">
        <label>Min RF Probability:</label>
        <input type="number" id="cfg-minprob" value="0.3" min="0.1" max="0.9" step="0.05">
      </div>
      <div style="margin-top:20px">
        <button class="btn" onclick="runBacktest()">Run Backtest</button>
        <button class="btn btn-secondary" onclick="resetConfig()" style="margin-left:10px">Reset Defaults</button>
      </div>
    </div>
  </div>

  <!-- TRADES TAB -->
  <div class="tab-content" id="tab-trades">
    <div class="card">
      <h2>Trade Log</h2>
      <div style="overflow-x:auto">
        <table id="trades-table">
          <thead>
            <tr><th>#</th><th>Entry</th><th>Exit</th><th>Direction</th><th>Rule</th>
                <th>RF Prob</th><th>Size</th><th>PnL</th><th>Days</th></tr>
          </thead>
          <tbody id="trades-body"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
let equityChart = null;
let pnlChart = null;

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t, i) => {
    t.classList.toggle('active', t.textContent.toLowerCase().includes(name.substring(0,4)));
  });
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
}

function fmt(val, type) {
  if (val === null || val === undefined) return '—';
  if (type === 'pct') return (val * 100).toFixed(1) + '%';
  if (type === 'pct0') return (val * 100).toFixed(0) + '%';
  if (type === 'num') return val.toFixed(2);
  if (type === 'int') return Math.round(val).toString();
  return val.toString();
}

function metricHtml(label, value, type, cls) {
  return `<div class="metric"><span class="label">${label}</span><span class="value ${cls||''}">${fmt(value, type)}</span></div>`;
}

function updateDashboard(data) {
  if (!data || data.error) {
    document.getElementById('status').textContent = data ? data.error : 'Error';
    return;
  }
  const m = data.metrics;
  const retCls = m.total_return >= 0 ? 'positive' : 'negative';

  document.getElementById('perf-metrics').innerHTML =
    metricHtml('Total Return', m.total_return, 'pct', retCls) +
    metricHtml('Sharpe Ratio', m.sharpe_ratio, 'num', m.sharpe_ratio > 1 ? 'positive' : 'neutral') +
    metricHtml('Buy & Hold', m.buy_hold_return, 'pct', 'neutral') +
    metricHtml('Avg Trade PnL', m.avg_trade_pnl, 'pct', m.avg_trade_pnl > 0 ? 'positive' : 'negative');

  document.getElementById('trade-metrics').innerHTML =
    metricHtml('Total Trades', m.n_trades, 'int', '') +
    metricHtml('Win Rate', m.win_rate, 'pct0', m.win_rate > 0.5 ? 'positive' : 'negative') +
    metricHtml('Long Trades', m.long_trades, 'int', '') +
    metricHtml('Short Trades', m.short_trades, 'int', '');

  document.getElementById('risk-metrics').innerHTML =
    metricHtml('Max Drawdown', m.max_drawdown, 'pct', 'negative') +
    metricHtml('Sizing', m.sizing_method, 'str', '') +
    metricHtml('Holding Period', m.holding_period + 'd', 'str', '') +
    metricHtml('Wins / Losses', (m.wins||0) + ' / ' + (m.losses||0), 'str', '');

  // Equity chart
  if (data.equity_curve) {
    const ctx = document.getElementById('equityChart').getContext('2d');
    if (equityChart) equityChart.destroy();
    equityChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.equity_curve.map((_, i) => i + 1),
        datasets: [{
          label: 'Cumulative PnL',
          data: data.equity_curve,
          borderColor: '#2ecc71',
          backgroundColor: 'rgba(46,204,113,0.1)',
          fill: true,
          tension: 0.1,
          pointRadius: 0,
          borderWidth: 2,
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#e0e6ed' } } },
        scales: {
          x: { ticks: { color: '#7f8c8d' }, grid: { color: '#2c3e50' }, title: { display: true, text: 'Trade #', color: '#7f8c8d' } },
          y: { ticks: { color: '#7f8c8d' }, grid: { color: '#2c3e50' }, title: { display: true, text: 'Cumulative PnL', color: '#7f8c8d' } }
        }
      }
    });
  }

  // PnL distribution
  if (data.trades) {
    const pnls = data.trades.map(t => t.pnl);
    const bins = 20;
    const min = Math.min(...pnls), max = Math.max(...pnls);
    const step = (max - min) / bins;
    const labels = [], counts = [], colors = [];
    for (let i = 0; i < bins; i++) {
      const lo = min + i * step, hi = lo + step;
      const mid = (lo + hi) / 2;
      labels.push(mid.toFixed(3));
      counts.push(pnls.filter(p => p >= lo && (i === bins-1 ? p <= hi : p < hi)).length);
      colors.push(mid >= 0 ? 'rgba(46,204,113,0.7)' : 'rgba(231,76,60,0.7)');
    }
    const ctx2 = document.getElementById('pnlChart').getContext('2d');
    if (pnlChart) pnlChart.destroy();
    pnlChart = new Chart(ctx2, {
      type: 'bar',
      data: { labels, datasets: [{ label: 'Count', data: counts, backgroundColor: colors }] },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#7f8c8d' }, grid: { color: '#2c3e50' } },
          y: { ticks: { color: '#7f8c8d' }, grid: { color: '#2c3e50' } }
        }
      }
    });
  }

  // Trade log
  const tbody = document.getElementById('trades-body');
  tbody.innerHTML = '';
  if (data.trades) {
    data.trades.forEach((t, i) => {
      const cls = t.pnl >= 0 ? 'positive' : 'negative';
      tbody.innerHTML += `<tr>
        <td>${i+1}</td><td>${t.entry_date}</td><td>${t.exit_date}</td>
        <td style="color:${t.direction==='LONG'?'#2ecc71':'#e74c3c'}">${t.direction}</td>
        <td>${t.rule}</td><td>${(t.rf_prob*100).toFixed(1)}%</td>
        <td>${t.size.toFixed(2)}</td>
        <td class="${cls}">${(t.pnl*100).toFixed(2)}%</td>
        <td>${t.holding_days}</td></tr>`;
    });
  }

  document.getElementById('status').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
}

function updateSignals(rules, features) {
  const container = document.getElementById('signal-rules');
  container.innerHTML = '';
  rules.forEach(r => {
    const cls = r.direction === 'LONG' ? 'signal-long' : 'signal-short';
    const conds = r.conditions.map(c => `${c[0]} ${c[1]} P${c[2]}`).join(' AND ');
    container.innerHTML += `<div class="signal-card ${cls}">
      <div class="name">${r.name}</div>
      <div class="conds">${conds} | Min prob: ${r.min_prob}</div>
    </div>`;
  });

  const fv = document.getElementById('feature-values');
  fv.innerHTML = '';
  const keyFeats = ['lambda_1','lambda_2','lambda_3','spectral_gap','eigenvalue_entropy',
    'volatility_5d','volatility_10d','volatility_20d','garch_vol',
    'drawdown_20d','drawdown_60d','return_60d','price_to_sma_20','price_to_sma_50',
    'absorption_ratio_5','mean_abs_corr','cross_dispersion','max_loss_20d',
    'downside_vol_20d','effective_rank'];
  keyFeats.forEach(f => {
    const val = features[f];
    if (val !== undefined && val !== null) {
      fv.innerHTML += `<div class="feature-item"><div class="fname">${f}</div><div class="fval">${val.toFixed !== undefined ? val.toFixed(4) : val}</div></div>`;
    }
  });
}

async function runBacktest() {
  document.getElementById('status').textContent = 'Running backtest...';
  const sizing = document.getElementById('cfg-sizing').value;
  const holding = document.getElementById('cfg-holding').value;
  const minprob = document.getElementById('cfg-minprob').value;

  const resp = await fetch(`/api/backtest?sizing=${sizing}&holding=${holding}&minprob=${minprob}`);
  const data = await resp.json();
  updateDashboard(data);
}

function resetConfig() {
  document.getElementById('cfg-sizing').value = 'binary';
  document.getElementById('cfg-holding').value = '3';
  document.getElementById('cfg-minprob').value = '0.3';
}

async function init() {
  // Poll until data is loaded
  const poll = setInterval(async () => {
    const resp = await fetch('/api/status');
    const s = await resp.json();
    if (s.loaded) {
      clearInterval(poll);
      document.getElementById('status').textContent = 'Data loaded. Running backtest...';
      // Run initial backtest
      const btResp = await fetch('/api/backtest');
      const btData = await btResp.json();
      updateDashboard(btData);
      // Load signals
      const sigResp = await fetch('/api/signals');
      const sigData = await sigResp.json();
      const featResp = await fetch('/api/features');
      const featData = await featResp.json();
      updateSignals(sigData.rules, featData.features);
    } else if (s.loading) {
      document.getElementById('status').textContent = 'Loading data... please wait';
    } else if (s.error) {
      document.getElementById('status').textContent = 'Error: ' + s.error;
      clearInterval(poll);
    }
  }, 2000);
}

init();
</script>
</body>
</html>
"""


# =================================================================
# API ROUTES
# =================================================================
@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/status")
def api_status():
    return jsonify({
        "loaded": _state["loaded"],
        "loading": _state["loading"],
        "error": _state["error"],
        "has_backtest": _state["backtest_result"] is not None,
    })


@app.route("/api/backtest")
def api_backtest():
    if not _state["loaded"]:
        return jsonify({"error": "Data not loaded yet. Please wait."}), 503

    # Read config from query params
    _state["sizing_method"] = request.args.get("sizing", _state["sizing_method"])
    _state["holding_period"] = int(request.args.get("holding", _state["holding_period"]))
    _state["min_rf_prob"] = float(request.args.get("minprob", _state["min_rf_prob"]))

    result = _run_backtest()
    return jsonify(result)


@app.route("/api/metrics")
def api_metrics():
    if _state["backtest_result"] is None:
        return jsonify({"error": "No backtest results yet"}), 404
    return jsonify(_state["backtest_result"].get("metrics", {}))


@app.route("/api/trades")
def api_trades():
    if _state["backtest_result"] is None:
        return jsonify({"error": "No backtest results yet"}), 404
    return jsonify({"trades": _state["backtest_result"].get("trades", [])})


@app.route("/api/equity")
def api_equity():
    if _state["backtest_result"] is None:
        return jsonify({"error": "No backtest results yet"}), 404
    return jsonify({"equity_curve": _state["backtest_result"].get("equity_curve", [])})


@app.route("/api/signals", methods=["GET"])
def api_signals_get():
    return jsonify({"rules": _state["rules"]})


@app.route("/api/signals", methods=["POST"])
def api_signals_post():
    data = request.get_json()
    if data and "rules" in data:
        _state["rules"] = data["rules"]
        return jsonify({"status": "ok", "rules": _state["rules"]})
    return jsonify({"error": "Invalid payload"}), 400


@app.route("/api/features")
def api_features():
    return jsonify({"features": _state["features_latest"]})


# =================================================================
# MAIN
# =================================================================
if __name__ == "__main__":
    print("Starting CGECD Web Interface...")
    print("Loading data in background...")
    thread = threading.Thread(target=_load_data_background, daemon=True)
    thread.start()
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"Dashboard running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
