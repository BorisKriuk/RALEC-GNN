#!/usr/bin/env python3
"""
visualization_server.py — CGECD 3D Minimalist Dashboard
=========================================================
Three.js force-directed correlation graph with glow nodes,
labelled assets, correlation-colored FAT edges with legend,
glassmorphism HUD, regime detection, and strategy signals.

Auto-refreshes the full data pipeline every 12 hours.
Flask-CORS enabled for cross-origin access.
"""

import warnings
warnings.filterwarnings('ignore')

import json, threading, time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from scipy import stats

from config import Config
from algorithm import (
    load_data, build_spectral_features, build_traditional_features,
    compute_all_targets, CGECDModel, walk_forward_evaluate,
)
from benchmarks import prepare_benchmark_features

app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════

STATE = dict(
    initialized=False, status_message='Starting...', step=0, total_steps=6,
    prices=None, returns=None, config=None,
    spectral_features=None, traditional_features=None, combined_features=None,
    rally_probs=None, crash_probs=None, all_targets=None,
    rally_metrics=None, crash_metrics=None,
    last_refresh=None, refresh_count=0,
)

REFRESH_INTERVAL = 12 * 60 * 60  # 12 hours in seconds

CATEGORIES = {
    'SP500':'equity','Nasdaq100':'equity','Russell2000':'equity',
    'Financials':'sector','Energy':'sector','Technology':'sector',
    'Healthcare':'sector','Utilities':'sector','ConsumerStaples':'sector',
    'ConsumerDisc':'sector','Industrials':'sector','Materials':'sector',
    'RealEstate':'realestate','DevIntl':'intl','EmergingMkts':'intl',
    'Europe':'intl','Japan':'intl',
    'LongTreasury':'bond','IntermTreasury':'bond',
    'InvGradeCorp':'bond','HighYield':'bond',
    'Gold':'commodity','Oil':'commodity',
    'USDollar':'currency','REITs':'realestate',
}

DISPLAY_NAMES = {
    'SP500':'S&P 500','Nasdaq100':'Nasdaq','Russell2000':'Russell 2K',
    'Financials':'Financials','Energy':'Energy','Technology':'Tech',
    'Healthcare':'Health','Utilities':'Utilities','ConsumerStaples':'Staples',
    'ConsumerDisc':'Cons Disc','Industrials':'Industrials','Materials':'Materials',
    'RealEstate':'Real Est','DevIntl':'Dev Intl','EmergingMkts':'EM',
    'Europe':'Europe','Japan':'Japan',
    'LongTreasury':'Long Tsy','IntermTreasury':'Mid Tsy',
    'InvGradeCorp':'IG Corp','HighYield':'HY Corp',
    'Gold':'Gold','Oil':'Oil',
    'USDollar':'USD','REITs':'REITs',
}

CAT_LABELS = {
    'equity':'EQUITY','sector':'SECTOR','intl':'INTL',
    'bond':'BOND','commodity':'CMDTY','currency':'FX',
    'realestate':'REIT','other':'OTHER',
}

CAT_HEX = {
    'equity':'#4C8BF5','sector':'#9966FF','intl':'#00BCD4',
    'bond':'#4CAF50','commodity':'#FF9800','currency':'#E91E63',
    'realestate':'#FF5722','other':'#607D8B',
}
CAT_INT = {
    'equity':0x4C8BF5,'sector':0x9966FF,'intl':0x00BCD4,
    'bond':0x4CAF50,'commodity':0xFF9800,'currency':0xE91E63,
    'realestate':0xFF5722,'other':0x607D8B,
}

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def sf(v):
    if v is None: return 0.0
    v = float(v)
    if np.isnan(v) or np.isinf(v): return 0.0
    return v

def rolling_rank(s, w=126):
    return s.rolling(w, min_periods=20).rank(pct=True)

def exposure_2d(rr, cr):
    if np.isnan(rr): rr = 0.5
    if np.isnan(cr): cr = 0.5
    if rr >= 0.90: return 0.0
    if cr >= 0.60: return 0.0
    if 0.78 <= rr < 0.90 and cr < 0.40: return 1.5
    if 0.78 <= rr < 0.90 and cr < 0.55: return 1.2
    if rr < 0.40 and cr < 0.55: return 1.0
    if cr < 0.55: return 0.7
    return 0.3

# ═══════════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline():
    """Execute the full data pipeline. Called on startup and every 12h."""
    global STATE
    try:
        cfg = Config(); STATE['config'] = cfg

        STATE['step'] = 1; STATE['status_message'] = 'Loading market data…'
        prices, returns = load_data(cfg)
        STATE['prices'] = prices; STATE['returns'] = returns

        STATE['step'] = 2; STATE['status_message'] = 'Building spectral features…'
        sp = build_spectral_features(returns, cfg)
        STATE['spectral_features'] = sp

        STATE['step'] = 3; STATE['status_message'] = 'Building traditional features…'
        tr = build_traditional_features(prices, returns)
        STATE['traditional_features'] = tr
        STATE['combined_features'] = pd.concat([sp, tr], axis=1)

        STATE['step'] = 4; STATE['status_message'] = 'Computing targets…'
        STATE['all_targets'] = compute_all_targets(prices)

        STATE['step'] = 5; STATE['status_message'] = 'Walk-forward rally…'
        comb = STATE['combined_features']
        rr = walk_forward_evaluate(comb, STATE['all_targets']['up_3pct_10d'], CGECDModel, cfg)
        STATE['rally_probs'] = pd.Series(rr['probabilities'], index=pd.DatetimeIndex(rr['dates'])).groupby(level=0).last().sort_index()
        STATE['rally_metrics'] = rr['metrics']

        STATE['status_message'] = 'Walk-forward crash…'
        cr = walk_forward_evaluate(comb, STATE['all_targets']['drawdown_7pct_10d'], CGECDModel, cfg)
        STATE['crash_probs'] = pd.Series(cr['probabilities'], index=pd.DatetimeIndex(cr['dates'])).groupby(level=0).last().sort_index()
        STATE['crash_metrics'] = cr['metrics']

        STATE['step'] = 6; STATE['status_message'] = 'Ready'
        STATE['initialized'] = True
        STATE['last_refresh'] = datetime.utcnow().isoformat() + 'Z'
        STATE['refresh_count'] += 1
        print(f"\n✓ Pipeline complete (run #{STATE['refresh_count']}) → {STATE['last_refresh']}\n")
    except Exception as e:
        STATE['status_message'] = f'Error: {e}'
        import traceback; traceback.print_exc()


def refresh_loop():
    """Initial run + repeat every REFRESH_INTERVAL seconds."""
    run_pipeline()
    print(f"  Next refresh in {REFRESH_INTERVAL // 3600}h\n")
    while True:
        time.sleep(REFRESH_INTERVAL)
        print(f"\n⟳ Auto-refresh triggered at {datetime.utcnow().isoformat()}Z")
        STATE['status_message'] = 'Refreshing data…'
        run_pipeline()
        print(f"  Next refresh in {REFRESH_INTERVAL // 3600}h\n")

# ═══════════════════════════════════════════════════════════════
# PAYLOADS
# ═══════════════════════════════════════════════════════════════

def build_graph(th=0.3):
    returns = STATE['returns']; prices = STATE['prices']
    if returns is None: return {}
    n = len(returns.columns); names = list(returns.columns)
    recent = returns.iloc[-60:]
    corr = recent.corr().values; corr = np.nan_to_num(corr,nan=0); np.fill_diagonal(corr,1.0)
    ev, evc = np.linalg.eigh(corr)
    idx = np.argsort(ev)[::-1]; ev = ev[idx]; evc = evc[:,idx]
    v1 = np.abs(evc[:,0]); v1 = v1/(v1.sum()+1e-10)

    nodes = []
    for i, nm in enumerate(names):
        cat = CATEGORIES.get(nm,'other')
        r20 = sf(prices[nm].pct_change(20).iloc[-1]) if nm in prices.columns else 0
        r5 = sf(prices[nm].pct_change(5).iloc[-1]) if nm in prices.columns else 0
        vol = sf(returns[nm].iloc[-20:].std()*np.sqrt(252))
        nodes.append(dict(
            id=nm,
            label=DISPLAY_NAMES.get(nm,nm),
            cat=cat,
            catLabel=CAT_LABELS.get(cat,'OTHER'),
            color=CAT_HEX.get(cat,'#607D8B'),
            colorInt=CAT_INT.get(cat,0x607D8B),
            loading=sf(v1[i]),ret20=r20,ret5=r5,vol=vol,degree=0))
    links = []
    for i in range(n):
        for j in range(i+1,n):
            c = corr[i,j]
            if abs(c) >= th:
                links.append(dict(source=names[i],target=names[j],
                                  corr=sf(c),abs_corr=sf(abs(c))))
                nodes[i]['degree'] += 1; nodes[j]['degree'] += 1
    up = corr[np.triu_indices(n,k=1)]
    topo = dict(mean_corr=sf(np.mean(np.abs(up))),frac70=sf(np.mean(np.abs(up)>0.7)),
                n_edges=len(links),max_edges=int(n*(n-1)/2),
                density=sf(len(links)/max(n*(n-1)/2,1)))
    return dict(nodes=nodes,links=links,topo=topo)

def build_spectrum():
    returns = STATE['returns']
    if returns is None: return {}
    n = len(returns.columns)
    c60 = returns.iloc[-60:].corr().values; c60=np.nan_to_num(c60); np.fill_diagonal(c60,1.0)
    ev = np.sort(np.maximum(np.linalg.eigvalsh(c60),1e-10))[::-1]
    t = ev.sum(); norm = ev/t
    entropy = float(-np.sum(norm*np.log(norm+1e-10)))
    q = n/60.0; mp = (1+1.0/np.sqrt(q))**2 if q>0 else 4.0
    return dict(eigenvalues=[sf(v) for v in ev], explained=[sf(v) for v in norm],
                mp=sf(mp), ar1=sf(ev[0]/t), ar5=sf(np.sum(ev[:5])/t),
                eff_rank=sf(np.exp(entropy)), gap=sf(ev[0]/(ev[1]+1e-10)),
                entropy=sf(entropy), n=n)

def build_signals():
    rp = STATE.get('rally_probs'); cp = STATE.get('crash_probs')
    if rp is None or cp is None or len(rp)==0:
        return dict(regime='LOADING',rc='#64748b',rp=0,cp=0,rr=0.5,cr=0.5,
                    exp=0.5,strat='Loading…',alerts=[],history=None,
                    rauc=0,cauc=0)
    r_rank = rolling_rank(rp); c_rank = rolling_rank(cp)
    ra = sf(rp.iloc[-1]); ca = sf(cp.iloc[-1])
    rr = sf(r_rank.iloc[-1]) if not np.isnan(r_rank.iloc[-1]) else 0.5
    cr = sf(c_rank.iloc[-1]) if not np.isnan(c_rank.iloc[-1]) else 0.5
    exp = exposure_2d(rr,cr)

    if cr>=0.60: regime,rc='CRISIS','#ff3d71'
    elif rr>=0.90: regime,rc='EUPHORIA','#ffaa00'
    elif 0.78<=rr<0.90 and cr<0.40: regime,rc='RALLY','#00e096'
    elif cr>=0.40: regime,rc='CAUTION','#ffaa00'
    else: regime,rc='NORMAL','#00e5ff'

    if exp==0: strat='EXIT → Defensive portfolio'
    elif exp>=1.3: strat='STRONG LONG — max equity'
    elif exp>=0.8: strat='LONG — standard equity'
    elif exp>=0.5: strat='MODERATE — reduced equity'
    else: strat='LIGHT — minimal equity'

    alerts=[]
    if cr>=0.80: alerts.append(dict(lv='crit',msg=f'Crash alert — rank {cr:.0%}',ic=''))
    elif cr>=0.60: alerts.append(dict(lv='warn',msg=f'Crash warning — rank {cr:.0%}',ic='⚠️'))
    if rr>=0.90: alerts.append(dict(lv='warn',msg=f'Rally overconfidence — {rr:.0%}',ic='⚠️'))
    if not alerts: alerts.append(dict(lv='ok',msg='No active alerts',ic='✓'))

    hl=min(252,len(rp))
    history=dict(
        dates=[d.strftime('%Y-%m-%d') for d in rp.index[-hl:]],
        rr=[sf(v) for v in r_rank.iloc[-hl:].fillna(0.5)],
        cr=[sf(v) for v in c_rank.iloc[-hl:].fillna(0.5)],
    )
    rm=STATE.get('rally_metrics'); cm=STATE.get('crash_metrics')
    return dict(regime=regime,rc=rc,rp=ra,cp=ca,rr=rr,cr=cr,exp=exp,strat=strat,
                alerts=alerts,history=history,
                rauc=sf(rm.auc_roc) if rm else 0,cauc=sf(cm.auc_roc) if cm else 0)

def build_market():
    p = STATE.get('prices')
    if p is None: return {}
    m = p['SP500'] if 'SP500' in p.columns else p.iloc[:,0]
    mr = m.pct_change(); hl = min(504,len(m)); s = m.iloc[-hl:]
    return dict(dates=[d.strftime('%Y-%m-%d') for d in s.index],
                prices=[sf(v) for v in s],
                r1=sf(mr.iloc[-1]),r5=sf(m.pct_change(5).iloc[-1]),
                r20=sf(m.pct_change(20).iloc[-1]),
                vol=sf(mr.iloc[-20:].std()*np.sqrt(252)))

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return Response(HTML, mimetype='text/html')

@app.route('/api/status')
def api_status():
    return jsonify(dict(ok=STATE['initialized'],msg=STATE['status_message'],
                        s=STATE['step'],t=STATE['total_steps'],
                        last_refresh=STATE.get('last_refresh'),
                        refresh_count=STATE.get('refresh_count',0)))

@app.route('/api/graph')
def api_graph():
    if not STATE['initialized']: return jsonify({})
    return jsonify(build_graph(float(request.args.get('threshold',0.3))))

@app.route('/api/spectrum')
def api_spectrum():
    if not STATE['initialized']: return jsonify({})
    return jsonify(build_spectrum())

@app.route('/api/signals')
def api_signals():
    return jsonify(build_signals())

@app.route('/api/market')
def api_market():
    if not STATE['initialized']: return jsonify({})
    return jsonify(build_market())

# ═══════════════════════════════════════════════════════════════
# HTML
# ═══════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CGECD</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--g:rgba(255,255,255,0.03);--gb:rgba(255,255,255,0.05);--b:rgba(255,255,255,0.06);--t1:rgba(255,255,255,.87);--t2:rgba(255,255,255,.4);--t3:rgba(255,255,255,.2)}
body{background:#000;color:var(--t1);font-family:'Inter',system-ui,sans-serif;overflow:hidden}

/* Loading */
#loading{position:fixed;inset:0;background:#000;display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:9999;transition:opacity .8s}
#loading.out{opacity:0;pointer-events:none}
.ring{width:44px;height:44px;border:2px solid rgba(255,255,255,.04);border-top-color:#00e5ff;border-radius:50%;animation:sp 1s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.ltitle{margin-top:20px;font-size:22px;font-weight:200;letter-spacing:10px;color:rgba(255,255,255,.45)}
.lmsg{margin-top:10px;font-size:11px;color:var(--t3);letter-spacing:1px}
.lbar{width:200px;height:2px;background:rgba(255,255,255,.04);border-radius:1px;margin-top:16px;overflow:hidden}
.lbar div{height:100%;background:#00e5ff;transition:width .5s;border-radius:1px}

/* Canvas */
canvas#c{position:fixed;inset:0;z-index:0}

/* Node labels */
.node-label{pointer-events:none;text-align:center;transform:translate(-50%,-50%);white-space:nowrap;transition:opacity .3s}
.node-name{font-size:10px;font-weight:600;letter-spacing:1.5px;text-shadow:0 0 8px rgba(0,0,0,.9),0 0 20px rgba(0,0,0,.7)}
.node-tag{font-size:7px;font-weight:500;letter-spacing:2px;opacity:.45;margin-top:1px;text-shadow:0 0 6px rgba(0,0,0,.9)}

/* Topbar */
#top{position:fixed;top:0;left:0;right:0;height:44px;z-index:20;display:flex;align-items:center;padding:0 20px;background:rgba(0,0,0,.5);backdrop-filter:blur(30px);-webkit-backdrop-filter:blur(30px);border-bottom:1px solid var(--b)}
.logo{font-size:13px;font-weight:300;letter-spacing:6px;color:var(--t2)}
.badge{margin-left:16px;padding:4px 14px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:2px}
#refresh-info{margin-left:12px;font-size:9px;color:var(--t3);letter-spacing:.5px;opacity:.6}
#alerts-row{margin-left:auto;display:flex;gap:8px}
.al{padding:3px 10px;border-radius:12px;font-size:10px;letter-spacing:.5px;white-space:nowrap}
.al.crit{background:rgba(255,61,113,.12);color:#ff3d71;border:1px solid rgba(255,61,113,.2)}
.al.warn{background:rgba(255,170,0,.1);color:#ffaa00;border:1px solid rgba(255,170,0,.15)}
.al.ok{background:rgba(0,224,150,.08);color:#00e096;border:1px solid rgba(0,224,150,.12)}

/* Sidebar */
#side{position:fixed;top:44px;right:0;bottom:0;width:300px;z-index:20;background:rgba(0,0,0,.55);backdrop-filter:blur(40px);-webkit-backdrop-filter:blur(40px);border-left:1px solid var(--b);overflow-y:auto;padding:16px}
#side::-webkit-scrollbar{width:3px}
#side::-webkit-scrollbar-thumb{background:rgba(255,255,255,.08);border-radius:2px}

/* Cards */
.cd{margin-bottom:10px;padding:12px;background:var(--g);border:1px solid var(--b);border-radius:12px}
.cl{font-size:9px;font-weight:500;text-transform:uppercase;letter-spacing:2px;color:var(--t3);margin-bottom:8px}

/* Regime */
#regime-txt{font-size:28px;font-weight:700;letter-spacing:4px;text-align:center;margin:4px 0}
.auc-row{display:flex;justify-content:center;gap:16px;margin-top:4px}
.auc-item{text-align:center}
.auc-v{font-size:14px;font-weight:600;font-variant-numeric:tabular-nums}
.auc-l{font-size:8px;color:var(--t3);text-transform:uppercase;letter-spacing:1px}

/* Gauges */
.gauges{display:flex;justify-content:center;gap:12px}
.gauge{text-align:center}

/* Exposure */
.exp-row{display:flex;justify-content:space-between;font-size:11px;color:var(--t2);margin-bottom:4px}
.exp-val{font-weight:600}
.exp-bar{width:100%;height:6px;background:rgba(255,255,255,.04);border-radius:3px;overflow:hidden}
.exp-fill{height:100%;border-radius:3px;transition:width .6s,background .6s}
#strat-txt{margin-top:8px;font-size:11px;font-weight:500;text-align:center;padding:7px;background:rgba(255,255,255,.02);border-radius:8px;border-left:2px solid #00e5ff}

/* Stats */
.sr{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:6px}
.si{background:rgba(255,255,255,.02);border-radius:6px;padding:6px;text-align:center}
.sv{font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;color:var(--t1)}
.sl{font-size:8px;color:var(--t3);margin-top:1px;text-transform:uppercase;letter-spacing:.5px}

/* Spectrum canvas */
.sc{width:100%;height:80px;border-radius:6px}
/* Market canvas */
.mc{width:100%;height:64px;border-radius:6px}

/* Signal history canvas */
.hc{width:100%;height:72px;border-radius:6px}

/* Range slider */
input[type=range]{width:100%;accent-color:#00e5ff;margin-top:4px;height:2px}
.th-val{float:right;font-size:11px;color:#00e5ff;font-weight:500}

/* Correlation legend */
.corr-legend{margin-top:8px}
.corr-bar{width:100%;height:10px;border-radius:5px;margin:4px 0}
.corr-labels{display:flex;justify-content:space-between;font-size:8px;letter-spacing:.5px;color:var(--t3)}
.corr-labels span.neg-l{color:#ff3d71}
.corr-labels span.pos-l{color:#4C8BF5}
.corr-labels span.zero-l{color:var(--t3)}
.corr-desc{font-size:9px;color:var(--t3);line-height:1.4;margin-top:6px;padding:6px;background:rgba(255,255,255,.015);border-radius:6px;border-left:2px solid rgba(255,255,255,.06)}

/* Tooltip */
#tip{position:fixed;z-index:100;background:rgba(0,0,0,.82);backdrop-filter:blur(16px);border:1px solid var(--b);border-radius:10px;padding:10px 14px;pointer-events:none;display:none;min-width:160px;font-size:11px}
.tip-name{font-size:13px;font-weight:600;margin-bottom:6px}
.tip-r{display:flex;justify-content:space-between;gap:12px;padding:2px 0;color:var(--t2)}
.tip-v{color:var(--t1);font-weight:500;font-variant-numeric:tabular-nums}
.pos{color:#00e096}.neg{color:#ff3d71}

/* Bottom hint */
#hint{position:fixed;bottom:12px;left:50%;transform:translateX(-50%);z-index:20;font-size:10px;color:var(--t3);letter-spacing:1px;opacity:.5}

@media(max-width:768px){#side{width:100%;top:auto;bottom:0;height:50vh;border-left:none;border-top:1px solid var(--b)}}
</style>
</head>
<body>

<div id="loading">
  <div class="ring"></div>
  <div class="ltitle">CGECD</div>
  <div class="lmsg" id="lmsg">Connecting…</div>
  <div class="lbar"><div id="lbar" style="width:0%"></div></div>
</div>

<div id="app" style="display:none">
  <div id="top">
    <div class="logo">CGECD</div>
    <div class="badge" id="badge">—</div>
    <span id="refresh-info"></span>
    <div id="alerts-row"></div>
  </div>

  <div id="side">
    <div class="cd" id="regime-card">
      <div class="cl">Regime</div>
      <div id="regime-txt">—</div>
      <div class="auc-row" id="auc-row"></div>
    </div>
    <div class="cd">
      <div class="cl">Signals</div>
      <div class="gauges" id="gauges"></div>
    </div>
    <div class="cd">
      <div class="cl">Strategy</div>
      <div class="exp-row"><span>Exposure</span><span class="exp-val" id="exp-v">0.0x</span></div>
      <div class="exp-bar"><div class="exp-fill" id="exp-fill"></div></div>
      <div id="strat-txt">—</div>
    </div>
    <div class="cd">
      <div class="cl">Signal History</div>
      <canvas class="hc" id="hist-c" width="268" height="72"></canvas>
    </div>
    <div class="cd">
      <div class="cl">Eigenvalue Spectrum</div>
      <canvas class="sc" id="spec-c" width="268" height="80"></canvas>
      <div class="sr" id="spec-st"></div>
    </div>
    <div class="cd">
      <div class="cl">S&amp;P 500</div>
      <canvas class="mc" id="mkt-c" width="268" height="64"></canvas>
      <div class="sr" id="mkt-st"></div>
    </div>
    <div class="cd">
      <div class="cl">Network</div>
      <div class="sr" id="net-st"></div>
    </div>
    <div class="cd">
      <div class="cl">Edge Correlation</div>
      <div class="corr-legend">
        <canvas class="corr-bar" id="corr-bar-c" width="268" height="10"></canvas>
        <div class="corr-labels">
          <span class="neg-l">−1.0</span>
          <span class="zero-l">0</span>
          <span class="pos-l">+1.0</span>
        </div>
        <div class="corr-desc">
          Edges show 60-day rolling Pearson correlation between asset pairs.
          <b style="color:#4C8BF5">Blue</b> = assets move together (positive ρ).
          <b style="color:#ff3d71">Red</b> = assets move opposite (negative ρ).
          Brighter, thicker lines = stronger correlation.
          Only pairs above the threshold are shown.
        </div>
      </div>
    </div>
    <div class="cd">
      <div class="cl">Threshold<span class="th-val" id="th-v">0.30</span></div>
      <input type="range" id="th-sl" min="0.10" max="0.90" step="0.05" value="0.30">
    </div>
  </div>

  <div id="tip">
    <div class="tip-name" id="tip-name"></div>
    <div id="tip-body"></div>
  </div>
  <div id="hint">drag to orbit · scroll to zoom · click node to focus · auto-refresh 12h</div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/renderers/CSS2DRenderer.js"></script>
<!-- Fat-line extensions for real linewidth support -->
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/lines/LineSegmentsGeometry.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/lines/LineGeometry.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/lines/LineMaterial.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/lines/LineSegments2.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/lines/Line2.js"></script>
<script>
(function(){

/* ── GLOBALS ──────────────────────────────────────── */
let scene,camera,renderer,labelRenderer,controls,clock;
let nodeGroup,linkGroup,particleSys,gridMesh,ringMesh;
let nodeMeshes=[],linkLines=[],nodeMap={},labelMap={};
let simNodes=[],simLinks=[],simAlpha=1;
let graphD,specD,sigD,mktD;
let hovered=null,focused=null,targetCam=null,targetLook=null;
let regimeColor=0x00e5ff;
const pointer=new THREE.Vector2(-999,-999);
const raycaster=new THREE.Raycaster();
let threshold=0.3;
let viewportRes=new THREE.Vector2(window.innerWidth,window.innerHeight);

/* Track server refresh to detect backend data updates */
let knownRefreshCount=0;

const CATC={equity:0x4C8BF5,sector:0x9966FF,intl:0x00BCD4,bond:0x4CAF50,
            commodity:0xFF9800,currency:0xE91E63,realestate:0xFF5722,other:0x607D8B};

/* ── CORRELATION → COLOR ──────────────────────────── */
function corrToRGB(c){
  const t=Math.abs(c);
  let r,g,b;
  if(c>=0){
    r=Math.round(0x22+(0x4C-0x22)*t);
    g=Math.round(0x22+(0x8B-0x22)*t);
    b=Math.round(0x33+(0xF5-0x33)*t);
  }else{
    r=Math.round(0x22+(0xFF-0x22)*t);
    g=Math.round(0x22+(0x3D-0x22)*t);
    b=Math.round(0x33+(0x71-0x33)*t);
  }
  return {r,g,b};
}
function corrToHex(c){
  const {r,g,b}=corrToRGB(c);
  return (r<<16)|(g<<8)|b;
}
function corrToCSS(c){
  const {r,g,b}=corrToRGB(c);
  return 'rgb('+r+','+g+','+b+')';
}
function corrOpacity(ac){
  return 0.08+ac*0.72;
}
function corrWidth(ac){
  return 1.0+ac*3.5;
}

/* ── BOOT ─────────────────────────────────────────── */
async function boot(){
  while(true){
    try{
      const r=await fetch('/api/status'); const s=await r.json();
      document.getElementById('lmsg').textContent=s.msg;
      document.getElementById('lbar').style.width=(s.s/s.t*100)+'%';
      if(s.ok){knownRefreshCount=s.refresh_count||0;break;}
    }catch(e){}
    await delay(2000);
  }
  document.getElementById('loading').classList.add('out');
  document.getElementById('app').style.display='';
  setTimeout(()=>{const el=document.getElementById('loading');if(el)el.remove()},900);
  await fetchAll();
  initScene();
  buildGraph();
  drawCorrLegendBar();
  updateUI();
  animate();

  /* Poll every 60s for UI data; also detect backend 12h refresh */
  setInterval(async()=>{
    try{
      const sr=await fetch('/api/status');const ss=await sr.json();
      const rc=ss.refresh_count||0;
      if(rc>knownRefreshCount){
        knownRefreshCount=rc;
        console.log('Backend refreshed — rebuilding graph');
        await fetchAll();
        buildGraph();
        updateUI();
        return;
      }
      updateRefreshInfo(ss.last_refresh);
    }catch(e){}
    await fetchAll();
    updateUI();
  },60000);
}
function delay(ms){return new Promise(r=>setTimeout(r,ms))}

function updateRefreshInfo(lr){
  const el=document.getElementById('refresh-info');
  if(!el||!lr)return;
  try{
    const d=new Date(lr);
    const ago=Math.round((Date.now()-d.getTime())/60000);
    if(ago<60) el.textContent='updated '+ago+'m ago';
    else el.textContent='updated '+Math.round(ago/60)+'h ago';
  }catch(e){el.textContent='';}
}

/* ── DATA ─────────────────────────────────────────── */
async function fetchAll(){
  const [g,sp,si,m]=await Promise.all([
    fetch('/api/graph?threshold='+threshold).then(r=>r.json()),
    fetch('/api/spectrum').then(r=>r.json()),
    fetch('/api/signals').then(r=>r.json()),
    fetch('/api/market').then(r=>r.json()),
  ]);
  graphD=g;specD=sp;sigD=si;mktD=m;
}

/* ── SCENE ────────────────────────────────────────── */
function initScene(){
  clock=new THREE.Clock();
  scene=new THREE.Scene();
  scene.fog=new THREE.FogExp2(0x000000,0.006);

  camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.1,800);
  camera.position.set(0,25,75);

  renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  renderer.setSize(innerWidth,innerHeight);
  renderer.domElement.id='c';
  document.getElementById('app').prepend(renderer.domElement);

  labelRenderer=new THREE.CSS2DRenderer();
  labelRenderer.setSize(innerWidth,innerHeight);
  labelRenderer.domElement.style.position='fixed';
  labelRenderer.domElement.style.top='0';
  labelRenderer.domElement.style.left='0';
  labelRenderer.domElement.style.pointerEvents='none';
  labelRenderer.domElement.style.zIndex='1';
  document.getElementById('app').appendChild(labelRenderer.domElement);

  controls=new THREE.OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true;controls.dampingFactor=0.04;
  controls.autoRotate=true;controls.autoRotateSpeed=0.25;
  controls.maxDistance=180;controls.minDistance=15;
  controls.enablePan=false;

  scene.add(new THREE.AmbientLight(0x111122,3));
  const pl=new THREE.PointLight(0xffffff,0.8,250);pl.position.set(40,60,40);scene.add(pl);
  const pl2=new THREE.PointLight(0x3355ff,0.3,200);pl2.position.set(-40,-30,-40);scene.add(pl2);

  gridMesh=new THREE.GridHelper(140,28,0x111133,0x111133);
  gridMesh.position.y=-22;gridMesh.material.opacity=0.08;gridMesh.material.transparent=true;
  scene.add(gridMesh);

  const rg=new THREE.RingGeometry(40,41,96);
  const rm=new THREE.MeshBasicMaterial({color:0x00e5ff,transparent:true,opacity:0.04,side:THREE.DoubleSide});
  ringMesh=new THREE.Mesh(rg,rm);ringMesh.rotation.x=-Math.PI/2;ringMesh.position.y=-22;
  scene.add(ringMesh);

  const pc=800,pp=new Float32Array(pc*3);
  for(let i=0;i<pc;i++){pp[i*3]=(Math.random()-.5)*250;pp[i*3+1]=(Math.random()-.5)*250;pp[i*3+2]=(Math.random()-.5)*250;}
  const pg=new THREE.BufferGeometry();pg.setAttribute('position',new THREE.BufferAttribute(pp,3));
  const pm=new THREE.PointsMaterial({size:.25,color:0x181830,transparent:true,opacity:.6,sizeAttenuation:true});
  particleSys=new THREE.Points(pg,pm);scene.add(particleSys);

  nodeGroup=new THREE.Group();linkGroup=new THREE.Group();
  scene.add(linkGroup);scene.add(nodeGroup);

  window.addEventListener('resize',onResize);
  renderer.domElement.addEventListener('pointermove',onPointerMove);
  renderer.domElement.addEventListener('pointerdown',onPointerDown);
  document.getElementById('th-sl').addEventListener('input',onThreshold);
}

/* ── GLOW TEXTURE ─────────────────────────────────── */
const _gc=document.createElement('canvas');_gc.width=128;_gc.height=128;
const _gx=_gc.getContext('2d');
const _gg=_gx.createRadialGradient(64,64,0,64,64,64);
_gg.addColorStop(0,'rgba(255,255,255,.9)');_gg.addColorStop(.2,'rgba(255,255,255,.35)');
_gg.addColorStop(.5,'rgba(255,255,255,.06)');_gg.addColorStop(1,'rgba(255,255,255,0)');
_gx.fillStyle=_gg;_gx.fillRect(0,0,128,128);
const glowTex=new THREE.CanvasTexture(_gc);

/* ── MAKE FAT LINE ────────────────────────────────── */
function makeFatLine(ld){
  const geo=new THREE.LineGeometry();
  geo.setPositions([0,0,0, 0,0,0]);
  const mat=new THREE.LineMaterial({
    color:corrToHex(ld.corr),
    linewidth:corrWidth(ld.abs_corr),
    transparent:true,
    opacity:corrOpacity(ld.abs_corr),
    resolution:viewportRes,
    dashed:false,
  });
  const line=new THREE.Line2(geo,mat);
  line.computeLineDistances();
  line.userData=ld;
  line.userData._baseOpacity=corrOpacity(ld.abs_corr);
  line.userData._baseWidth=corrWidth(ld.abs_corr);
  return line;
}

/* ── BUILD GRAPH ──────────────────────────────────── */
function buildGraph(){
  if(!graphD||!graphD.nodes)return;
  while(nodeGroup.children.length)nodeGroup.remove(nodeGroup.children[0]);
  while(linkGroup.children.length){
    const c=linkGroup.children[0];
    if(c.geometry)c.geometry.dispose();
    if(c.material)c.material.dispose();
    linkGroup.remove(c);
  }
  nodeMeshes=[];linkLines=[];nodeMap={};labelMap={};simNodes=[];simLinks=[];

  const nodes=graphD.nodes;
  const n=nodes.length;

  for(let i=0;i<n;i++){
    const y=1-(i/(n-1))*2;
    const rad=Math.sqrt(1-y*y);
    const th=i*2.399963;
    const R=28;
    nodes[i]._x=R*Math.cos(th)*rad;nodes[i]._y=R*y;nodes[i]._z=R*Math.sin(th)*rad;
    nodes[i]._vx=0;nodes[i]._vy=0;nodes[i]._vz=0;
  }
  simNodes=nodes;

  nodes.forEach(nd=>{
    const col=CATC[nd.cat]||0x607D8B;
    const r=0.6+nd.loading*10;
    const grp=new THREE.Group();

    const geo=new THREE.IcosahedronGeometry(r,2);
    const mat=new THREE.MeshPhongMaterial({color:col,emissive:col,emissiveIntensity:0.6,
      shininess:100,transparent:true,opacity:.92});
    const mesh=new THREE.Mesh(geo,mat);
    grp.add(mesh);

    const wf=new THREE.LineSegments(new THREE.WireframeGeometry(geo),
      new THREE.LineBasicMaterial({color:col,transparent:true,opacity:.18}));
    grp.add(wf);

    const sm=new THREE.SpriteMaterial({map:glowTex,color:col,transparent:true,
      blending:THREE.AdditiveBlending,depthWrite:false,opacity:.35});
    const spr=new THREE.Sprite(sm);spr.scale.set(r*6,r*6,1);
    grp.add(spr);

    const labelDiv=document.createElement('div');
    labelDiv.className='node-label';
    labelDiv.innerHTML=
      '<div class="node-name" style="color:'+nd.color+'">'+nd.label+'</div>'+
      '<div class="node-tag" style="color:'+nd.color+'">'+nd.catLabel+'</div>';
    const cssLabel=new THREE.CSS2DObject(labelDiv);
    cssLabel.position.set(0, r+1.6, 0);
    grp.add(cssLabel);

    grp.position.set(nd._x,nd._y,nd._z);
    grp.userData={...nd,radius:r,baseMat:mat,sprMat:sm,labelDiv:labelDiv};
    nodeGroup.add(grp);
    nodeMeshes.push(grp);
    nodeMap[nd.id]=grp;
    labelMap[nd.id]=labelDiv;
  });

  graphD.links.forEach(ld=>{
    const line=makeFatLine(ld);
    linkGroup.add(line);
    linkLines.push(line);
    simLinks.push({sId:ld.source,tId:ld.target,ac:ld.abs_corr});
  });

  simAlpha=1;
}

/* ── CORRELATION LEGEND BAR ───────────────────────── */
function drawCorrLegendBar(){
  const cv=document.getElementById('corr-bar-c');
  if(!cv)return;
  const ctx=cv.getContext('2d'),w=cv.width,h=cv.height;
  ctx.clearRect(0,0,w,h);
  for(let x=0;x<w;x++){
    const c=(x/w)*2-1;
    ctx.fillStyle=corrToCSS(c);
    ctx.fillRect(x,0,1,h);
  }
  ctx.globalCompositeOperation='destination-in';
  ctx.beginPath();
  ctx.roundRect(0,0,w,h,5);
  ctx.fill();
  ctx.globalCompositeOperation='source-over';
}

/* ── FORCE SIMULATION ─────────────────────────────── */
function simStep(){
  const ns=simNodes,nl=simLinks;
  if(!ns.length)return;
  const rep=700,att=0.006,cen=0.01,damp=0.88;

  ns.forEach(n=>{n._fx=0;n._fy=0;n._fz=0;});

  for(let i=0;i<ns.length;i++){
    for(let j=i+1;j<ns.length;j++){
      const a=ns[i],b=ns[j];
      const dx=a._x-b._x,dy=a._y-b._y,dz=a._z-b._z;
      const d2=dx*dx+dy*dy+dz*dz;
      const d=Math.sqrt(d2)+.5;
      const f=rep/d2;
      const fx=f*dx/d,fy=f*dy/d,fz=f*dz/d;
      a._fx+=fx;a._fy+=fy;a._fz+=fz;
      b._fx-=fx;b._fy-=fy;b._fz-=fz;
    }
  }

  nl.forEach(l=>{
    const s=nodeMap[l.sId],t=nodeMap[l.tId];
    if(!s||!t)return;
    const si2=ns.find(n=>n.id===l.sId),ti=ns.find(n=>n.id===l.tId);
    if(!si2||!ti)return;
    const dx=ti._x-si2._x,dy=ti._y-si2._y,dz=ti._z-si2._z;
    const d=Math.sqrt(dx*dx+dy*dy+dz*dz)+.1;
    const ideal=18*(1-l.ac);
    const f=att*l.ac*(d-ideal);
    const fx=f*dx/d,fy=f*dy/d,fz=f*dz/d;
    si2._fx+=fx;si2._fy+=fy;si2._fz+=fz;
    ti._fx-=fx;ti._fy-=fy;ti._fz-=fz;
  });

  ns.forEach(n=>{
    n._fx-=cen*n._x;n._fy-=cen*n._y;n._fz-=cen*n._z;
    n._vx=(n._vx+n._fx*.3)*damp*simAlpha;
    n._vy=(n._vy+n._fy*.3)*damp*simAlpha;
    n._vz=(n._vz+n._fz*.3)*damp*simAlpha;
    n._x+=n._vx;n._y+=n._vy;n._z+=n._vz;
  });
  if(simAlpha>0.001)simAlpha*=0.997;
}

function syncPositions(){
  simNodes.forEach(nd=>{
    const m=nodeMap[nd.id];
    if(m){m.position.set(nd._x,nd._y,nd._z);}
  });
  linkLines.forEach(ln=>{
    const s=nodeMap[ln.userData.source],t=nodeMap[ln.userData.target];
    if(!s||!t)return;
    ln.geometry.setPositions([
      s.position.x, s.position.y, s.position.z,
      t.position.x, t.position.y, t.position.z,
    ]);
    ln.computeLineDistances();
  });
}

/* ── ANIMATION ────────────────────────────────────── */
function animate(){
  requestAnimationFrame(animate);
  const t=clock.getElapsedTime();

  for(let i=0;i<3;i++)simStep();
  syncPositions();

  nodeMeshes.forEach((grp,i)=>{
    const ud=grp.userData;
    const pulse=.85+.15*Math.sin(t*1.8+i*.7);
    ud.baseMat.emissiveIntensity=0.6*pulse;
    ud.sprMat.opacity=0.3*pulse;
  });

  if(ringMesh){
    ringMesh.material.opacity=0.03+0.015*Math.sin(t*1.2);
    ringMesh.material.color.set(regimeColor);
  }

  if(particleSys) particleSys.rotation.y=t*0.008;

  if(targetCam){
    camera.position.lerp(targetCam,0.04);
    controls.target.lerp(targetLook,0.04);
    if(camera.position.distanceTo(targetCam)<1){targetCam=null;targetLook=null;}
  }

  controls.update();
  renderer.render(scene,camera);
  labelRenderer.render(scene,camera);
}

/* ── INTERACTION ──────────────────────────────────── */
function onPointerMove(e){
  const rect=renderer.domElement.getBoundingClientRect();
  pointer.x=((e.clientX-rect.left)/rect.width)*2-1;
  pointer.y=-((e.clientY-rect.top)/rect.height)*2+1;

  raycaster.setFromCamera(pointer,camera);
  const cores=nodeMeshes.map(g=>g.children[0]);
  const hits=raycaster.intersectObjects(cores);

  const tip=document.getElementById('tip');
  if(hits.length>0){
    const grp=hits[0].object.parent;
    const d=grp.userData;
    hovered=grp;
    document.getElementById('tip-name').textContent=d.label+' · '+d.catLabel;
    document.getElementById('tip-name').style.color=d.color;
    document.getElementById('tip-body').innerHTML=
      tr('Category',d.catLabel)+tr('Loading',(d.loading*100).toFixed(1)+'%')
      +tr('20d Return',pct(d.ret20))+tr('5d Return',pct(d.ret5))
      +tr('Volatility',(d.vol*100).toFixed(1)+'%')
      +tr('Connections',d.degree);
    tip.style.display='';
    tip.style.left=Math.min(e.clientX+14,innerWidth-200)+'px';
    tip.style.top=Math.min(e.clientY-14,innerHeight-200)+'px';
    renderer.domElement.style.cursor='pointer';
  }else{
    hovered=null;
    tip.style.display='none';
    renderer.domElement.style.cursor='default';
  }

  nodeMeshes.forEach(g=>{
    const linked=hovered&&(g===hovered||isLinked(hovered.userData.id,g.userData.id));
    const dim=hovered&&!linked;
    g.userData.baseMat.opacity=dim?.15:.92;
    g.userData.labelDiv.style.opacity=dim?'0.08':'1';
  });
  linkLines.forEach(ln=>{
    const ud=ln.userData;
    if(!hovered){
      ln.material.color.set(corrToHex(ud.corr));
      ln.material.opacity=ud._baseOpacity;
      ln.material.linewidth=ud._baseWidth;
      return;
    }
    const h=hovered.userData.id;
    const on=ud.source===h||ud.target===h;
    if(on){
      ln.material.color.set(corrToHex(ud.corr));
      ln.material.opacity=Math.min(1,ud._baseOpacity*1.8);
      ln.material.linewidth=ud._baseWidth*1.6;
    }else{
      ln.material.opacity=0.015;
      ln.material.linewidth=0.5;
    }
  });
}

function isLinked(a,b){
  if(!graphD)return false;
  return graphD.links.some(l=>(l.source===a&&l.target===b)||(l.source===b&&l.target===a));
}

function onPointerDown(){
  if(!hovered)return;
  if(focused===hovered){
    focused=null;controls.autoRotate=true;
    targetCam=new THREE.Vector3(0,25,75);
    targetLook=new THREE.Vector3(0,0,0);
    return;
  }
  focused=hovered;controls.autoRotate=false;
  const p=focused.position.clone();
  targetCam=p.clone().add(new THREE.Vector3(12,8,12));
  targetLook=p.clone();
}

function tr(l,v){return '<div class="tip-r"><span>'+l+'</span><span class="tip-v">'+v+'</span></div>';}
function pct(v){
  const s=(v*100).toFixed(1)+'%';
  return '<span class="'+(v>=0?'pos':'neg')+'">'+s+'</span>';
}

function onResize(){
  camera.aspect=innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight);
  labelRenderer.setSize(innerWidth,innerHeight);
  viewportRes.set(innerWidth,innerHeight);
  linkLines.forEach(ln=>{
    ln.material.resolution.set(innerWidth,innerHeight);
  });
}

async function onThreshold(){
  threshold=parseFloat(this.value);
  document.getElementById('th-v').textContent=threshold.toFixed(2);
  const r=await fetch('/api/graph?threshold='+threshold);
  graphD=await r.json();
  rebuildLinks();
  updateNet();
}

function rebuildLinks(){
  while(linkGroup.children.length){
    const c=linkGroup.children[0];
    if(c.geometry)c.geometry.dispose();
    if(c.material)c.material.dispose();
    linkGroup.remove(c);
  }
  linkLines=[];simLinks=[];
  if(!graphD)return;
  simNodes.forEach(n=>n.degree=0);

  graphD.links.forEach(ld=>{
    const line=makeFatLine(ld);
    linkGroup.add(line);
    linkLines.push(line);
    simLinks.push({sId:ld.source,tId:ld.target,ac:ld.abs_corr});

    const sn=simNodes.find(n=>n.id===ld.source);
    const tn=simNodes.find(n=>n.id===ld.target);
    if(sn)sn.degree=(sn.degree||0)+1;
    if(tn)tn.degree=(tn.degree||0)+1;
  });
  simAlpha=Math.max(simAlpha,0.3);
}

/* ── UI ───────────────────────────────────────────── */
function updateUI(){
  if(sigD) updateSignals();
  if(specD) drawSpectrum();
  if(mktD) drawMarket();
  if(graphD) updateNet();
  if(sigD&&sigD.history) drawHistory();
}

function updateSignals(){
  const s=sigD;
  document.getElementById('regime-txt').textContent=s.regime;
  document.getElementById('regime-txt').style.color=s.rc;
  const bd=document.getElementById('badge');
  bd.textContent=s.regime;bd.style.color=s.rc;
  bd.style.background=s.rc+'18';bd.style.border='1px solid '+s.rc+'33';
  regimeColor=parseInt(s.rc.replace('#',''),16);

  document.getElementById('auc-row').innerHTML=
    '<div class="auc-item"><div class="auc-v" style="color:#00e096">'+s.rauc.toFixed(3)+'</div><div class="auc-l">Rally AUC</div></div>'+
    '<div class="auc-item"><div class="auc-v" style="color:#ff3d71">'+s.cauc.toFixed(3)+'</div><div class="auc-l">Crash AUC</div></div>';

  document.getElementById('gauges').innerHTML=
    gauge('Rally',s.rr,s.rr>=.78?'#00e096':'#00e5ff')+
    gauge('Crash',s.cr,s.cr>=.6?'#ff3d71':(s.cr>=.4?'#ffaa00':'#00e096'));

  const ec=s.exp===0?'#ff3d71':s.exp>=1?'#00e096':'#ffaa00';
  document.getElementById('exp-v').textContent=s.exp.toFixed(1)+'x';
  document.getElementById('exp-v').style.color=ec;
  const ef=document.getElementById('exp-fill');
  ef.style.width=(s.exp/1.5*100)+'%';ef.style.background=ec;
  const st=document.getElementById('strat-txt');
  st.textContent=s.strat;st.style.borderLeftColor=s.rc;

  document.getElementById('alerts-row').innerHTML=
    s.alerts.map(a=>'<div class="al '+a.lv+'">'+a.ic+' '+a.msg+'</div>').join('');
}

function gauge(label,val,color){
  const v=Math.max(0,Math.min(1,val));
  const r=26,sw=3,sz=64,cx=sz/2,cy=sz/2;
  const c=2*Math.PI*r,off=c*(1-v);
  return '<div class="gauge"><svg width="'+sz+'" height="'+sz+'" viewBox="0 0 '+sz+' '+sz+'">'
    +'<circle cx="'+cx+'" cy="'+cy+'" r="'+r+'" fill="none" stroke="rgba(255,255,255,.04)" stroke-width="'+sw+'"/>'
    +'<circle cx="'+cx+'" cy="'+cy+'" r="'+r+'" fill="none" stroke="'+color+'" stroke-width="'+sw+'"'
    +' stroke-dasharray="'+c+'" stroke-dashoffset="'+off+'" transform="rotate(-90 '+cx+' '+cy+')" stroke-linecap="round" style="transition:stroke-dashoffset .8s"/>'
    +'<text x="'+cx+'" y="'+(cy+1)+'" text-anchor="middle" fill="'+color+'" font-size="13" font-weight="600" font-family="Inter">'+Math.round(v*100)+'</text>'
    +'<text x="'+cx+'" y="'+(cy+11)+'" text-anchor="middle" fill="rgba(255,255,255,.2)" font-size="7" font-family="Inter">'+label+'</text>'
    +'</svg></div>';
}

/* ── CANVAS CHARTS ────────────────────────────────── */
function drawSpectrum(){
  const cv=document.getElementById('spec-c');
  if(!cv||!specD)return;
  const ctx=cv.getContext('2d'),w=cv.width,h=cv.height;
  ctx.clearRect(0,0,w,h);
  const ev=specD.eigenvalues,mx=Math.max(...ev);
  const bw=w/ev.length-1;
  ev.forEach((v,i)=>{
    const bh=(v/mx)*h*.85;
    ctx.fillStyle=v>specD.mp?'#ff3d71':(i<5?'rgba(76,139,245,.7)':'rgba(255,255,255,.06)');
    ctx.fillRect(i*(bw+1),h-bh,bw,bh);
  });
  const my=h-(specD.mp/mx)*h*.85;
  ctx.strokeStyle='#ffaa00';ctx.lineWidth=1;ctx.setLineDash([3,3]);
  ctx.beginPath();ctx.moveTo(0,my);ctx.lineTo(w,my);ctx.stroke();ctx.setLineDash([]);

  document.getElementById('spec-st').innerHTML=
    si('AR₁',(specD.ar1*100).toFixed(1)+'%')+si('AR₅',(specD.ar5*100).toFixed(1)+'%')
    +si('Eff Rank',specD.eff_rank.toFixed(1))+si('Gap',specD.gap.toFixed(1));
}

function drawMarket(){
  const cv=document.getElementById('mkt-c');
  if(!cv||!mktD||!mktD.prices)return;
  const ctx=cv.getContext('2d'),w=cv.width,h=cv.height;
  ctx.clearRect(0,0,w,h);
  const p=mktD.prices,mn=Math.min(...p),mx=Math.max(...p),rng=mx-mn||1;
  ctx.strokeStyle='#4C8BF5';ctx.lineWidth=1.2;
  ctx.beginPath();
  p.forEach((v,i)=>{
    const x=i/(p.length-1)*w,y=h-((v-mn)/rng)*h*.88-h*.06;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke();
  ctx.lineTo(w,h);ctx.lineTo(0,h);ctx.closePath();
  ctx.fillStyle='rgba(76,139,245,.04)';ctx.fill();

  const rc=mktD.r20>=0?'pos':'neg';
  document.getElementById('mkt-st').innerHTML=
    si('1d',(mktD.r1*100).toFixed(2)+'%')+si('5d',(mktD.r5*100).toFixed(1)+'%')
    +si('20d','<span class="'+rc+'">'+(mktD.r20*100).toFixed(1)+'%</span>')
    +si('Vol',(mktD.vol*100).toFixed(1)+'%');
}

function drawHistory(){
  const cv=document.getElementById('hist-c');
  if(!cv||!sigD||!sigD.history)return;
  const ctx=cv.getContext('2d'),w=cv.width,h=cv.height;
  ctx.clearRect(0,0,w,h);
  const hi=sigD.history,n=hi.rr.length;

  ctx.strokeStyle='rgba(255,255,255,.06)';ctx.lineWidth=1;ctx.setLineDash([2,3]);
  [0.60,0.90].forEach(th=>{
    const y=h-th*h;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();
  });
  ctx.setLineDash([]);

  ctx.strokeStyle='#00e096';ctx.lineWidth=1.2;ctx.beginPath();
  hi.rr.forEach((v,i)=>{const x=i/(n-1)*w,y=h-v*h;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
  ctx.stroke();

  ctx.strokeStyle='#ff3d71';ctx.lineWidth=1.2;ctx.beginPath();
  hi.cr.forEach((v,i)=>{const x=i/(n-1)*w,y=h-v*h;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
  ctx.stroke();

  ctx.font='8px Inter';ctx.fillStyle='#00e096';ctx.fillText('Rally',4,10);
  ctx.fillStyle='#ff3d71';ctx.fillText('Crash',36,10);
}

function updateNet(){
  if(!graphD||!graphD.topo)return;
  const t=graphD.topo;
  document.getElementById('net-st').innerHTML=
    si('Mean |ρ|',t.mean_corr.toFixed(3))+si('Edges',t.n_edges+'/'+t.max_edges)
    +si('Density',(t.density*100).toFixed(1)+'%')+si('>0.7',(t.frac70*100).toFixed(1)+'%');
}

function si(l,v){return '<div class="si"><div class="sv">'+v+'</div><div class="sl">'+l+'</div></div>';}

/* ── GO ───────────────────────────────────────────── */
boot();
})();
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  CGECD 3D Visualization")
    print(f"  Auto-refresh every {REFRESH_INTERVAL // 3600}h")
    print("  http://localhost:80")
    print("=" * 60)
    threading.Thread(target=refresh_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=80, debug=False, threaded=True)