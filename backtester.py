#!/usr/bin/env python3
"""
CGECD Backtester — Walk-forward strategy backtest
===================================================

Trains CGECD models for Rally + Crash detection using walk-forward splits,
generates trading signals, and computes strategy performance.

Usage:
    python backtester.py              # Run backtest, save results
    python backtester.py --json       # Output JSON to stdout

Output:
    results/backtest/
        backtest_results.json   — trades, equity, stats
        equity_curve.png        — equity curve plot
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

from config import Config
from algorithm import (
    load_data, build_spectral_features, build_traditional_features,
    compute_all_targets,
)

OUT = Path("results/backtest")
OUT.mkdir(parents=True, exist_ok=True)


def run_backtest(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    target_up: pd.Series,
    target_down: pd.Series,
    config: Config,
    hold_days: int = 10,
    up_threshold: float = 0.4,
    down_threshold: float = 0.4,
):
    """
    Walk-forward backtest with non-overlapping positions.

    Parameters
    ----------
    features : combined feature DataFrame
    prices : multi-asset price DataFrame
    target_up : binary Rally target
    target_down : binary Crash target
    config : Config object
    hold_days : position holding period (matches 10-day target horizon)
    up_threshold : probability threshold to go long
    down_threshold : probability threshold to go short
    """
    market = prices["SP500"] if "SP500" in prices.columns else prices.iloc[:, 0]
    market_ret = market.pct_change().fillna(0)

    # Align indices
    common_idx = features.dropna(thresh=int(len(features.columns) * 0.5)).index
    common_idx = common_idx.intersection(target_up.dropna().index)
    common_idx = common_idx.intersection(target_down.dropna().index)
    X = features.loc[common_idx]

    train_size = int(config.train_years * 252)
    test_size = int(config.test_months * 21)
    gap = config.gap_days
    available = len(X) - train_size - gap - test_size
    step = max(test_size, available // config.n_splits)

    rf_params = dict(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_leaf=config.rf_min_samples_leaf,
        min_samples_split=config.rf_min_samples_split,
        class_weight="balanced_subsample",
        random_state=config.random_seed,
        n_jobs=-1,
    )

    all_trades = []
    daily_signals = []

    for fold in range(config.n_splits):
        start = fold * step
        train_end = start + train_size
        test_start = train_end + gap
        test_end = min(test_start + test_size, len(X))
        if test_end > len(X):
            break

        X_tr = np.nan_to_num(X.iloc[start:train_end].values, nan=0, posinf=0, neginf=0)
        X_te = np.nan_to_num(X.iloc[test_start:test_end].values, nan=0, posinf=0, neginf=0)
        y_up_tr = target_up.loc[X.index[start:train_end]].values
        y_down_tr = target_down.loc[X.index[start:train_end]].values
        test_dates = X.iloc[test_start:test_end].index

        if len(np.unique(y_up_tr)) < 2 or len(np.unique(y_down_tr)) < 2:
            continue

        scaler = RobustScaler()
        Xs_tr = scaler.fit_transform(X_tr)
        Xs_te = scaler.transform(X_te)

        rf_up = RandomForestClassifier(**rf_params)
        rf_up.fit(Xs_tr, y_up_tr)
        prob_up = rf_up.predict_proba(Xs_te)[:, 1]

        rf_down = RandomForestClassifier(**rf_params)
        rf_down.fit(Xs_tr, y_down_tr)
        prob_down = rf_down.predict_proba(Xs_te)[:, 1]

        # Record daily signals
        for i in range(len(test_dates)):
            daily_signals.append({
                "date": str(test_dates[i].date()),
                "prob_up": float(prob_up[i]),
                "prob_down": float(prob_down[i]),
                "fold": fold,
            })

        # Generate trades (non-overlapping)
        pos_end = -1
        for i in range(len(test_dates)):
            if i <= pos_end:
                continue
            end_i = min(i + hold_days, len(test_dates) - 1)
            if end_i <= i:
                continue
            tr = market_ret.reindex(test_dates[i + 1 : end_i + 1]).values
            if len(tr) == 0:
                continue

            direction = None
            prob = 0.0
            if prob_up[i] > up_threshold and prob_up[i] >= prob_down[i]:
                direction = "LONG"
                prob = float(prob_up[i])
                pnl = float(np.sum(tr))
            elif prob_down[i] > down_threshold:
                direction = "SHORT"
                prob = float(prob_down[i])
                pnl = float(-np.sum(tr))
            else:
                continue

            all_trades.append({
                "entry": str(test_dates[i].date()),
                "exit": str(test_dates[end_i].date()),
                "direction": direction,
                "probability": round(prob, 4),
                "pnl": round(pnl, 6),
                "fold": fold,
            })
            pos_end = end_i

    if not all_trades:
        return {"error": "No trades generated", "trades": [], "stats": {}}

    tdf = pd.DataFrame(all_trades)
    cum = tdf["pnl"].cumsum()

    # Buy-and-hold benchmark
    first_date = pd.Timestamp(tdf["entry"].iloc[0])
    last_date = pd.Timestamp(tdf["exit"].iloc[-1])
    bh_ret = float(market.loc[last_date] / market.loc[first_date] - 1) if first_date in market.index and last_date in market.index else None

    stats = {
        "total_return": round(float(cum.iloc[-1]), 4),
        "n_trades": len(tdf),
        "n_long": int((tdf["direction"] == "LONG").sum()),
        "n_short": int((tdf["direction"] == "SHORT").sum()),
        "win_rate": round(float((tdf["pnl"] > 0).mean()), 4),
        "avg_pnl": round(float(tdf["pnl"].mean()), 6),
        "sharpe": round(float(tdf["pnl"].mean() / (tdf["pnl"].std() + 1e-10) * np.sqrt(252 / hold_days)), 2),
        "max_drawdown": round(float((cum - cum.cummax()).min()), 4),
        "buy_hold_return": round(bh_ret, 4) if bh_ret is not None else None,
        "hold_days": hold_days,
        "up_threshold": up_threshold,
        "down_threshold": down_threshold,
    }

    return {
        "trades": all_trades,
        "daily_signals": daily_signals,
        "equity": cum.values.tolist(),
        "stats": stats,
    }


def main():
    t0 = datetime.now()
    json_mode = "--json" in sys.argv

    if not json_mode:
        print("=" * 60)
        print("  CGECD BACKTESTER")
        print("=" * 60)

    cfg = Config()

    if not json_mode:
        print("[1/3] Loading data & features ...")
    prices, returns = load_data(cfg)
    spectral = build_spectral_features(returns, cfg)
    traditional = build_traditional_features(prices, returns)
    combined = pd.concat([spectral, traditional], axis=1)

    all_targets = compute_all_targets(prices)
    target_up = all_targets["up_3pct_10d"]
    target_down = all_targets["drawdown_7pct_10d"]

    if not json_mode:
        print(f"\n[2/3] Running backtest ({cfg.n_splits} folds, {cfg.train_years}yr train) ...")
    result = run_backtest(combined, prices, target_up, target_down, cfg)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    # Save
    with open(OUT / "backtest_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    if json_mode:
        print(json.dumps(result, indent=2, default=str))
    else:
        s = result["stats"]
        print(f"\n[3/3] Results:")
        print(f"  Total Return:   {s['total_return']:.1%}")
        print(f"  Sharpe Ratio:   {s['sharpe']:.2f}")
        print(f"  Max Drawdown:   {s['max_drawdown']:.1%}")
        print(f"  Win Rate:       {s['win_rate']:.0%}")
        print(f"  Trades:         {s['n_trades']} ({s['n_long']}L / {s['n_short']}S)")
        if s.get("buy_hold_return") is not None:
            print(f"  Buy & Hold:     {s['buy_hold_return']:.1%}")
        print(f"\n  Saved to {OUT}/")
        print(f"  Elapsed: {datetime.now() - t0}")


if __name__ == "__main__":
    main()
