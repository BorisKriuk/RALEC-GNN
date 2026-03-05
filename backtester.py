#!/usr/bin/env python3
"""
Backtester module for CGECD trading strategy.
==============================================
Standalone module extracted from investigate_v3.py.
Provides:
  - SignalRule: dataclass for defining trading signals
  - Strategy: class combining signal rules with RF models
  - Backtester: walk-forward backtesting engine
"""

import warnings
warnings.filterwarnings("ignore")

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

from config import Config
from algorithm import (
    load_data,
    build_spectral_features,
    build_traditional_features,
    _compute_fold_schedule,
)


# =================================================================
# SIGNAL RULES
# =================================================================
@dataclass
class SignalRule:
    """A trading signal rule with feature conditions and RF probability gate."""
    name: str
    direction: str  # "LONG" or "SHORT"
    feature_conditions: List[Tuple[str, str, float]]  # [(feature, "<"/">", percentile)]
    min_rf_prob: float = 0.3

    def to_dict(self):
        return {
            "name": self.name,
            "direction": self.direction,
            "conditions": self.feature_conditions,
            "min_prob": self.min_rf_prob,
        }


DEFAULT_RULES = [
    SignalRule(
        name="UP: Oversold bounce",
        direction="LONG",
        feature_conditions=[("drawdown_60d", "<", 25), ("volatility_10d", "<", 75)],
        min_rf_prob=0.3,
    ),
    SignalRule(
        name="UP: Deep drawdown + low momentum",
        direction="LONG",
        feature_conditions=[("drawdown_60d", "<", 25), ("return_60d", "<", 10)],
        min_rf_prob=0.3,
    ),
    SignalRule(
        name="DOWN: Vol spike + spectral",
        direction="SHORT",
        feature_conditions=[("volatility_20d", ">", 75), ("lambda_2", ">", 50)],
        min_rf_prob=0.3,
    ),
    SignalRule(
        name="DOWN: Drawdown + vol cluster",
        direction="SHORT",
        feature_conditions=[("drawdown_60d", "<", 25), ("volatility_10d", "<", 75)],
        min_rf_prob=0.3,
    ),
]


# =================================================================
# STRATEGY
# =================================================================
class Strategy:
    """Signal-based trading strategy with RF probability confirmation."""

    def __init__(self, rules: List[SignalRule] = None, holding_period: int = 3,
                 sizing_method: str = "binary"):
        self.rules = rules or DEFAULT_RULES
        self.holding_period = holding_period
        self.sizing_method = sizing_method

    def check_signal(self, feature_values: Dict[str, float],
                     percentile_cache: Dict[Tuple[str, float], float],
                     prob_up: float, prob_down: float) -> Optional[Dict]:
        """Check all rules against current feature values. Returns first matching signal or None."""
        for rule in self.rules:
            conditions_met = True
            for feat, op, pctile in rule.feature_conditions:
                threshold = percentile_cache.get((feat, pctile), None)
                if threshold is None:
                    conditions_met = False
                    break
                val = feature_values.get(feat, 0)
                if op == "<" and val >= threshold:
                    conditions_met = False
                    break
                elif op == ">" and val <= threshold:
                    conditions_met = False
                    break

            if not conditions_met:
                continue

            rf_prob = prob_up if rule.direction == "LONG" else prob_down
            if rf_prob < rule.min_rf_prob:
                continue

            # Compute position size
            if self.sizing_method == "linear":
                size = float(np.clip((rf_prob - rule.min_rf_prob) / 0.5, 0.2, 1.0))
            else:
                size = 1.0

            return {
                "rule": rule.name,
                "direction": rule.direction,
                "rf_prob": float(rf_prob),
                "size": size,
            }

        return None

    def get_rules_summary(self) -> List[Dict]:
        return [r.to_dict() for r in self.rules]


# =================================================================
# BACKTESTER
# =================================================================
class Backtester:
    """Walk-forward backtesting engine."""

    def __init__(self, config: Config, strategy: Strategy = None):
        self.config = config
        self.strategy = strategy or Strategy()

    def run(self, features: pd.DataFrame, prices: pd.DataFrame,
            target_up: pd.Series, target_down: pd.Series) -> Dict:
        """Execute walk-forward backtest."""
        market = prices["SP500"] if "SP500" in prices.columns else prices.iloc[:, 0]
        market_returns = market.pct_change().fillna(0)

        valid = target_up.dropna().index.intersection(target_down.dropna().index)
        X = features.reindex(valid).ffill().fillna(0)
        feat_names = list(X.columns)

        train_size = int(self.config.train_years * 252)
        test_size = int(self.config.test_months * 21)
        gap = self.config.gap_days
        schedule = _compute_fold_schedule(len(X), train_size, test_size, gap, self.config.n_splits)

        trades = []
        fold_info = []

        for fold_idx, (train_end, test_start, test_end) in enumerate(schedule):
            X_tr = np.nan_to_num(X.iloc[:train_end].values, nan=0, posinf=0, neginf=0)
            y_up_tr = target_up.reindex(valid).iloc[:train_end].values
            y_down_tr = target_down.reindex(valid).iloc[:train_end].values
            X_te = np.nan_to_num(X.iloc[test_start:test_end].values, nan=0, posinf=0, neginf=0)
            test_dates = X.iloc[test_start:test_end].index

            if len(np.unique(y_up_tr)) < 2 or len(np.unique(y_down_tr)) < 2:
                continue

            # Train RF models
            scaler = RobustScaler()
            Xs_tr = scaler.fit_transform(X_tr)
            Xs_te = scaler.transform(X_te)

            rf_params = dict(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_leaf=self.config.rf_min_samples_leaf,
                min_samples_split=self.config.rf_min_samples_split,
                class_weight="balanced_subsample",
                random_state=self.config.random_seed,
                n_jobs=-1,
            )

            rf_up = RandomForestClassifier(**rf_params)
            rf_up.fit(Xs_tr, y_up_tr)
            prob_up = rf_up.predict_proba(Xs_te)[:, 1]

            rf_down = RandomForestClassifier(**rf_params)
            rf_down.fit(Xs_tr, y_down_tr)
            prob_down = rf_down.predict_proba(Xs_te)[:, 1]

            # Percentile thresholds from training data
            percentile_cache = {}
            for feat in feat_names:
                fi = feat_names.index(feat)
                for p in [10, 25, 50, 75, 90]:
                    percentile_cache[(feat, p)] = float(np.percentile(X_tr[:, fi], p))

            # Generate signals
            position_end_idx = -1
            for i in range(len(test_dates)):
                if i <= position_end_idx:
                    continue

                date = test_dates[i]
                fv = {feat_names[j]: float(X_te[i, j]) for j in range(len(feat_names))}

                signal = self.strategy.check_signal(fv, percentile_cache,
                                                     float(prob_up[i]), float(prob_down[i]))
                if signal is None:
                    continue

                end_idx = min(i + self.strategy.holding_period, len(test_dates) - 1)
                if end_idx <= i:
                    continue

                trade_returns = market_returns.reindex(test_dates[i + 1: end_idx + 1]).values
                if len(trade_returns) == 0:
                    continue

                direction_mult = 1.0 if signal["direction"] == "LONG" else -1.0
                trade_pnl = direction_mult * signal["size"] * float(np.sum(trade_returns))

                trades.append({
                    "entry_date": str(date.date()) if hasattr(date, "date") else str(date),
                    "exit_date": str(test_dates[end_idx].date()) if hasattr(test_dates[end_idx], "date") else str(test_dates[end_idx]),
                    "direction": signal["direction"],
                    "rule": signal["rule"],
                    "rf_prob": signal["rf_prob"],
                    "size": signal["size"],
                    "pnl": trade_pnl,
                    "holding_days": end_idx - i,
                    "fold": fold_idx,
                })
                position_end_idx = end_idx

            fold_info.append({
                "fold": fold_idx,
                "train_end": str(X.index[train_end - 1].date()),
                "test_start": str(test_dates[0].date()),
                "test_end": str(test_dates[-1].date()),
                "n_trades": sum(1 for t in trades if t.get("fold") == fold_idx),
            })

        return self._compute_metrics(trades, market_returns, schedule, X, fold_info)

    def _compute_metrics(self, trades, market_returns, schedule, X, fold_info):
        if not trades:
            return {"error": "No trades generated", "trades": [], "metrics": {}, "folds": fold_info}

        trades_df = pd.DataFrame(trades)
        total_pnl = float(trades_df["pnl"].sum())
        n_trades = len(trades_df)
        wins = int((trades_df["pnl"] > 0).sum())
        win_rate = wins / n_trades
        avg_pnl = float(trades_df["pnl"].mean())
        sharpe = float(trades_df["pnl"].mean() / (trades_df["pnl"].std() + 1e-10) * np.sqrt(252 / 3))

        # Buy and hold
        all_test_dates = []
        for _, test_start, test_end in schedule:
            all_test_dates.extend(X.iloc[test_start:test_end].index.tolist())
        bnh_returns = float(market_returns.reindex(all_test_dates).sum())

        # Max drawdown
        cum_pnl = trades_df["pnl"].cumsum()
        max_dd = float((cum_pnl - cum_pnl.cummax()).min())

        metrics = {
            "total_return": total_pnl,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "wins": wins,
            "losses": n_trades - wins,
            "avg_trade_pnl": avg_pnl,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "buy_hold_return": bnh_returns,
            "long_trades": int((trades_df["direction"] == "LONG").sum()),
            "short_trades": int((trades_df["direction"] == "SHORT").sum()),
            "sizing_method": self.strategy.sizing_method,
            "holding_period": self.strategy.holding_period,
        }

        return {
            "trades": trades_df.to_dict("records"),
            "metrics": metrics,
            "folds": fold_info,
            "equity_curve": cum_pnl.values.tolist(),
        }


# =================================================================
# CONVENIENCE: run full backtest from scratch
# =================================================================
def run_full_backtest(sizing_method="binary", holding_period=3, rules=None):
    """Load data, build features, run backtest — returns result dict."""
    config = Config()
    print("Loading data ...")
    prices, returns = load_data(config)

    print("Building features ...")
    spectral = build_spectral_features(returns, config)
    traditional = build_traditional_features(prices, returns)
    combined = pd.concat([spectral, traditional], axis=1)

    market = prices["SP500"] if "SP500" in prices.columns else prices.iloc[:, 0]
    fut_3d = market.pct_change(3).shift(-3)
    target_up = (fut_3d > 0.03).astype(int)
    target_down = (fut_3d < -0.03).astype(int)

    strategy = Strategy(rules=rules, holding_period=holding_period, sizing_method=sizing_method)
    backtester = Backtester(config, strategy)

    print(f"Running backtest (sizing={sizing_method}, hold={holding_period}d) ...")
    result = backtester.run(combined, prices, target_up, target_down)
    return result


if __name__ == "__main__":
    result = run_full_backtest()
    if "error" not in result:
        m = result["metrics"]
        print(f"\nResults: {m['n_trades']} trades, Return={m['total_return']:.1%}, "
              f"Sharpe={m['sharpe_ratio']:.2f}, WinRate={m['win_rate']:.0%}, "
              f"MaxDD={m['max_drawdown']:.1%}")
    else:
        print(f"Error: {result['error']}")
