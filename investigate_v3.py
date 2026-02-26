#!/usr/bin/env python3
"""
investigate_v3.py — Feature Understanding, Trading Strategy, Spectral Explanation
==================================================================================

10 steps across 3 tasks:

Task 1: Improve Feature Understanding
  1. Extended fold thresholds (7 folds for DOWN)
  2. Interaction-aware thresholds for lambda_2
  3. Regime-specific RF models
  4. Task-specific feature subsets

Task 2: Simple Practical Trading Strategy
  5. Formalize signal rules
  6. Walk-forward backtest
  7. Position sizing comparison

Task 3: Explain Why Spectral Features Contribute
  8. Economic narrative with evidence
  9. Event temporal profiles (anatomy of a crash)
  10. Ablation: with vs without eigenvalue features

Output → results/investigate_v3/
"""

import warnings

warnings.filterwarnings("ignore")

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from config import Config
from algorithm import (
    load_data,
    build_spectral_features,
    build_traditional_features,
    _compute_fold_schedule,
    CGECDModel,
    CGECDNoSelModel,
    walk_forward_evaluate,
)
from investigate_v2 import (
    value_thresholds,
    joint_thresholds,
    feature_group,
    PALETTE,
    unbiased_importance,
)

OUT = Path("results/investigate_v3")
OUT.mkdir(parents=True, exist_ok=True)


def _safe(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(">", "gt")
        .replace("%", "pct")
    )


# =================================================================
# STEP 1: EXTENDED FOLD THRESHOLDS
# =================================================================
def step1_extended_thresholds(
    features: pd.DataFrame,
    target: pd.Series,
    feat_names: List[str],
    config: Config,
    shap_ranking: np.ndarray,
    n_folds: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pool test data from last n_folds for robust threshold evaluation."""
    valid = target.dropna().index
    X = features.reindex(valid).ffill().fillna(0)
    y = target.loc[valid]

    train_size = int(config.train_years * 252)
    test_size = int(config.test_months * 21)
    gap = config.gap_days
    schedule = _compute_fold_schedule(len(X), train_size, test_size, gap, config.n_splits)

    folds_to_use = schedule[-n_folds:]
    pooled_X_te = []
    pooled_y_te = []

    for train_end, test_start, test_end in folds_to_use:
        pooled_X_te.append(X.iloc[test_start:test_end].values)
        pooled_y_te.append(y.iloc[test_start:test_end].values)

    # Use last fold's training data for threshold fitting
    last_train_end = folds_to_use[-1][0]
    X_tr = np.nan_to_num(X.iloc[:last_train_end].values, nan=0, posinf=0, neginf=0)
    y_tr = y.iloc[:last_train_end].values
    X_te = np.nan_to_num(np.vstack(pooled_X_te), nan=0, posinf=0, neginf=0)
    y_te = np.concatenate(pooled_y_te)

    n_events = int(y_te.sum())
    print(f"    Pooled {n_folds} folds: {len(y_te)} samples, {n_events} events ({n_events/len(y_te):.1%})")

    thresh_df = value_thresholds(X_tr, y_tr, X_te, y_te, feat_names, top_k=25, feature_ranking=shap_ranking)
    jt_df = joint_thresholds(X_te, y_te, feat_names, shap_ranking, X_tr, top_k=10, n_best=30)

    return thresh_df, jt_df


def plot_step1(thresh_df: pd.DataFrame, jt_df: pd.DataFrame, task: str):
    """Plot extended threshold results."""
    n = min(len(thresh_df), 20)
    fig, ax = plt.subplots(figsize=(14, 10))
    names = [f"{r['feature']} {r['direction']} {r['threshold']:.3f}" for _, r in thresh_df.head(n).iterrows()]
    colors = [PALETTE.get(r["group"], "#95a5a6") for _, r in thresh_df.head(n).iterrows()]

    ax.barh(np.arange(n) - 0.2, thresh_df["precision"].values[:n] * 100, 0.35, color=colors, alpha=0.9, label="Precision %")
    ax.barh(np.arange(n) + 0.2, thresh_df["recall"].values[:n] * 100, 0.35, color=colors, alpha=0.4, label="Recall %")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Percentage (%)")
    ax.set_title(f"Extended Thresholds (7 folds) — {task}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    for i, (_, r) in enumerate(thresh_df.head(n).iterrows()):
        lift = r["precision"] / max(r.get("importance", 0.01), 1e-10) if r["precision"] > 0 else 0
        ax.text(max(r["precision"], r["recall"]) * 100 + 1, i,
                f"{r['events_captured']}/{r['total_events']} events, {r['total_signals']} sig", va="center", fontsize=6)
    plt.tight_layout()
    plt.savefig(OUT / f"extended_thresholds_{_safe(task)}.png", dpi=150, bbox_inches="tight")
    plt.close()


# =================================================================
# STEP 2: INTERACTION-AWARE THRESHOLDS
# =================================================================
def _extract_tree_rules(tree, subset_names, X_test, y_test):
    """Walk a decision tree and extract leaf rules."""
    t = tree.tree_
    rules = []

    def walk(node_id, conditions):
        if t.children_left[node_id] == -1:  # leaf
            vals = t.value[node_id][0]
            n_neg, n_pos = vals[0], vals[1] if len(vals) > 1 else 0
            if n_pos / (n_neg + n_pos + 1e-10) > 0.1:  # positive-leaning leaf
                # Evaluate this leaf on test data
                mask = np.ones(len(X_test), dtype=bool)
                for feat_idx, op, thresh in conditions:
                    if op == "<=":
                        mask &= X_test[:, feat_idx] <= thresh
                    else:
                        mask &= X_test[:, feat_idx] > thresh
                n_sig = mask.sum()
                n_hits = int((mask & (y_test == 1)).sum())
                if n_sig >= 3:
                    cond_str = " AND ".join(
                        f"{subset_names[fi]} {op} {th:.4f}" for fi, op, th in conditions
                    )
                    rules.append({
                        "rule": cond_str,
                        "n_conditions": len(conditions),
                        "precision": n_hits / n_sig if n_sig > 0 else 0,
                        "recall": n_hits / max(y_test.sum(), 1),
                        "n_signals": n_sig,
                        "n_hits": n_hits,
                        "lift": (n_hits / n_sig) / max(y_test.mean(), 1e-10) if n_sig > 0 else 0,
                    })
            return

        feat = t.feature[node_id]
        thresh = t.threshold[node_id]
        walk(t.children_left[node_id], conditions + [(feat, "<=", thresh)])
        walk(t.children_right[node_id], conditions + [(feat, ">", thresh)])

    walk(0, [])
    return rules


def step2_interaction_thresholds(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feat_names: List[str],
    target_features: List[str] = None,
) -> pd.DataFrame:
    """Train depth-2/3 trees to find interaction rules involving target features."""
    if target_features is None:
        target_features = ["lambda_2", "volatility_20d", "drawdown_20d"]

    all_rules = []
    for target_feat in target_features:
        if target_feat not in feat_names:
            continue
        t_idx = feat_names.index(target_feat)

        # Find top-8 correlated features (potential interaction partners)
        corrs = np.abs(np.corrcoef(X_train.T)[t_idx])
        corrs[t_idx] = 0  # exclude self
        partner_idx = np.argsort(corrs)[::-1][:8]
        subset_idx = np.unique(np.concatenate([[t_idx], partner_idx]))
        subset_names = [feat_names[i] for i in subset_idx]

        X_tr_sub = X_train[:, subset_idx]
        X_te_sub = X_test[:, subset_idx]

        for depth in [2, 3]:
            tree = DecisionTreeClassifier(
                max_depth=depth, class_weight="balanced", random_state=42,
                min_samples_leaf=max(5, int(y_train.sum() * 0.05)),
            )
            tree.fit(X_tr_sub, y_train)
            rules = _extract_tree_rules(tree, subset_names, X_te_sub, y_test)
            for r in rules:
                r["target_feature"] = target_feat
                r["depth"] = depth
            all_rules.extend(rules)

    if not all_rules:
        return pd.DataFrame()
    df = pd.DataFrame(all_rules).sort_values("precision", ascending=False).drop_duplicates("rule").reset_index(drop=True)
    return df


def plot_step2(rules_df: pd.DataFrame, task: str):
    if rules_df.empty:
        return
    top = rules_df.head(15)
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = ["#e74c3c" if "lambda_2" in r else "#3498db" for r in top["rule"]]
    ax.barh(range(len(top)), top["precision"].values * 100, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["rule"].values, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Precision (%)")
    ax.set_title(f"Interaction Rules (depth 2-3 trees) — {task}", fontsize=13, fontweight="bold")
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r["precision"] * 100 + 0.5, i,
                f"lift={r['lift']:.1f}x  {r['n_hits']}/{r['n_signals']}  recall={r['recall']:.0%}",
                va="center", fontsize=6)
    plt.tight_layout()
    plt.savefig(OUT / f"interaction_rules_{_safe(task)}.png", dpi=150, bbox_inches="tight")
    plt.close()


# =================================================================
# STEP 3: REGIME-SPECIFIC RF MODELS
# =================================================================
def step3_regime_models(
    features: pd.DataFrame,
    target: pd.Series,
    feat_names: List[str],
    config: Config,
    split_feature: str = "volatility_20d",
) -> Dict:
    """Walk-forward regime-specific RF vs baseline."""
    valid = target.dropna().index
    X = features.reindex(valid).ffill().fillna(0)
    y = target.loc[valid]

    train_size = int(config.train_years * 252)
    test_size = int(config.test_months * 21)
    gap = config.gap_days
    schedule = _compute_fold_schedule(len(X), train_size, test_size, gap, config.n_splits)

    baseline_probs, baseline_actuals = [], []
    routed_probs, routed_actuals = [], []
    high_probs, high_actuals = [], []
    low_probs, low_actuals = [], []

    split_idx = feat_names.index(split_feature)

    for fold_idx, (train_end, test_start, test_end) in enumerate(schedule):
        X_tr = np.nan_to_num(X.iloc[:train_end].values, nan=0, posinf=0, neginf=0)
        y_tr = y.iloc[:train_end].values
        X_te = np.nan_to_num(X.iloc[test_start:test_end].values, nan=0, posinf=0, neginf=0)
        y_te = y.iloc[test_start:test_end].values

        if len(np.unique(y_tr)) < 2:
            continue

        # Regime split on training data
        median_val = np.median(X_tr[:, split_idx])
        high_mask_tr = X_tr[:, split_idx] > median_val
        low_mask_tr = ~high_mask_tr
        high_mask_te = X_te[:, split_idx] > median_val
        low_mask_te = ~high_mask_te

        scaler = RobustScaler()
        Xs_tr = scaler.fit_transform(X_tr)
        Xs_te = scaler.transform(X_te)

        rf_params = dict(
            n_estimators=config.rf_n_estimators, max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf, min_samples_split=config.rf_min_samples_split,
            class_weight="balanced_subsample", random_state=config.random_seed, n_jobs=-1,
        )

        # Baseline
        rf_all = RandomForestClassifier(**rf_params)
        rf_all.fit(Xs_tr, y_tr)
        p_all = rf_all.predict_proba(Xs_te)[:, 1]
        baseline_probs.extend(p_all.tolist())
        baseline_actuals.extend(y_te.tolist())

        # Regime models
        routed = np.zeros(len(y_te))
        for mask_tr, mask_te, probs_list, actuals_list, regime_name in [
            (high_mask_tr, high_mask_te, high_probs, high_actuals, "high"),
            (low_mask_tr, low_mask_te, low_probs, low_actuals, "low"),
        ]:
            if mask_tr.sum() < 50 or len(np.unique(y_tr[mask_tr])) < 2:
                # Fallback to baseline for this regime
                if mask_te.sum() > 0:
                    routed[mask_te] = p_all[mask_te]
                    probs_list.extend(p_all[mask_te].tolist())
                    actuals_list.extend(y_te[mask_te].tolist())
                continue

            rf_regime = RandomForestClassifier(**rf_params)
            rf_regime.fit(Xs_tr[mask_tr], y_tr[mask_tr])
            if mask_te.sum() > 0:
                p_regime = rf_regime.predict_proba(Xs_te[mask_te])[:, 1]
                routed[mask_te] = p_regime
                probs_list.extend(p_regime.tolist())
                actuals_list.extend(y_te[mask_te].tolist())

        routed_probs.extend(routed.tolist())
        routed_actuals.extend(y_te.tolist())

    results = {}
    for name, probs, actuals in [
        ("baseline", baseline_probs, baseline_actuals),
        ("routed", routed_probs, routed_actuals),
        ("high_vol_only", high_probs, high_actuals),
        ("low_vol_only", low_probs, low_actuals),
    ]:
        p, a = np.array(probs), np.array(actuals)
        if len(np.unique(a)) >= 2:
            results[name] = {"auc": float(roc_auc_score(a, p)), "n_events": int(a.sum()), "n_samples": len(a)}
        else:
            results[name] = {"auc": float("nan"), "n_events": int(a.sum()), "n_samples": len(a)}

    return results


def plot_step3(results_up: Dict, results_down: Dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, results, title in [(ax1, results_up, "UP"), (ax2, results_down, "DOWN")]:
        names = list(results.keys())
        aucs = [results[n]["auc"] for n in names]
        colors = ["#3498db", "#e74c3c", "#e67e22", "#2ecc71"]
        bars = ax.bar(range(len(names)), aucs, color=colors[:len(names)], alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, fontsize=8)
        ax.set_ylabel("AUC")
        ax.set_title(f"Regime-Specific Models — {title}", fontsize=12, fontweight="bold")
        ax.set_ylim(0.4, 1.0)
        for bar, auc in zip(bars, aucs):
            if not np.isnan(auc):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{auc:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "regime_specific_auc.png", dpi=150, bbox_inches="tight")
    plt.close()


# =================================================================
# STEP 4: TASK-SPECIFIC FEATURE SUBSETS
# =================================================================
def step4_feature_subsets(
    spectral_features: pd.DataFrame,
    traditional_features: pd.DataFrame,
    targets: Dict[str, pd.Series],
    config: Config,
) -> Dict:
    """Compare AUC with different feature subsets per task."""
    combined = pd.concat([spectral_features, traditional_features], axis=1)
    spectral_cols = list(spectral_features.columns)
    traditional_cols = list(traditional_features.columns)
    eigenvalue_cols = [c for c in combined.columns if feature_group(c) == "Eigenvalue"]

    # Curated subsets based on investigate_v2 SHAP findings
    up_curated = [c for c in ["drawdown_60d", "garch_vol", "drawdown_20d", "downside_vol_20d",
                               "cross_dispersion", "volatility_10d", "price_to_sma_50",
                               "max_loss_5d", "volatility_20d", "return_60d", "price_to_sma_20"]
                  if c in combined.columns]
    down_curated = [c for c in ["volatility_20d", "lambda_2", "drawdown_20d", "drawdown_60d",
                                 "downside_vol_20d", "volatility_10d", "garch_vol", "volatility_5d",
                                 "max_loss_20d", "spectral_gap", "lambda_1", "absorption_ratio_5"]
                    if c in combined.columns]

    results = {}
    for task_name, target in targets.items():
        curated = up_curated if "UP" in task_name else down_curated
        subsets = {
            "All 85": list(combined.columns),
            "Traditional": traditional_cols,
            "Spectral": spectral_cols,
            "Curated": curated,
            "No Eigenvalue": [c for c in combined.columns if c not in eigenvalue_cols],
        }
        task_results = {}
        for subset_name, cols in subsets.items():
            feat_subset = combined[cols]
            print(f"      {task_name} / {subset_name} ({len(cols)} features) ...")
            res = walk_forward_evaluate(feat_subset, target, CGECDNoSelModel, config)
            if "error" not in res:
                task_results[subset_name] = {
                    "auc": float(res["metrics"].auc_roc),
                    "avg_precision": float(res["metrics"].avg_precision),
                    "n_features": len(cols),
                }
            else:
                task_results[subset_name] = {"auc": float("nan"), "avg_precision": float("nan"), "n_features": len(cols)}
        results[task_name] = task_results
    return results


def plot_step4(results: Dict):
    tasks = list(results.keys())
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 6))
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        subsets = list(results[task].keys())
        aucs = [results[task][s]["auc"] for s in subsets]
        colors = ["#3498db", "#f39c12", "#2ecc71", "#e74c3c", "#9b59b6"]
        bars = ax.bar(range(len(subsets)), aucs, color=colors[:len(subsets)], alpha=0.8)
        ax.set_xticks(range(len(subsets)))
        ax.set_xticklabels(subsets, rotation=30, fontsize=8)
        ax.set_ylabel("AUC")
        ax.set_title(f"Feature Subsets — {task}", fontsize=12, fontweight="bold")
        ax.set_ylim(0.4, 1.0)
        for bar, auc in zip(bars, aucs):
            if not np.isnan(auc):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{auc:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "feature_subset_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


# =================================================================
# STEPS 5-7: TRADING STRATEGY
# =================================================================
@dataclass
class SignalRule:
    name: str
    direction: str  # "LONG" or "SHORT"
    feature_conditions: List[Tuple[str, str, float]]  # [(feature, "<" or ">", percentile)]
    min_rf_prob: float = 0.3


def build_signal_rules() -> List[SignalRule]:
    """Formalize top joint threshold findings into signal rules."""
    return [
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


def backtest_strategy(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    target_up: pd.Series,
    target_down: pd.Series,
    config: Config,
    rules: List[SignalRule],
    holding_period: int = 3,
    sizing_method: str = "binary",
) -> Dict:
    """Walk-forward backtest with signal rules."""
    market = prices["SP500"] if "SP500" in prices.columns else prices.iloc[:, 0]
    market_returns = market.pct_change().fillna(0)

    valid = target_up.dropna().index.intersection(target_down.dropna().index)
    X = features.reindex(valid).ffill().fillna(0)
    feat_names = list(X.columns)

    train_size = int(config.train_years * 252)
    test_size = int(config.test_months * 21)
    gap = config.gap_days
    schedule = _compute_fold_schedule(len(X), train_size, test_size, gap, config.n_splits)

    trades = []
    daily_pnl = {}

    for fold_idx, (train_end, test_start, test_end) in enumerate(schedule):
        X_tr = np.nan_to_num(X.iloc[:train_end].values, nan=0, posinf=0, neginf=0)
        y_up_tr = target_up.reindex(valid).iloc[:train_end].values
        y_down_tr = target_down.reindex(valid).iloc[:train_end].values
        X_te = np.nan_to_num(X.iloc[test_start:test_end].values, nan=0, posinf=0, neginf=0)
        test_dates = X.iloc[test_start:test_end].index

        if len(np.unique(y_up_tr)) < 2 or len(np.unique(y_down_tr)) < 2:
            continue

        # Train RF models for UP and DOWN
        scaler = RobustScaler()
        Xs_tr = scaler.fit_transform(X_tr)
        Xs_te = scaler.transform(X_te)

        rf_params = dict(
            n_estimators=config.rf_n_estimators, max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf, min_samples_split=config.rf_min_samples_split,
            class_weight="balanced_subsample", random_state=config.random_seed, n_jobs=-1,
        )

        rf_up = RandomForestClassifier(**rf_params)
        rf_up.fit(Xs_tr, y_up_tr)
        prob_up = rf_up.predict_proba(Xs_te)[:, 1]

        rf_down = RandomForestClassifier(**rf_params)
        rf_down.fit(Xs_tr, y_down_tr)
        prob_down = rf_down.predict_proba(Xs_te)[:, 1]

        # Compute percentile thresholds from training data (no look-ahead)
        percentile_cache = {}
        for feat in feat_names:
            fi = feat_names.index(feat)
            for p in [10, 25, 50, 75, 90]:
                percentile_cache[(feat, p)] = np.percentile(X_tr[:, fi], p)

        # Generate signals
        position_end_idx = -1  # Track when current position expires

        for i in range(len(test_dates)):
            if i <= position_end_idx:
                continue  # Already in a position

            date = test_dates[i]

            for rule in rules:
                # Check feature conditions
                conditions_met = True
                for feat, op, pctile in rule.feature_conditions:
                    if feat not in feat_names:
                        conditions_met = False
                        break
                    fi = feat_names.index(feat)
                    threshold = percentile_cache.get((feat, pctile), 0)
                    val = X_te[i, fi]
                    if op == "<" and val >= threshold:
                        conditions_met = False
                        break
                    elif op == ">" and val <= threshold:
                        conditions_met = False
                        break

                if not conditions_met:
                    continue

                # Check RF probability
                rf_prob = prob_up[i] if rule.direction == "LONG" else prob_down[i]
                if rf_prob < rule.min_rf_prob:
                    continue

                # Execute trade
                end_idx = min(i + holding_period, len(test_dates) - 1)
                if end_idx <= i:
                    continue

                entry_date = date
                exit_date = test_dates[end_idx]

                # Compute PnL from market returns
                trade_returns = market_returns.reindex(test_dates[i + 1 : end_idx + 1]).values
                if len(trade_returns) == 0:
                    continue

                direction_mult = 1.0 if rule.direction == "LONG" else -1.0

                # Position sizing
                if sizing_method == "linear":
                    size = np.clip((rf_prob - rule.min_rf_prob) / 0.5, 0.2, 1.0)
                else:
                    size = 1.0

                trade_pnl = direction_mult * size * np.sum(trade_returns)

                trades.append({
                    "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
                    "exit_date": str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
                    "direction": rule.direction,
                    "rule": rule.name,
                    "rf_prob": float(rf_prob),
                    "size": float(size),
                    "pnl": float(trade_pnl),
                    "holding_days": end_idx - i,
                })

                position_end_idx = end_idx
                break  # Only one trade at a time

    if not trades:
        return {"error": "No trades generated", "trades": [], "metrics": {}}

    trades_df = pd.DataFrame(trades)
    total_pnl = trades_df["pnl"].sum()
    n_trades = len(trades_df)
    wins = (trades_df["pnl"] > 0).sum()
    win_rate = wins / n_trades
    avg_pnl = trades_df["pnl"].mean()
    sharpe = trades_df["pnl"].mean() / (trades_df["pnl"].std() + 1e-10) * np.sqrt(252 / 3)

    # Buy and hold benchmark
    all_test_dates = []
    for _, test_start, test_end in schedule:
        all_test_dates.extend(X.iloc[test_start:test_end].index.tolist())
    bnh_returns = market_returns.reindex(all_test_dates).sum()

    # Max drawdown of trade cumulative PnL
    cum_pnl = trades_df["pnl"].cumsum()
    running_max = cum_pnl.cummax()
    drawdowns = cum_pnl - running_max
    max_dd = drawdowns.min()

    metrics = {
        "total_return": float(total_pnl),
        "n_trades": n_trades,
        "win_rate": float(win_rate),
        "avg_trade_pnl": float(avg_pnl),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "buy_hold_return": float(bnh_returns),
        "long_trades": int((trades_df["direction"] == "LONG").sum()),
        "short_trades": int((trades_df["direction"] == "SHORT").sum()),
        "sizing_method": sizing_method,
    }

    return {"trades": trades_df.to_dict("records"), "metrics": metrics}


def plot_strategy(bt_binary: Dict, bt_linear: Dict):
    """Plot equity curves and comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for col, (bt, label) in enumerate([(bt_binary, "Binary Sizing"), (bt_linear, "Linear Sizing")]):
        if "error" in bt:
            continue
        trades = pd.DataFrame(bt["trades"])
        m = bt["metrics"]

        # Equity curve
        ax = axes[0, col]
        cum_pnl = trades["pnl"].cumsum()
        ax.plot(cum_pnl.values, "b-", linewidth=1.5)
        ax.fill_between(range(len(cum_pnl)), 0, cum_pnl.values, alpha=0.2,
                         where=cum_pnl.values >= 0, color="green")
        ax.fill_between(range(len(cum_pnl)), 0, cum_pnl.values, alpha=0.2,
                         where=cum_pnl.values < 0, color="red")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_title(f"Equity Curve — {label}\nSharpe={m['sharpe_ratio']:.2f}  Return={m['total_return']:.1%}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative PnL")

        # Trade distribution
        ax = axes[1, col]
        long_pnl = trades[trades["direction"] == "LONG"]["pnl"]
        short_pnl = trades[trades["direction"] == "SHORT"]["pnl"]
        if len(long_pnl) > 0:
            ax.hist(long_pnl, bins=20, alpha=0.6, color="green", label=f"Long ({len(long_pnl)})")
        if len(short_pnl) > 0:
            ax.hist(short_pnl, bins=20, alpha=0.6, color="red", label=f"Short ({len(short_pnl)})")
        ax.axvline(0, color="grey", linewidth=0.5)
        ax.set_title(f"Trade PnL Distribution — {label}\nWin Rate={m['win_rate']:.0%}  MaxDD={m['max_drawdown']:.1%}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Trade PnL")
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUT / "equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


# =================================================================
# STEP 8: ECONOMIC NARRATIVE
# =================================================================
def step8_economic_narrative(
    features: pd.DataFrame,
    target_down: pd.Series,
    feat_names: List[str],
) -> Dict:
    """Build quantitative evidence for spectral feature economic interpretation."""
    valid = target_down.dropna().index
    X = features.reindex(valid).ffill().fillna(0)
    y = target_down.loc[valid]

    narrative = {}

    # 1. Correlation between lambda_2 and volatility_20d
    if "lambda_2" in X.columns and "volatility_20d" in X.columns:
        corr = X["lambda_2"].corr(X["volatility_20d"])
        narrative["lambda2_vol20d_correlation"] = float(corr)
        narrative["lambda2_vol20d_interpretation"] = (
            f"Correlation = {corr:.3f}. "
            f"{'Moderate' if abs(corr) < 0.7 else 'High'} correlation — "
            f"lambda_2 captures information {'beyond' if abs(corr) < 0.7 else 'largely redundant with'} volatility."
        )

    # 2. Partial correlation: lambda_2 after controlling for vol
    if "lambda_2" in X.columns and "volatility_20d" in X.columns:
        from sklearn.linear_model import LinearRegression
        # Residualize lambda_2 on volatility_20d
        vol = X["volatility_20d"].values.reshape(-1, 1)
        lam2 = X["lambda_2"].values
        lr = LinearRegression().fit(vol, lam2)
        lambda2_resid = lam2 - lr.predict(vol)

        # Correlation of residualized lambda_2 with DOWN events
        partial_corr = np.corrcoef(lambda2_resid, y.values)[0, 1]
        narrative["lambda2_partial_corr_controlling_vol"] = float(partial_corr)

        # AUC of residualized lambda_2 alone for DOWN
        if len(np.unique(y.values)) >= 2:
            auc_resid = roc_auc_score(y.values, lambda2_resid)
            auc_raw = roc_auc_score(y.values, X["lambda_2"].values)
            auc_vol = roc_auc_score(y.values, X["volatility_20d"].values)
            narrative["lambda2_raw_auc"] = float(auc_raw)
            narrative["lambda2_residualized_auc"] = float(auc_resid)
            narrative["vol20d_auc"] = float(auc_vol)

    # 3. Quintile concentration of lambda_2 during DOWN events
    if "lambda_2" in X.columns:
        event_mask = y.values == 1
        lam2_vals = X["lambda_2"].values
        edges = np.percentile(lam2_vals, [0, 20, 40, 60, 80, 100])
        quintile_pct = []
        for q in range(5):
            lo, hi = edges[q], edges[q + 1]
            if q == 4:
                mask = (lam2_vals[event_mask] >= lo) & (lam2_vals[event_mask] <= hi)
            else:
                mask = (lam2_vals[event_mask] >= lo) & (lam2_vals[event_mask] < hi)
            quintile_pct.append(float(mask.sum() / max(event_mask.sum(), 1) * 100))
        narrative["lambda2_down_quintile_pct"] = quintile_pct
        narrative["lambda2_down_bottom40_pct"] = quintile_pct[0] + quintile_pct[1]

    # 4. Spectral gap interpretation
    if "spectral_gap" in X.columns:
        sg = X["spectral_gap"].values
        event_mean = np.mean(sg[y.values == 1])
        non_event_mean = np.mean(sg[y.values == 0])
        narrative["spectral_gap_event_mean"] = float(event_mean)
        narrative["spectral_gap_non_event_mean"] = float(non_event_mean)
        narrative["spectral_gap_interpretation"] = (
            f"During DOWN events: spectral_gap = {event_mean:.2f} vs normal = {non_event_mean:.2f}. "
            f"{'Higher' if event_mean > non_event_mean else 'Lower'} spectral gap during crashes "
            f"{'suggests dominant factor strengthens' if event_mean > non_event_mean else 'suggests second eigenvalue catching up (diversification breakdown)'}."
        )

    # 5. Economic narrative text
    narrative["economic_story"] = {
        "lambda_2": (
            "Lambda_2 (second eigenvalue of the correlation matrix) measures the strength of the "
            "second principal component in market returns. When lambda_2 is low relative to its "
            "history, the market structure is collapsing into a single factor — all stocks move "
            "together. This is the signature of contagion: diversification breaks down, and "
            "portfolio risk concentrates. Our analysis shows lambda_2 is the #2 predictor for "
            "DOWN moves (SHAP=0.034), works only in high-vol regimes, and 61% of DOWN events "
            "occur when lambda_2 is in the bottom two quintiles."
        ),
        "spectral_gap": (
            "The spectral gap (lambda_1 / lambda_2) measures how dominant the first factor is. "
            "A shrinking spectral gap means the second factor is gaining relative importance — "
            "markets are splitting into distinct risk groups (e.g., cyclicals vs defensives), "
            "which often precedes selloffs."
        ),
        "eigenvalue_entropy": (
            "Eigenvalue entropy measures how evenly distributed risk is across factors. "
            "Low entropy = concentrated risk (one or two factors dominate). "
            "Our regime analysis shows eigenvalue_entropy is a LOW-VOL ONLY predictor for DOWN — "
            "it detects crashes that start from calm, concentrated markets."
        ),
    }

    return narrative


# =================================================================
# STEP 9: EVENT TEMPORAL PROFILES
# =================================================================
def step9_event_profiles(
    features: pd.DataFrame,
    target_down: pd.Series,
    profile_features: List[str] = None,
    window: int = 20,
) -> Dict:
    """Extract temporal profiles around DOWN events for spectral features."""
    if profile_features is None:
        profile_features = [
            "lambda_1", "lambda_2", "lambda_3", "spectral_gap",
            "eigenvalue_entropy", "effective_rank",
            "volatility_20d", "mean_abs_corr", "absorption_ratio_5",
        ]

    valid = target_down.dropna().index
    X = features.reindex(valid).ffill().fillna(0)
    y = target_down.loc[valid]

    event_dates = y[y == 1].index
    all_dates = list(X.index)
    profiles = {}

    for feat in profile_features:
        if feat not in X.columns:
            continue

        vals = X[feat].values
        event_windows = []

        for edate in event_dates:
            if edate not in all_dates:
                continue
            eidx = all_dates.index(edate)

            # Need at least window days before and after, plus baseline window
            if eidx < window + 20 or eidx + window >= len(all_dates):
                continue

            # Extract window
            win = vals[eidx - window: eidx + window + 1]

            # Normalize using pre-event baseline (days -40 to -21)
            baseline_start = max(0, eidx - 40)
            baseline_end = eidx - 21
            if baseline_end > baseline_start:
                baseline = vals[baseline_start:baseline_end]
                mu = np.mean(baseline)
                sigma = np.std(baseline)
                if sigma > 1e-10:
                    win = (win - mu) / sigma
                else:
                    win = win - mu

            event_windows.append(win)

        if not event_windows:
            continue

        windows = np.array(event_windows)
        offsets = np.arange(-window, window + 1)
        mean_profile = np.mean(windows, axis=0)
        std_profile = np.std(windows, axis=0)
        median_profile = np.median(windows, axis=0)

        # Warning time: first day where mean exceeds +1 z-score (counting from day -20)
        warning_day = None
        for d in range(len(offsets)):
            if offsets[d] < 0 and abs(mean_profile[d]) > 1.0:
                warning_day = int(offsets[d])
                break

        profiles[feat] = {
            "offsets": offsets.tolist(),
            "mean": mean_profile.tolist(),
            "std": std_profile.tolist(),
            "median": median_profile.tolist(),
            "n_events": len(event_windows),
            "warning_day": warning_day,
        }

    return profiles


def plot_step9(profiles: Dict):
    feats = list(profiles.keys())
    n = len(feats)
    if n == 0:
        return
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    fig.suptitle("Anatomy of a Crash — Spectral Feature Evolution Around DOWN Events",
                 fontsize=14, fontweight="bold")

    for i, feat in enumerate(feats):
        r, c = i // ncols, i % ncols
        ax = axes[r][c] if nrows > 1 else axes[c]
        p = profiles[feat]
        offsets = np.array(p["offsets"])
        mean = np.array(p["mean"])
        std = np.array(p["std"])

        ax.fill_between(offsets, mean - std, mean + std, alpha=0.2, color="#3498db")
        ax.plot(offsets, mean, "b-", linewidth=2, label="Mean z-score")
        ax.plot(offsets, np.array(p["median"]), "g--", linewidth=1, alpha=0.7, label="Median")
        ax.axvline(0, color="red", linewidth=2, linestyle="--", alpha=0.8, label="Event day")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axhline(1, color="orange", linewidth=0.5, linestyle=":", alpha=0.5)
        ax.axhline(-1, color="orange", linewidth=0.5, linestyle=":", alpha=0.5)

        warning = p.get("warning_day")
        title = f"{feat}  [{feature_group(feat)}]\n({p['n_events']} events)"
        if warning is not None:
            title += f"  Warning: day {warning}"
            ax.axvline(warning, color="orange", linewidth=1.5, linestyle=":", alpha=0.8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Days from event")
        ax.set_ylabel("Z-score")
        ax.legend(fontsize=7)

    for i in range(n, nrows * ncols):
        r, c = i // ncols, i % ncols
        (axes[r][c] if nrows > 1 else axes[c]).set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "event_profiles_DOWN.png", dpi=150, bbox_inches="tight")
    plt.close()


# =================================================================
# STEP 10: SPECTRAL ABLATION
# =================================================================
def step10_ablation(
    spectral_features: pd.DataFrame,
    traditional_features: pd.DataFrame,
    targets: Dict[str, pd.Series],
    config: Config,
) -> Dict:
    """Ablation: measure marginal contribution of eigenvalue features."""
    combined = pd.concat([spectral_features, traditional_features], axis=1)
    eigenvalue_cols = [c for c in combined.columns if feature_group(c) == "Eigenvalue"]
    spectral_cols = list(spectral_features.columns)
    traditional_cols = list(traditional_features.columns)

    configs = {
        "Full (85)": list(combined.columns),
        "No Eigenvalue": [c for c in combined.columns if c not in eigenvalue_cols],
        "No Spectral": traditional_cols,
        "No Traditional": spectral_cols,
        "Eigenvalue Only": eigenvalue_cols,
    }

    results = {}
    for task_name, target in targets.items():
        task_results = {}
        for cfg_name, cols in configs.items():
            if len(cols) < 3:
                task_results[cfg_name] = {"auc": float("nan"), "n_features": len(cols)}
                continue
            feat_sub = combined[cols]
            print(f"      {task_name} / {cfg_name} ({len(cols)} features) ...")
            res = walk_forward_evaluate(feat_sub, target, CGECDNoSelModel, config)
            if "error" not in res:
                task_results[cfg_name] = {
                    "auc": float(res["metrics"].auc_roc),
                    "avg_precision": float(res["metrics"].avg_precision),
                    "n_features": len(cols),
                }
            else:
                task_results[cfg_name] = {"auc": float("nan"), "n_features": len(cols)}
        # Marginal contribution
        full_auc = task_results.get("Full (85)", {}).get("auc", float("nan"))
        no_eig_auc = task_results.get("No Eigenvalue", {}).get("auc", float("nan"))
        task_results["eigenvalue_marginal_auc"] = full_auc - no_eig_auc if not (np.isnan(full_auc) or np.isnan(no_eig_auc)) else float("nan")
        results[task_name] = task_results

    return results


def plot_step10(results: Dict):
    tasks = list(results.keys())
    cfg_names = ["Full (85)", "No Eigenvalue", "No Spectral", "No Traditional", "Eigenvalue Only"]
    colors = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71", "#9b59b6"]

    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 6))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        aucs = [results[task].get(c, {}).get("auc", float("nan")) for c in cfg_names]
        bars = ax.bar(range(len(cfg_names)), aucs, color=colors, alpha=0.8)
        ax.set_xticks(range(len(cfg_names)))
        ax.set_xticklabels(cfg_names, rotation=30, fontsize=8)
        ax.set_ylabel("AUC")
        marginal = results[task].get("eigenvalue_marginal_auc", float("nan"))
        title = f"Spectral Ablation — {task}"
        if not np.isnan(marginal):
            title += f"\nEigenvalue marginal: {marginal:+.4f} AUC"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0.4, 1.0)
        for bar, auc in zip(bars, aucs):
            if not np.isnan(auc):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{auc:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "spectral_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()


# =================================================================
# CONSOLE REPORTS
# =================================================================
def print_thresholds(thresh_df, task, base_rate):
    print(f"\n  {'='*115}")
    print(f"  VALUE THRESHOLDS — {task}  (base rate: {base_rate:.1%})")
    print(f"  {'='*115}")
    print(f"  {'Feature':<32s} {'Threshold':>10s} {'Dir':>4s} {'Pctile':>7s} "
          f"{'Precision':>10s} {'Recall':>8s} {'Events':>10s} {'Lift':>6s}")
    print(f"  {'-'*115}")
    for _, r in thresh_df.iterrows():
        lift = r["precision"] / max(base_rate, 1e-10)
        print(f"  {r['feature']:<32s} {r['threshold']:>10.4f} {r['direction']:>4s} "
              f"{r['threshold_percentile']:>6.1f}% "
              f"{r['precision']:>9.1%} {r['recall']:>7.1%} "
              f"{r['events_captured']:>3.0f}/{r['total_events']:<3.0f}    "
              f"{lift:>5.1f}x")


def print_joint_thresholds(jt, task, base_rate, top_k=15):
    if jt.empty:
        print(f"\n  No joint threshold rules found for {task}")
        return
    print(f"\n  {'='*130}")
    print(f"  JOINT THRESHOLDS — {task}  (base rate: {base_rate:.1%})")
    print(f"  {'='*130}")
    for i, (_, r) in enumerate(jt.head(top_k).iterrows(), 1):
        rule_a = f"{r['feature_A']} {r['dir_A']} P{r['pctile_A']:.0f}"
        rule_b = f"{r['feature_B']} {r['dir_B']} P{r['pctile_B']:.0f}"
        print(f"  #{i:<3d} {rule_a:<38s} AND {rule_b:<38s} "
              f"prec={r['precision']:.1%}  recall={r['recall']:.1%}  "
              f"lift={r['lift']:.1f}x  ({r['n_hits']}/{r['n_signals']} signals)")


def print_interaction_rules(rules_df, task, top_k=15):
    if rules_df.empty:
        print(f"\n  No interaction rules found for {task}")
        return
    print(f"\n  {'='*130}")
    print(f"  INTERACTION RULES (depth 2-3 trees) — {task}")
    print(f"  {'='*130}")
    for i, (_, r) in enumerate(rules_df.head(top_k).iterrows(), 1):
        print(f"  #{i:<3d} {r['rule']:<70s} "
              f"prec={r['precision']:.1%}  recall={r['recall']:.1%}  "
              f"lift={r['lift']:.1f}x  ({r['n_hits']}/{r['n_signals']} sig)  depth={r['depth']}")


def print_regime_results(results, task):
    print(f"\n  {'='*80}")
    print(f"  REGIME-SPECIFIC MODELS — {task}")
    print(f"  {'='*80}")
    print(f"  {'Config':<20s} {'AUC':>8s} {'Events':>8s} {'Samples':>8s}")
    print(f"  {'-'*60}")
    for name, vals in results.items():
        auc_str = f"{vals['auc']:.3f}" if not np.isnan(vals['auc']) else "N/A"
        print(f"  {name:<20s} {auc_str:>8s} {vals['n_events']:>8d} {vals['n_samples']:>8d}")


def print_feature_subsets(results):
    print(f"\n  {'='*100}")
    print(f"  TASK-SPECIFIC FEATURE SUBSETS")
    print(f"  {'='*100}")
    for task, subsets in results.items():
        print(f"\n  {task}:")
        print(f"    {'Subset':<20s} {'AUC':>8s} {'AvgPrec':>10s} {'N feat':>8s}")
        print(f"    {'-'*60}")
        for name, vals in subsets.items():
            auc_str = f"{vals['auc']:.3f}" if not np.isnan(vals['auc']) else "N/A"
            ap_str = f"{vals['avg_precision']:.3f}" if not np.isnan(vals.get('avg_precision', float('nan'))) else "N/A"
            print(f"    {name:<20s} {auc_str:>8s} {ap_str:>10s} {vals['n_features']:>8d}")


def print_backtest(metrics, label):
    if not metrics:
        print(f"\n  Backtest ({label}): No trades")
        return
    print(f"\n  {'='*80}")
    print(f"  BACKTEST — {label}")
    print(f"  {'='*80}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:<25s}: {v:>10.4f}")
        else:
            print(f"    {k:<25s}: {v}")


def print_ablation(results):
    print(f"\n  {'='*100}")
    print(f"  SPECTRAL ABLATION STUDY")
    print(f"  {'='*100}")
    for task, configs in results.items():
        print(f"\n  {task}:")
        print(f"    {'Config':<20s} {'AUC':>8s} {'N feat':>8s}")
        print(f"    {'-'*50}")
        for name, vals in configs.items():
            if name == "eigenvalue_marginal_auc":
                continue
            auc_str = f"{vals['auc']:.3f}" if not np.isnan(vals.get('auc', float('nan'))) else "N/A"
            print(f"    {name:<20s} {auc_str:>8s} {vals.get('n_features', '-'):>8}")
        marginal = configs.get("eigenvalue_marginal_auc", float("nan"))
        if not np.isnan(marginal):
            print(f"    → Eigenvalue marginal contribution: {marginal:+.4f} AUC")


def print_narrative(narrative):
    print(f"\n  {'='*100}")
    print(f"  ECONOMIC NARRATIVE — Why Spectral Features Contribute")
    print(f"  {'='*100}")
    for key, val in narrative.items():
        if key == "economic_story":
            for feat, text in val.items():
                print(f"\n  {feat}:")
                # Wrap text
                words = text.split()
                line = "    "
                for w in words:
                    if len(line) + len(w) > 100:
                        print(line)
                        line = "    " + w
                    else:
                        line += " " + w
                print(line)
        elif isinstance(val, str):
            print(f"  {key}: {val}")
        elif isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        elif isinstance(val, list):
            print(f"  {key}: {[f'{v:.1f}%' for v in val]}")


def print_event_profiles(profiles):
    print(f"\n  {'='*80}")
    print(f"  EVENT TEMPORAL PROFILES — DOWN Events")
    print(f"  {'='*80}")
    print(f"  {'Feature':<30s} {'Group':<14s} {'N events':>10s} {'Warning day':>12s} {'Peak z-score':>12s}")
    print(f"  {'-'*80}")
    for feat, p in profiles.items():
        mean = np.array(p["mean"])
        peak_z = mean[np.argmax(np.abs(mean))]
        warning = p.get("warning_day")
        w_str = f"day {warning}" if warning is not None else "none"
        print(f"  {feat:<30s} {feature_group(feat):<14s} {p['n_events']:>10d} {w_str:>12s} {peak_z:>+12.2f}")


# =================================================================
# MAIN
# =================================================================
def main():
    t0 = datetime.now()
    print("=" * 80)
    print("INVESTIGATE v3 — Feature Understanding, Strategy, Spectral Explanation")
    print("=" * 80)
    print(f"Start: {t0:%Y-%m-%d %H:%M:%S}")
    print(f"Output: {OUT}/\n")

    config = Config()

    # ── 1. Data ──────────────────────────────────────────────────
    print("[1/3] Loading data & features ...")
    prices, returns = load_data(config)
    spectral = build_spectral_features(returns, config)
    traditional = build_traditional_features(prices, returns)
    combined = pd.concat([spectral, traditional], axis=1)
    feat_names = list(combined.columns)
    print(f"  {len(feat_names)} features")

    market = prices["SP500"] if "SP500" in prices.columns else prices.iloc[:, 0]
    fut_3d = market.pct_change(3).shift(-3)
    targets = {
        "Large UP move 3d (>3%)": (fut_3d > 0.03).astype(int),
        "Large DOWN move 3d (>3%)": (fut_3d < -0.03).astype(int),
    }
    for name, tgt in targets.items():
        print(f"  {name}: {tgt.dropna().mean():.1%} ({int(tgt.dropna().sum())} events)")

    # Pre-compute SHAP rankings from investigate_v2 (just use RF importance as proxy)
    # We'll do a quick unbiased importance run for ranking
    print("\n  Computing feature rankings ...")
    shap_rankings = {}
    for task_name, target in targets.items():
        valid = target.dropna().index
        X = combined.reindex(valid).ffill().fillna(0)
        y = target.loc[valid]
        train_size = int(config.train_years * 252)
        test_size = int(config.test_months * 21)
        gap = config.gap_days
        schedule = _compute_fold_schedule(len(X), train_size, test_size, gap, config.n_splits)
        last_fold = schedule[-2]  # Use second-to-last fold (last may have 0 events)
        train_end, test_start, test_end = last_fold
        X_tr = np.nan_to_num(X.iloc[:train_end].values, nan=0, posinf=0, neginf=0)
        y_tr = y.iloc[:train_end].values
        X_te = np.nan_to_num(X.iloc[test_start:test_end].values, nan=0, posinf=0, neginf=0)
        y_te = y.iloc[test_start:test_end].values
        res = unbiased_importance(X_tr, y_tr, X_te, y_te, feat_names, config)
        shap_rankings[task_name] = np.mean(np.abs(res["shap_values"]), axis=0) if res["shap_values"] is not None else res["rf_imp"]
        print(f"    {task_name}: ranking computed")

    summary = {}

    # ══════════════════════════════════════════════════════════════
    # TASK 1: IMPROVE FEATURE UNDERSTANDING
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("TASK 1: IMPROVE FEATURE UNDERSTANDING")
    print(f"{'='*80}")

    # Step 1: Extended thresholds
    print("\n  [Step 1] Extended fold thresholds (7 folds) ...")
    step1_results = {}
    for task_name, target in targets.items():
        base_rate = float(target.dropna().mean())
        print(f"\n    {task_name}:")
        thresh_df, jt_df = step1_extended_thresholds(
            combined, target, feat_names, config, shap_rankings[task_name], n_folds=7
        )
        print_thresholds(thresh_df, task_name, base_rate)
        print_joint_thresholds(jt_df, task_name, base_rate)
        plot_step1(thresh_df, jt_df, task_name)
        step1_results[task_name] = {
            "thresholds": thresh_df.head(10).to_dict("records"),
            "joint_thresholds": jt_df.head(5).to_dict("records") if not jt_df.empty else [],
        }

    # Step 2: Interaction thresholds
    print("\n  [Step 2] Interaction-aware thresholds ...")
    step2_results = {}
    for task_name, target in targets.items():
        valid = target.dropna().index
        X = combined.reindex(valid).ffill().fillna(0)
        y = target.loc[valid]
        schedule = _compute_fold_schedule(
            len(X), int(config.train_years * 252), int(config.test_months * 21),
            config.gap_days, config.n_splits
        )
        # Pool last 7 folds
        folds = schedule[-7:]
        pooled_X_te, pooled_y_te = [], []
        for te, ts, tend in folds:
            pooled_X_te.append(X.iloc[ts:tend].values)
            pooled_y_te.append(y.iloc[ts:tend].values)
        last_te = folds[-1][0]
        X_tr = np.nan_to_num(X.iloc[:last_te].values, nan=0, posinf=0, neginf=0)
        y_tr = y.iloc[:last_te].values
        X_te = np.nan_to_num(np.vstack(pooled_X_te), nan=0, posinf=0, neginf=0)
        y_te = np.concatenate(pooled_y_te)

        target_feats = ["lambda_2", "volatility_20d"] if "DOWN" in task_name else ["drawdown_60d", "garch_vol"]
        rules_df = step2_interaction_thresholds(X_tr, y_tr, X_te, y_te, feat_names, target_feats)
        print_interaction_rules(rules_df, task_name)
        plot_step2(rules_df, task_name)
        step2_results[task_name] = rules_df.head(10).to_dict("records") if not rules_df.empty else []

    # Step 3: Regime-specific models
    print("\n  [Step 3] Regime-specific RF models ...")
    step3_results = {}
    for task_name, target in targets.items():
        print(f"\n    {task_name}:")
        regime_res = step3_regime_models(combined, target, feat_names, config)
        print_regime_results(regime_res, task_name)
        step3_results[task_name] = regime_res
    plot_step3(
        step3_results.get("Large UP move 3d (>3%)", {}),
        step3_results.get("Large DOWN move 3d (>3%)", {}),
    )

    # Step 4: Feature subsets
    print("\n  [Step 4] Task-specific feature subsets ...")
    step4_results = step4_feature_subsets(spectral, traditional, targets, config)
    print_feature_subsets(step4_results)
    plot_step4(step4_results)

    # ══════════════════════════════════════════════════════════════
    # TASK 2: TRADING STRATEGY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("TASK 2: PRACTICAL TRADING STRATEGY")
    print(f"{'='*80}")

    # Step 5: Signal rules
    rules = build_signal_rules()
    print("\n  [Step 5] Signal Rules:")
    for r in rules:
        conds = " AND ".join(f"{f} {op} P{p}" for f, op, p in r.feature_conditions)
        print(f"    {r.name}: {conds}  (RF prob > {r.min_rf_prob})")

    # Step 6 & 7: Backtest
    print("\n  [Step 6-7] Walk-forward backtest ...")
    bt_binary = backtest_strategy(
        combined, prices, targets["Large UP move 3d (>3%)"], targets["Large DOWN move 3d (>3%)"],
        config, rules, holding_period=3, sizing_method="binary",
    )
    bt_linear = backtest_strategy(
        combined, prices, targets["Large UP move 3d (>3%)"], targets["Large DOWN move 3d (>3%)"],
        config, rules, holding_period=3, sizing_method="linear",
    )
    print_backtest(bt_binary.get("metrics", {}), "Binary Sizing")
    print_backtest(bt_linear.get("metrics", {}), "Linear Sizing")
    plot_strategy(bt_binary, bt_linear)

    # ══════════════════════════════════════════════════════════════
    # TASK 3: SPECTRAL EXPLANATION
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("TASK 3: WHY SPECTRAL FEATURES CONTRIBUTE")
    print(f"{'='*80}")

    # Step 8: Economic narrative
    print("\n  [Step 8] Economic narrative ...")
    narrative = step8_economic_narrative(combined, targets["Large DOWN move 3d (>3%)"], feat_names)
    print_narrative(narrative)

    # Step 9: Event profiles
    print("\n  [Step 9] Event temporal profiles ...")
    profiles = step9_event_profiles(combined, targets["Large DOWN move 3d (>3%)"])
    print_event_profiles(profiles)
    plot_step9(profiles)

    # Step 10: Ablation
    print("\n  [Step 10] Spectral ablation study ...")
    ablation = step10_ablation(spectral, traditional, targets, config)
    print_ablation(ablation)
    plot_step10(ablation)

    # ══════════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("Saving results ...")

    summary = {
        "step1_extended_thresholds": step1_results,
        "step2_interaction_rules": step2_results,
        "step3_regime_models": step3_results,
        "step4_feature_subsets": step4_results,
        "step5_signal_rules": [{"name": r.name, "direction": r.direction,
                                 "conditions": r.feature_conditions, "min_prob": r.min_rf_prob}
                                for r in rules],
        "step6_backtest_binary": bt_binary.get("metrics", {}),
        "step7_backtest_linear": bt_linear.get("metrics", {}),
        "step8_narrative": narrative,
        "step9_event_profiles": {k: {"n_events": v["n_events"], "warning_day": v["warning_day"]}
                                  for k, v in profiles.items()},
        "step10_ablation": ablation,
    }

    with open(OUT / "summary_v3.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save signal rules as text
    with open(OUT / "signal_rules.txt", "w") as f:
        for r in rules:
            conds = " AND ".join(f"{feat} {op} P{p}" for feat, op, p in r.feature_conditions)
            f.write(f"{r.name}\n  Direction: {r.direction}\n  Conditions: {conds}\n  Min RF prob: {r.min_rf_prob}\n\n")

    # Save narrative
    with open(OUT / "economic_narrative.json", "w") as f:
        json.dump(narrative, f, indent=2, default=str)

    print(f"\n  All outputs saved to {OUT}/")
    print(f"  Runtime: {datetime.now() - t0}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
