#!/usr/bin/env python3
"""
Generate publication-quality figures for the CGECD paper.
==========================================================
Outputs 300 DPI PNG + PDF to results/paper_figures/

Figures:
  1. SHAP beeswarm plots (UP + DOWN)
  2. SHAP feature group bar plot
  3. ROC curves (UP + DOWN)
  4. Precision-Recall curves (UP + DOWN)
  5. Model comparison bar chart
  6. Ablation heatmap
  7. Event temporal profiles ("anatomy of a crash")
  8. Equity curve from backtest
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

from config import Config
from metrics import compute_metrics
from algorithm import (
    load_data,
    build_spectral_features,
    build_traditional_features,
    _compute_fold_schedule,
    CGECDModel,
    CGECDNoSelModel,
    walk_forward_evaluate,
)
from benchmarks import (
    prepare_benchmark_features,
    AbsorptionRatioModel,
    TurbulenceModel,
    GARCHModel,
    HARRVModel,
    RandomForestBaselineModel,
    SVMBaselineModel,
    GradientBoostingModel,
    LogisticRegressionModel,
)

OUT = Path("results/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

GROUP_PALETTE = {
    "Eigenvalue": "#2ecc71",
    "Correlation": "#3498db",
    "Topology": "#9b59b6",
    "Dynamics": "#e74c3c",
    "Returns": "#f39c12",
    "Volatility": "#1abc9c",
    "Momentum": "#34495e",
    "Drawdown": "#e67e22",
    "TailRisk": "#c0392b",
    "CrossAsset": "#2980b9",
    "Other": "#95a5a6",
}


def feature_group(name: str) -> str:
    n = name.lower()
    if "_zscore_" in n or "_roc_" in n or "_accel_" in n:
        return "Dynamics"
    if "_ratio" in n and any(k in n for k in ("lambda", "absorption", "effective_rank", "edge_density", "mean_abs_corr")):
        return "Eigenvalue"
    if any(k in n for k in ("lambda", "spectral_gap", "absorption_ratio", "eigenvalue_entropy", "effective_rank", "mp_excess")):
        return "Eigenvalue"
    if "edge_density" in n:
        return "Topology"
    if "corr" in n:
        return "Correlation"
    if "drawdown" in n:
        return "Drawdown"
    if any(k in n for k in ("volatility", "vol_ratio", "vol_change", "vol_of_vol", "garch")):
        return "Volatility"
    if "return" in n:
        return "Returns"
    if any(k in n for k in ("sma", "rsi", "momentum")):
        return "Momentum"
    if any(k in n for k in ("skew", "kurtosis", "max_loss", "downside_vol", "down_up_vol", "neg_days")):
        return "TailRisk"
    if any(k in n for k in ("credit", "flight", "breadth", "dispersion", "em_stress")):
        return "CrossAsset"
    return "Other"


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png + .pdf")


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace(">", "gt").replace("%", "pct").replace("<", "lt")


# =================================================================
# FIGURE 1 & 2: SHAP ANALYSIS
# =================================================================
def compute_shap_values(features, target, config, n_folds=5):
    """Compute SHAP values using last n_folds of walk-forward."""
    valid = target.dropna().index
    X = features.reindex(valid).ffill().fillna(0)
    y = target.loc[valid]
    feat_names = list(X.columns)

    train_size = int(config.train_years * 252)
    test_size = int(config.test_months * 21)
    gap = config.gap_days
    schedule = _compute_fold_schedule(len(X), train_size, test_size, gap, config.n_splits)

    folds_to_use = schedule[-n_folds:]
    all_shap = []
    all_X_test = []

    for fold_idx, (train_end, test_start, test_end) in enumerate(folds_to_use):
        print(f"    SHAP fold {fold_idx+1}/{len(folds_to_use)} ...")
        X_tr = np.nan_to_num(X.iloc[:train_end].values, nan=0, posinf=0, neginf=0)
        y_tr = y.iloc[:train_end].values
        X_te = np.nan_to_num(X.iloc[test_start:test_end].values, nan=0, posinf=0, neginf=0)

        if len(np.unique(y_tr)) < 2:
            continue

        scaler = RobustScaler()
        Xs_tr = scaler.fit_transform(X_tr)
        Xs_te = scaler.transform(X_te)

        rf = RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            min_samples_split=config.rf_min_samples_split,
            class_weight="balanced_subsample",
            random_state=config.random_seed,
            n_jobs=-1,
        )
        rf.fit(Xs_tr, y_tr)

        # Use TreeExplainer for speed
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(Xs_te)

        # For binary classification, take class 1 SHAP values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        all_shap.append(shap_vals)
        all_X_test.append(Xs_te)

    if not all_shap:
        return None, None, feat_names

    shap_combined = np.vstack(all_shap)
    X_combined = np.vstack(all_X_test)
    return shap_combined, X_combined, feat_names


def plot_shap_beeswarm(shap_values, X_test, feat_names, task_name, top_k=20):
    """Publication-quality SHAP beeswarm plot."""
    # Ensure 2D (samples x features)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # Take class 1

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.ravel()

    top_idx = np.argsort(mean_abs_shap)[::-1][:top_k].ravel()

    shap_top = shap_values[:, top_idx]
    X_top = X_test[:, top_idx]
    names_top = [feat_names[int(i)] for i in top_idx]
    groups_top = [feature_group(n) for n in names_top]

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(len(names_top)):
        y_pos = len(names_top) - 1 - i
        sv = shap_top[:, i]
        fv = X_top[:, i]

        # Normalize feature values to [0, 1] for coloring
        fv_min, fv_max = fv.min(), fv.max()
        if fv_max - fv_min > 1e-10:
            fv_norm = (fv - fv_min) / (fv_max - fv_min)
        else:
            fv_norm = np.full_like(fv, 0.5)

        # Jitter y
        jitter = np.random.RandomState(42).normal(0, 0.15, len(sv))
        ax.scatter(sv, y_pos + jitter, c=fv_norm, cmap="coolwarm",
                   s=3, alpha=0.5, rasterized=True)

    ax.set_yticks(range(len(names_top)))
    ax.set_yticklabels(list(reversed(names_top)), fontsize=9)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("SHAP value (impact on model output)")

    short_task = "UP" if "UP" in task_name else "DOWN"
    ax.set_title(f"SHAP Feature Importance — {short_task} Prediction", fontweight="bold")

    # Color bar
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, aspect=40, pad=0.02)
    cbar.set_label("Feature value (normalized)", fontsize=9)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    _save(fig, f"shap_beeswarm_{short_task}")


def plot_shap_group_bar(shap_values_dict, feat_names_dict):
    """Bar chart of mean |SHAP| by feature group for both tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (task_name, shap_vals) in zip(axes, shap_values_dict.items()):
        feat_names = feat_names_dict[task_name]
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.ravel()

        group_imp = {}
        for i, fn in enumerate(feat_names):
            g = feature_group(fn)
            group_imp[g] = group_imp.get(g, 0) + mean_abs[i]

        sorted_groups = sorted(group_imp.items(), key=lambda x: x[1], reverse=True)
        names = [g for g, _ in sorted_groups]
        vals = [v for _, v in sorted_groups]
        colors = [GROUP_PALETTE.get(g, "#95a5a6") for g in names]

        ax.barh(range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP| value")
        short = "UP" if "UP" in task_name else "DOWN"
        ax.set_title(f"Feature Group Importance — {short}", fontweight="bold")

    plt.tight_layout()
    _save(fig, "shap_group_importance")


# =================================================================
# FIGURE 3 & 4: ROC + PR CURVES
# =================================================================
def plot_roc_publication(all_results):
    """Publication ROC curves for both tasks."""
    tasks = list(all_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, task_name in zip(axes, tasks):
        raw = all_results[task_name]["raw"]
        for model_name, res in raw.items():
            if "actuals" not in res:
                continue
            fpr, tpr, _ = roc_curve(res["actuals"], res["probabilities"])
            auc = res["metrics"].auc_roc
            lw = 2.5 if res.get("is_ours") else 1.2
            ls = "-" if res.get("is_ours") else "--"
            color = "#e74c3c" if res.get("is_ours") else None
            ax.plot(fpr, tpr, label=f"{model_name} ({auc:.3f})",
                    linewidth=lw, linestyle=ls, color=color)

        ax.plot([0, 1], [0, 1], "k:", alpha=0.4, linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        short = "UP" if "UP" in task_name else "DOWN"
        ax.set_title(f"ROC Curves — {short} Prediction", fontweight="bold")
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    _save(fig, "roc_curves")


def plot_pr_publication(all_results):
    """Publication PR curves for both tasks."""
    tasks = list(all_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, task_name in zip(axes, tasks):
        raw = all_results[task_name]["raw"]
        pos_rate = all_results[task_name]["pos_rate"]

        for model_name, res in raw.items():
            if "actuals" not in res:
                continue
            prec, rec, _ = precision_recall_curve(res["actuals"], res["probabilities"])
            ap = res["metrics"].avg_precision
            lw = 2.5 if res.get("is_ours") else 1.2
            ls = "-" if res.get("is_ours") else "--"
            color = "#e74c3c" if res.get("is_ours") else None
            ax.plot(rec, prec, label=f"{model_name} (AP={ap:.3f})",
                    linewidth=lw, linestyle=ls, color=color)

        ax.axhline(pos_rate, color="grey", linewidth=0.8, linestyle=":", alpha=0.5, label=f"Baseline ({pos_rate:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        short = "UP" if "UP" in task_name else "DOWN"
        ax.set_title(f"Precision–Recall — {short} Prediction", fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    _save(fig, "pr_curves")


# =================================================================
# FIGURE 5: MODEL COMPARISON
# =================================================================
def plot_model_comparison_pub(all_results):
    """Grouped horizontal bar chart comparing all models."""
    tasks = list(all_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, task_name in zip(axes, tasks):
        results = all_results[task_name]["results"]
        sr = sorted(results, key=lambda x: x["auc_roc"])
        names = [r["model"] for r in sr]
        aucs = [r["auc_roc"] for r in sr]

        colors = []
        for r in sr:
            if r.get("is_ours"):
                colors.append("#e74c3c")
            elif r["model"] == "Random Baseline":
                colors.append("#bdc3c7")
            else:
                colors.append("#3498db")

        bars = ax.barh(range(len(names)), aucs, color=colors, edgecolor="white", height=0.6)
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{auc:.3f}", va="center", fontsize=9, fontweight="bold")

        ax.axvline(0.5, color="red", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("AUC-ROC")
        ax.set_xlim(0.4, 1.0)
        short = "UP" if "UP" in task_name else "DOWN"
        ax.set_title(f"Model Comparison — {short}", fontweight="bold")

    plt.tight_layout()
    _save(fig, "model_comparison")


# =================================================================
# FIGURE 6: ABLATION HEATMAP
# =================================================================
def run_ablation(spectral_features, traditional_features, targets, config):
    """Run ablation and return results."""
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
        task_res = {}
        for cfg_name, cols in configs.items():
            if len(cols) < 3:
                task_res[cfg_name] = {"auc_roc": float("nan"), "avg_precision": float("nan")}
                continue
            print(f"    Ablation: {task_name} / {cfg_name} ({len(cols)} feat) ...")
            res = walk_forward_evaluate(combined[cols], target, CGECDNoSelModel, config)
            if "error" not in res:
                task_res[cfg_name] = {
                    "auc_roc": float(res["metrics"].auc_roc),
                    "avg_precision": float(res["metrics"].avg_precision),
                }
            else:
                task_res[cfg_name] = {"auc_roc": float("nan"), "avg_precision": float("nan")}
        results[task_name] = task_res
    return results


def plot_ablation_heatmap(ablation_results):
    """Heatmap of AUC across feature configurations and tasks."""
    configs = ["Full (85)", "No Eigenvalue", "No Spectral", "No Traditional", "Eigenvalue Only"]
    tasks = list(ablation_results.keys())
    short_tasks = ["UP" if "UP" in t else "DOWN" for t in tasks]

    # Build matrix
    auc_matrix = np.zeros((len(tasks), len(configs)))
    ap_matrix = np.zeros((len(tasks), len(configs)))
    for i, task in enumerate(tasks):
        for j, cfg in enumerate(configs):
            auc_matrix[i, j] = ablation_results[task].get(cfg, {}).get("auc_roc", float("nan"))
            ap_matrix[i, j] = ablation_results[task].get(cfg, {}).get("avg_precision", float("nan"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    for ax, matrix, metric_name in [(ax1, auc_matrix, "AUC-ROC"), (ax2, ap_matrix, "Avg Precision")]:
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.45, vmax=0.9)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(short_tasks, fontsize=10)
        ax.set_title(f"Feature Ablation — {metric_name}", fontweight="bold")

        for i in range(len(tasks)):
            for j in range(len(configs)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if val < 0.6 else "black")

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    _save(fig, "ablation_heatmap")


# =================================================================
# FIGURE 7: EVENT TEMPORAL PROFILES
# =================================================================
def compute_event_profiles(features, target_down, window=20):
    """Compute z-score profiles around DOWN events."""
    profile_features = [
        "lambda_1", "lambda_2", "spectral_gap",
        "eigenvalue_entropy", "volatility_20d", "absorption_ratio_5",
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
            if eidx < window + 20 or eidx + window >= len(all_dates):
                continue

            win = vals[eidx - window: eidx + window + 1]
            baseline_start = max(0, eidx - 40)
            baseline_end = eidx - 21
            if baseline_end > baseline_start:
                baseline = vals[baseline_start:baseline_end]
                mu, sigma = np.mean(baseline), np.std(baseline)
                if sigma > 1e-10:
                    win = (win - mu) / sigma
                else:
                    win = win - mu
            event_windows.append(win)

        if not event_windows:
            continue

        windows_arr = np.array(event_windows)
        offsets = np.arange(-window, window + 1)
        mean_profile = np.mean(windows_arr, axis=0)
        std_profile = np.std(windows_arr, axis=0)

        warning_day = None
        for d in range(len(offsets)):
            if offsets[d] < 0 and abs(mean_profile[d]) > 1.0:
                warning_day = int(offsets[d])
                break

        profiles[feat] = {
            "offsets": offsets,
            "mean": mean_profile,
            "std": std_profile,
            "n_events": len(event_windows),
            "warning_day": warning_day,
        }
    return profiles


def plot_event_profiles_pub(profiles):
    """Publication-quality event profiles."""
    feats = list(profiles.keys())
    n = len(feats)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
    fig.suptitle("Anatomy of a Crash — Feature Evolution Around DOWN Events",
                 fontsize=14, fontweight="bold", y=1.02)

    for i, feat in enumerate(feats):
        r, c = i // ncols, i % ncols
        ax = axes[r][c] if nrows > 1 else axes[c]
        p = profiles[feat]

        ax.fill_between(p["offsets"], p["mean"] - p["std"], p["mean"] + p["std"],
                         alpha=0.2, color="#3498db")
        ax.plot(p["offsets"], p["mean"], "-", color="#2c3e50", linewidth=2, label="Mean z-score")
        ax.axvline(0, color="#e74c3c", linewidth=2, linestyle="--", alpha=0.8, label="Event")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axhline(1, color="#f39c12", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.axhline(-1, color="#f39c12", linewidth=0.8, linestyle=":", alpha=0.6)

        warning = p.get("warning_day")
        group = feature_group(feat)
        title = f"{feat}  [{group}]  (n={p['n_events']})"
        if warning is not None:
            title += f"\nWarning: day {warning}"
            ax.axvline(warning, color="#f39c12", linewidth=1.5, linestyle=":", alpha=0.8)

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Days from event")
        ax.set_ylabel("Z-score")
        ax.legend(fontsize=7, loc="upper left")

    for i in range(n, nrows * ncols):
        r, c = i // ncols, i % ncols
        (axes[r][c] if nrows > 1 else axes[c]).set_visible(False)

    plt.tight_layout()
    _save(fig, "event_profiles")


# =================================================================
# FIGURE 8: EQUITY CURVE
# =================================================================
def run_backtest(features, prices, target_up, target_down, config):
    """Run backtest and return trades."""
    from investigate_v3 import build_signal_rules, backtest_strategy

    rules = build_signal_rules()
    bt_binary = backtest_strategy(features, prices, target_up, target_down, config, rules, sizing_method="binary")
    bt_linear = backtest_strategy(features, prices, target_up, target_down, config, rules, sizing_method="linear")
    return bt_binary, bt_linear


def plot_equity_pub(bt_binary, bt_linear):
    """Publication equity curve."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (bt, label) in enumerate([(bt_binary, "Binary Sizing"), (bt_linear, "Linear Sizing")]):
        if "error" in bt or not bt.get("trades"):
            continue
        trades = pd.DataFrame(bt["trades"])
        m = bt["metrics"]

        # Equity curve
        ax = axes[0, col]
        cum_pnl = trades["pnl"].cumsum()
        ax.plot(cum_pnl.values, color="#2c3e50", linewidth=1.5)
        ax.fill_between(range(len(cum_pnl)), 0, cum_pnl.values, alpha=0.15,
                         where=cum_pnl.values >= 0, color="#2ecc71")
        ax.fill_between(range(len(cum_pnl)), 0, cum_pnl.values, alpha=0.15,
                         where=cum_pnl.values < 0, color="#e74c3c")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_title(f"Equity Curve — {label}\n"
                     f"Return={m['total_return']:.1%}   Sharpe={m['sharpe_ratio']:.2f}   "
                     f"MaxDD={m['max_drawdown']:.1%}", fontweight="bold")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative PnL")

        # Trade distribution
        ax = axes[1, col]
        long_pnl = trades[trades["direction"] == "LONG"]["pnl"]
        short_pnl = trades[trades["direction"] == "SHORT"]["pnl"]
        if len(long_pnl) > 0:
            ax.hist(long_pnl, bins=20, alpha=0.6, color="#2ecc71", label=f"Long ({len(long_pnl)})", edgecolor="white")
        if len(short_pnl) > 0:
            ax.hist(short_pnl, bins=20, alpha=0.6, color="#e74c3c", label=f"Short ({len(short_pnl)})", edgecolor="white")
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_title(f"Trade PnL Distribution — {label}\n"
                     f"Win Rate={m['win_rate']:.0%}   Trades={m['n_trades']}", fontweight="bold")
        ax.set_xlabel("Trade PnL")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    _save(fig, "equity_curve")


# =================================================================
# MAIN
# =================================================================
def main():
    t0 = datetime.now()
    print("=" * 70)
    print("PAPER FIGURE GENERATION")
    print("=" * 70)
    print(f"Start: {t0:%Y-%m-%d %H:%M:%S}\n")

    config = Config()

    # ── Load data ────────────────────────────────────────────────
    print("[1/7] Loading data ...")
    prices, returns = load_data(config)

    print("[2/7] Building features ...")
    spectral_features = build_spectral_features(returns, config)
    traditional_features = build_traditional_features(prices, returns)
    combined = pd.concat([spectral_features, traditional_features], axis=1)
    bench_features = prepare_benchmark_features(prices, returns)

    market = prices["SP500"] if "SP500" in prices.columns else prices.iloc[:, 0]
    fut_3d = market.pct_change(3).shift(-3)
    tasks = {
        "Large UP move 3d (>3%)": (fut_3d > 0.03).astype(int),
        "Large DOWN move 3d (>3%)": (fut_3d < -0.03).astype(int),
    }

    # ── Run all models (for ROC/PR/comparison) ───────────────────
    print("[3/7] Running model comparison ...")
    all_results = {}

    for task_name, target in tasks.items():
        pos_rate = float(target.dropna().mean())
        task_results = []
        task_raw = {}

        # CGECD
        print(f"  CGECD — {task_name} ...")
        res = walk_forward_evaluate(combined, target, CGECDModel, config)
        if "error" not in res:
            m = res["metrics"]
            task_results.append(dict(model="CGECD (Ours)", auc_roc=m.auc_roc,
                                     avg_precision=m.avg_precision, precision=m.precision,
                                     recall=m.recall, f1=m.f1, is_ours=True))
            task_raw["CGECD (Ours)"] = {**res, "is_ours": True}
            print(f"    AUC={m.auc_roc:.3f}")

        # ML baselines
        for bname, bclass in [("Logistic Regression", LogisticRegressionModel),
                               ("SVM", SVMBaselineModel),
                               ("Gradient Boosting", GradientBoostingModel),
                               ("Random Forest", RandomForestBaselineModel)]:
            print(f"  {bname} — {task_name} ...")
            res = walk_forward_evaluate(traditional_features, target, bclass, config)
            if "error" not in res:
                m = res["metrics"]
                task_results.append(dict(model=bname, auc_roc=m.auc_roc,
                                         avg_precision=m.avg_precision, precision=m.precision,
                                         recall=m.recall, f1=m.f1, is_ours=False))
                task_raw[bname] = {**res, "is_ours": False}
                print(f"    AUC={m.auc_roc:.3f}")

        # Signal benchmarks
        for bname, bfeat, bclass in [("Absorption Ratio", bench_features["absorption_ratio"], AbsorptionRatioModel),
                                      ("Turbulence Index", bench_features["turbulence"], TurbulenceModel),
                                      ("GARCH(1,1)", bench_features["garch"], GARCHModel),
                                      ("HAR-RV", bench_features["har_rv"], HARRVModel)]:
            print(f"  {bname} — {task_name} ...")
            res = walk_forward_evaluate(bfeat, target, bclass, config)
            if "error" not in res:
                m = res["metrics"]
                task_results.append(dict(model=bname, auc_roc=m.auc_roc,
                                         avg_precision=m.avg_precision, precision=m.precision,
                                         recall=m.recall, f1=m.f1, is_ours=False))
                task_raw[bname] = {**res, "is_ours": False}
                print(f"    AUC={m.auc_roc:.3f}")

        all_results[task_name] = dict(results=task_results, pos_rate=pos_rate, raw=task_raw)

    # ── SHAP values ──────────────────────────────────────────────
    print("\n[4/7] Computing SHAP values ...")
    shap_dict = {}
    feat_names_dict = {}

    for task_name, target in tasks.items():
        print(f"  {task_name} ...")
        shap_vals, X_test, fnames = compute_shap_values(combined, target, config, n_folds=5)
        if shap_vals is not None:
            shap_dict[task_name] = shap_vals
            feat_names_dict[task_name] = fnames
            plot_shap_beeswarm(shap_vals, X_test, fnames, task_name, top_k=20)

    if shap_dict:
        plot_shap_group_bar(shap_dict, feat_names_dict)

    # ── Curves + Comparison ──────────────────────────────────────
    print("\n[5/7] Plotting ROC, PR, and model comparison ...")
    plot_roc_publication(all_results)
    plot_pr_publication(all_results)
    plot_model_comparison_pub(all_results)

    # ── Ablation ─────────────────────────────────────────────────
    print("\n[6/7] Running ablation + heatmap ...")
    ablation = run_ablation(spectral_features, traditional_features, tasks, config)
    plot_ablation_heatmap(ablation)

    # ── Event profiles + Equity curve ────────────────────────────
    print("\n[7/7] Event profiles + backtest equity curve ...")
    target_down = tasks["Large DOWN move 3d (>3%)"]
    profiles = compute_event_profiles(combined, target_down)
    plot_event_profiles_pub(profiles)

    bt_binary, bt_linear = run_backtest(combined, prices,
                                         tasks["Large UP move 3d (>3%)"],
                                         tasks["Large DOWN move 3d (>3%)"],
                                         config)
    plot_equity_pub(bt_binary, bt_linear)

    # ── Summary ──────────────────────────────────────────────────
    summary = {
        "generated_at": str(datetime.now()),
        "figures": [f.name for f in sorted(OUT.iterdir())],
        "n_figures": len(list(OUT.iterdir())),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = datetime.now() - t0
    print(f"\nDone! {len(list(OUT.iterdir()))} files in {OUT}/")
    print(f"Elapsed: {elapsed}")


if __name__ == "__main__":
    main()
