#!/usr/bin/env python3
"""
Generate publication-quality figures for the CGECD paper.
==========================================================
Based on the latest model (two-task: Rally + Crash).

Figures:
  1. SHAP beeswarm plots (Rally + Crash)
  2. SHAP feature group bar plot
  3. ROC curves (both tasks)
  4. Precision-Recall curves (both tasks)
  5. Model comparison bar chart (BCD-AUC)
  6. Ablation heatmap
  7. Event temporal profiles ("anatomy of a crash")
  8. Equity curve from backtest

Output: results/paper_figures/ (300 DPI PNG + PDF)
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
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, precision_recall_curve

from config import Config
from metrics import compute_metrics
from algorithm import (
    load_data, build_spectral_features, build_traditional_features,
    compute_all_targets, CGECDModel, walk_forward_evaluate,
)
from benchmarks import (
    prepare_benchmark_features,
    RandomForestModel, LogisticRegressionModel,
)

OUT = Path("results/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})

GROUP_PALETTE = {
    "Eigenvalue": "#2ecc71", "Correlation": "#3498db", "Topology": "#9b59b6",
    "Dynamics": "#e74c3c", "Returns": "#f39c12", "Volatility": "#1abc9c",
    "Momentum": "#34495e", "Drawdown": "#e67e22", "TailRisk": "#c0392b",
    "CrossAsset": "#2980b9", "Other": "#95a5a6",
}


def feature_group(name: str) -> str:
    n = name.lower()
    if "_roc_" in n or "_accel_" in n or "_zscore_" in n or "_diff_" in n or "_pctrank_" in n:
        return "Dynamics"
    if any(k in n for k in ("lambda", "spectral_gap", "absorption_ratio", "eigenvalue_entropy",
                             "effective_rank", "mp_excess", "condition_number", "tail_eigenvalue")):
        return "Eigenvalue"
    if any(k in n for k in ("edge_density", "degree_", "clustering_coef", "centralization")):
        return "Topology"
    if "corr" in n or "loading" in n or "v1_" in n or "herfindahl" in n:
        return "Correlation"
    if "drawdown" in n:
        return "Drawdown"
    if any(k in n for k in ("volatility", "vol_ratio", "vol_change", "vol_of_vol", "garch",
                             "range_vol", "var_ratio", "vol_dispersion")):
        return "Volatility"
    if "return" in n:
        return "Returns"
    if any(k in n for k in ("sma", "rsi", "momentum", "price_to_sma")):
        return "Momentum"
    if any(k in n for k in ("skew", "kurtosis", "max_loss", "downside_vol", "down_up_vol", "neg_")):
        return "TailRisk"
    if any(k in n for k in ("credit", "flight", "breadth", "dispersion", "em_stress")):
        return "CrossAsset"
    return "Other"


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png + .pdf")


# =================================================================
# SHAP
# =================================================================
def compute_shap_values(features, target, config, n_folds=5):
    common_idx = features.dropna(thresh=int(len(features.columns) * 0.5)).index
    common_idx = common_idx.intersection(target.dropna().index)
    X = features.loc[common_idx]
    y = target.loc[common_idx]
    feat_names = list(X.columns)

    train_size = int(config.train_years * 252)
    test_size = int(config.test_months * 21)
    gap = config.gap_days
    available = len(X) - train_size - gap - test_size
    step = max(test_size, available // config.n_splits)

    # Use last n_folds
    all_folds = []
    for fold in range(config.n_splits):
        start = fold * step
        train_end = start + train_size
        test_start = train_end + gap
        test_end = min(test_start + test_size, len(X))
        if test_end > len(X):
            break
        all_folds.append((start, train_end, test_start, test_end))

    folds_to_use = all_folds[-n_folds:]
    all_shap, all_X_test = [], []

    for i, (start, train_end, test_start, test_end) in enumerate(folds_to_use):
        print(f"    SHAP fold {i+1}/{len(folds_to_use)} ...")
        X_tr = X.iloc[start:train_end].values
        y_tr = y.iloc[start:train_end].values
        X_te = X.iloc[test_start:test_end].values

        if len(np.unique(y_tr)) < 2:
            continue

        scaler = RobustScaler()
        Xs_tr = scaler.fit_transform(np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0))
        Xs_te = scaler.transform(np.nan_to_num(X_te, nan=0, posinf=0, neginf=0))

        rf = RandomForestClassifier(
            n_estimators=config.rf_n_estimators, max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf, min_samples_split=config.rf_min_samples_split,
            class_weight="balanced_subsample", random_state=config.random_seed, n_jobs=-1,
        )
        rf.fit(Xs_tr, y_tr)

        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(Xs_te)
        if isinstance(sv, list):
            sv = sv[1]
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        all_shap.append(sv)
        all_X_test.append(Xs_te)

    if not all_shap:
        return None, None, feat_names
    return np.vstack(all_shap), np.vstack(all_X_test), feat_names


def plot_shap_beeswarm(shap_values, X_test, feat_names, task_label, top_k=20):
    mean_abs = np.mean(np.abs(shap_values), axis=0).ravel()
    top_idx = np.argsort(mean_abs)[::-1][:top_k].ravel()

    shap_top = shap_values[:, top_idx]
    X_top = X_test[:, top_idx]
    names_top = [feat_names[int(i)] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(len(names_top)):
        y_pos = len(names_top) - 1 - i
        sv = shap_top[:, i]
        fv = X_top[:, i]
        fv_min, fv_max = fv.min(), fv.max()
        fv_norm = (fv - fv_min) / (fv_max - fv_min) if fv_max - fv_min > 1e-10 else np.full_like(fv, 0.5)
        jitter = np.random.RandomState(42).normal(0, 0.15, len(sv))
        ax.scatter(sv, y_pos + jitter, c=fv_norm, cmap="coolwarm", s=3, alpha=0.5, rasterized=True)

    ax.set_yticks(range(len(names_top)))
    ax.set_yticklabels(list(reversed(names_top)), fontsize=9)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(f"SHAP Feature Importance — {task_label}", fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, aspect=40, pad=0.02)
    cbar.set_label("Feature value (normalized)", fontsize=9)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    _save(fig, f"shap_beeswarm_{task_label}")


def plot_shap_group_bar(shap_dict, feat_names_dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (task, sv) in zip(axes, shap_dict.items()):
        fn = feat_names_dict[task]
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        mean_abs = np.mean(np.abs(sv), axis=0).ravel()
        group_imp = {}
        for i, f in enumerate(fn):
            g = feature_group(f)
            group_imp[g] = group_imp.get(g, 0) + mean_abs[i]
        sg = sorted(group_imp.items(), key=lambda x: x[1], reverse=True)
        names = [g for g, _ in sg]
        vals = [v for _, v in sg]
        colors = [GROUP_PALETTE.get(g, "#95a5a6") for g in names]
        ax.barh(range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP| value")
        ax.set_title(f"Feature Group Importance — {task}", fontweight="bold")
    plt.tight_layout()
    _save(fig, "shap_group_importance")


# =================================================================
# ROC + PR CURVES
# =================================================================
def plot_roc_pub(all_results):
    tasks = list(all_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, task in zip(axes, tasks):
        raw = all_results[task]["raw"]
        for mn, res in raw.items():
            if "actuals" not in res:
                continue
            fpr, tpr, _ = roc_curve(res["actuals"], res["probabilities"])
            auc = res["metrics"].auc_roc
            is_ours = res.get("role") == "ours"
            lw = 2.5 if is_ours else 1.2
            ls = "-" if is_ours else "--"
            color = "#e74c3c" if is_ours else None
            ax.plot(fpr, tpr, label=f"{mn} ({auc:.3f})", linewidth=lw, linestyle=ls, color=color)
        ax.plot([0, 1], [0, 1], "k:", alpha=0.4, linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {task}", fontweight="bold")
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    _save(fig, "roc_curves")


def plot_pr_pub(all_results):
    tasks = list(all_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, task in zip(axes, tasks):
        raw = all_results[task]["raw"]
        pr = all_results[task]["pos_rate"]
        for mn, res in raw.items():
            if "actuals" not in res:
                continue
            prec, rec, _ = precision_recall_curve(res["actuals"], res["probabilities"])
            ap = res["metrics"].avg_precision
            is_ours = res.get("role") == "ours"
            lw = 2.5 if is_ours else 1.2
            ls = "-" if is_ours else "--"
            color = "#e74c3c" if is_ours else None
            ax.plot(rec, prec, label=f"{mn} (AP={ap:.3f})", linewidth=lw, linestyle=ls, color=color)
        ax.axhline(pr, color="grey", linewidth=0.8, linestyle=":", alpha=0.5, label=f"Baseline ({pr:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision–Recall — {task}", fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    _save(fig, "pr_curves")


# =================================================================
# MODEL COMPARISON + BCD-AUC
# =================================================================
def plot_model_comparison(all_results, bcd_rows):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Per-task
    for ax, task in zip(axes[:2], all_results.keys()):
        res = sorted(all_results[task]["results"], key=lambda x: x["auc"])
        names = [r["model"] for r in res]
        aucs = [r["auc"] for r in res]
        colors = ["#e74c3c" if r["role"] == "ours" else "#3498db" if r["role"] == "bench" else "#f39c12" for r in res]
        bars = ax.barh(range(len(names)), aucs, color=colors, edgecolor="white", height=0.6)
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{auc:.3f}", va="center", fontsize=9, fontweight="bold")
        ax.axvline(0.5, color="red", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("AUC-ROC")
        ax.set_xlim(0.4, 1.0)
        ax.set_title(f"{task}", fontweight="bold")

    # BCD-AUC
    ax = axes[2]
    sr = sorted(bcd_rows, key=lambda x: x["gmean"])
    names = [r["model"] for r in sr]
    vals = [r["gmean"] for r in sr]
    colors = ["#e74c3c" if r["role"] == "ours" else "#3498db" if r["role"] == "bench" else "#f39c12" for r in sr]
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="white", height=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("BCD-AUC")
    ax.set_xlim(0.4, 1.0)
    ax.set_title("BCD-AUC (Balanced Crisis Detection)", fontweight="bold")

    plt.tight_layout()
    _save(fig, "model_comparison")


# =================================================================
# ABLATION HEATMAP
# =================================================================
def run_ablation(spectral, traditional, tasks, config):
    combined = pd.concat([spectral, traditional], axis=1)
    configs_map = {
        "Full (Combined)": list(combined.columns),
        "Spectral Only": list(spectral.columns),
        "Traditional Only": list(traditional.columns),
    }
    results = {}
    for task_name, target in tasks.items():
        task_res = {}
        for cfg_name, cols in configs_map.items():
            print(f"    Ablation: {task_name} / {cfg_name} ({len(cols)} feat) ...")
            res = walk_forward_evaluate(combined[cols] if cols != list(combined.columns) else combined,
                                         target, CGECDModel, config)
            if "error" not in res:
                task_res[cfg_name] = {"auc_roc": float(res["metrics"].auc_roc),
                                      "avg_precision": float(res["metrics"].avg_precision)}
            else:
                task_res[cfg_name] = {"auc_roc": float("nan"), "avg_precision": float("nan")}
        results[task_name] = task_res
    return results


def plot_ablation_heatmap(ablation):
    configs = list(next(iter(ablation.values())).keys())
    tasks = list(ablation.keys())
    short_tasks = ["Rally" if "Rally" in t else "Crash" for t in tasks]

    auc_matrix = np.zeros((len(tasks), len(configs)))
    for i, task in enumerate(tasks):
        for j, cfg in enumerate(configs):
            auc_matrix[i, j] = ablation[task].get(cfg, {}).get("auc_roc", float("nan"))

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(auc_matrix, cmap="RdYlGn", aspect="auto", vmin=0.45, vmax=0.85)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(short_tasks, fontsize=11)
    ax.set_title("Feature Ablation — AUC-ROC", fontweight="bold")

    for i in range(len(tasks)):
        for j in range(len(configs)):
            val = auc_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color="white" if val < 0.6 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    _save(fig, "ablation_heatmap")


# =================================================================
# EVENT PROFILES
# =================================================================
def compute_event_profiles(features, target, window=20):
    profile_features = [f for f in ["lambda_1_60d", "lambda_2_60d", "spectral_gap_60d",
                                     "eigenvalue_entropy_60d", "absorption_ratio_5_60d",
                                     "mean_abs_corr_60d", "volatility_20d", "garch_vol"]
                        if f in features.columns]

    common_idx = features.dropna(thresh=int(len(features.columns) * 0.5)).index
    common_idx = common_idx.intersection(target.dropna().index)
    X = features.loc[common_idx]
    y = target.loc[common_idx]
    event_dates = y[y == 1].index
    all_dates = list(X.index)

    profiles = {}
    for feat in profile_features:
        vals = X[feat].values
        event_windows = []
        for edate in event_dates:
            if edate not in all_dates:
                continue
            eidx = all_dates.index(edate)
            if eidx < window + 20 or eidx + window >= len(all_dates):
                continue
            win = vals[eidx - window: eidx + window + 1]
            baseline = vals[max(0, eidx - 40): eidx - 21]
            if len(baseline) > 0:
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
        mean_p = np.mean(windows_arr, axis=0)
        std_p = np.std(windows_arr, axis=0)

        warning_day = None
        for d in range(len(offsets)):
            if offsets[d] < 0 and abs(mean_p[d]) > 1.0:
                warning_day = int(offsets[d])
                break

        profiles[feat] = {"offsets": offsets, "mean": mean_p, "std": std_p,
                          "n_events": len(event_windows), "warning_day": warning_day}
    return profiles


def plot_event_profiles(profiles):
    feats = list(profiles.keys())
    n = len(feats)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Anatomy of a Crash — Feature Evolution Around Events",
                 fontsize=14, fontweight="bold", y=1.02)

    for i, feat in enumerate(feats):
        r, c = i // ncols, i % ncols
        ax = axes[r][c]
        p = profiles[feat]
        ax.fill_between(p["offsets"], p["mean"] - p["std"], p["mean"] + p["std"], alpha=0.2, color="#3498db")
        ax.plot(p["offsets"], p["mean"], "-", color="#2c3e50", linewidth=2)
        ax.axvline(0, color="#e74c3c", linewidth=2, linestyle="--", alpha=0.8)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axhline(1, color="#f39c12", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.axhline(-1, color="#f39c12", linewidth=0.8, linestyle=":", alpha=0.6)
        warning = p.get("warning_day")
        title = f"{feat}  [{feature_group(feat)}]  (n={p['n_events']})"
        if warning is not None:
            title += f"\nWarning: day {warning}"
            ax.axvline(warning, color="#f39c12", linewidth=1.5, linestyle=":", alpha=0.8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Days from event")
        ax.set_ylabel("Z-score")

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)
    plt.tight_layout()
    _save(fig, "event_profiles")


# =================================================================
# EQUITY CURVE (simple backtest)
# =================================================================
def simple_backtest(features, prices, target_up, target_down, config):
    market = prices["SP500"] if "SP500" in prices.columns else prices.iloc[:, 0]
    market_ret = market.pct_change().fillna(0)

    common_idx = features.dropna(thresh=int(len(features.columns) * 0.5)).index
    for t in [target_up, target_down]:
        common_idx = common_idx.intersection(t.dropna().index)
    X = features.loc[common_idx]

    train_size = int(config.train_years * 252)
    test_size = int(config.test_months * 21)
    gap = config.gap_days
    available = len(X) - train_size - gap - test_size
    step = max(test_size, available // config.n_splits)

    trades = []
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

        rf_params = dict(n_estimators=config.rf_n_estimators, max_depth=config.rf_max_depth,
                         min_samples_leaf=config.rf_min_samples_leaf, min_samples_split=config.rf_min_samples_split,
                         class_weight="balanced_subsample", random_state=config.random_seed, n_jobs=-1)

        rf_up = RandomForestClassifier(**rf_params)
        rf_up.fit(Xs_tr, y_up_tr)
        prob_up = rf_up.predict_proba(Xs_te)[:, 1]

        rf_down = RandomForestClassifier(**rf_params)
        rf_down.fit(Xs_tr, y_down_tr)
        prob_down = rf_down.predict_proba(Xs_te)[:, 1]

        pos_end = -1
        hold = 10  # 10-day holding to match target horizon
        for i in range(len(test_dates)):
            if i <= pos_end:
                continue
            date = test_dates[i]
            end_i = min(i + hold, len(test_dates) - 1)
            if end_i <= i:
                continue
            tr = market_ret.reindex(test_dates[i + 1: end_i + 1]).values
            if len(tr) == 0:
                continue

            if prob_up[i] > 0.4:
                pnl = float(np.sum(tr))
                trades.append({"direction": "LONG", "pnl": pnl, "prob": float(prob_up[i]),
                               "entry": str(date.date()), "exit": str(test_dates[end_i].date())})
                pos_end = end_i
            elif prob_down[i] > 0.4:
                pnl = float(-np.sum(tr))
                trades.append({"direction": "SHORT", "pnl": pnl, "prob": float(prob_down[i]),
                               "entry": str(date.date()), "exit": str(test_dates[end_i].date())})
                pos_end = end_i

    if not trades:
        return None
    tdf = pd.DataFrame(trades)
    cum = tdf["pnl"].cumsum()
    sharpe = float(tdf["pnl"].mean() / (tdf["pnl"].std() + 1e-10) * np.sqrt(252 / 10))
    return {"trades": trades, "equity": cum.values.tolist(),
            "total_return": float(cum.iloc[-1]),
            "n_trades": len(tdf), "win_rate": float((tdf["pnl"] > 0).mean()),
            "sharpe": sharpe, "max_dd": float((cum - cum.cummax()).min()),
            "long": int((tdf["direction"] == "LONG").sum()),
            "short": int((tdf["direction"] == "SHORT").sum())}


def plot_equity(bt):
    if bt is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    eq = bt["equity"]
    ax1.plot(eq, color="#2c3e50", linewidth=1.5)
    ax1.fill_between(range(len(eq)), 0, eq, where=np.array(eq) >= 0, alpha=0.15, color="#2ecc71")
    ax1.fill_between(range(len(eq)), 0, eq, where=np.array(eq) < 0, alpha=0.15, color="#e74c3c")
    ax1.axhline(0, color="grey", linewidth=0.5)
    ax1.set_title(f"Equity Curve\nReturn={bt['total_return']:.1%}  Sharpe={bt['sharpe']:.2f}  "
                  f"MaxDD={bt['max_dd']:.1%}", fontweight="bold")
    ax1.set_xlabel("Trade #")
    ax1.set_ylabel("Cumulative PnL")

    tdf = pd.DataFrame(bt["trades"])
    long_pnl = tdf[tdf["direction"] == "LONG"]["pnl"]
    short_pnl = tdf[tdf["direction"] == "SHORT"]["pnl"]
    if len(long_pnl) > 0:
        ax2.hist(long_pnl, bins=20, alpha=0.6, color="#2ecc71", label=f"Long ({len(long_pnl)})", edgecolor="white")
    if len(short_pnl) > 0:
        ax2.hist(short_pnl, bins=20, alpha=0.6, color="#e74c3c", label=f"Short ({len(short_pnl)})", edgecolor="white")
    ax2.axvline(0, color="grey", linewidth=0.8)
    ax2.set_title(f"Trade PnL Distribution\nWin Rate={bt['win_rate']:.0%}  Trades={bt['n_trades']}", fontweight="bold")
    ax2.set_xlabel("Trade PnL")
    ax2.legend()
    plt.tight_layout()
    _save(fig, "equity_curve")


# =================================================================
# MAIN
# =================================================================
def main():
    t0 = datetime.now()
    print("=" * 70)
    print("PAPER FIGURE GENERATION (New Model)")
    print("=" * 70)

    cfg = Config()

    print("[1/7] Loading data ...")
    prices, returns = load_data(cfg)

    print("[2/7] Building features ...")
    spectral = build_spectral_features(returns, cfg)
    traditional = build_traditional_features(prices, returns)
    combined = pd.concat([spectral, traditional], axis=1)
    bench_feat = prepare_benchmark_features(prices, returns)

    all_targets = compute_all_targets(prices)
    tasks = {
        "Rally (Up >3% 10d)": all_targets["up_3pct_10d"],
        "Crash (DD >7% 10d)": all_targets["drawdown_7pct_10d"],
    }

    # ── Run all models ───────────────────────────────────────────
    print("[3/7] Running model comparison ...")
    methods = [
        ("CGECD Combined (Ours)", combined, CGECDModel, "ours"),
        ("Spectral Only RF", spectral, RandomForestModel, "ablation"),
        ("Traditional RF", traditional, RandomForestModel, "bench"),
        ("Turbulence RF", bench_feat["turbulence"], RandomForestModel, "bench"),
        ("HAR-RV LR", bench_feat["har_rv"], LogisticRegressionModel, "bench"),
        ("SMA Vol LR", bench_feat["sma_vol"], LogisticRegressionModel, "bench"),
    ]

    all_results = {}
    auc_map = {}

    for task_name, target in tasks.items():
        pr = float(target.dropna().mean())
        task_res, task_raw = [], {}
        for ml, mf, mc, mr in methods:
            print(f"  {ml} — {task_name} ...")
            try:
                r = walk_forward_evaluate(mf, target, mc, cfg)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
            if "error" not in r:
                m = r["metrics"]
                task_res.append(dict(model=ml, nf=len(mf.columns), role=mr,
                                     auc=m.auc_roc, avgp=m.avg_precision,
                                     prec=m.precision, rec=m.recall, f1=m.f1))
                task_raw[ml] = {**r, "role": mr}
                auc_map.setdefault(ml, {"role": mr})[task_name] = m.auc_roc
                print(f"    AUC={m.auc_roc:.3f}")
        all_results[task_name] = dict(results=task_res, pos_rate=pr, raw=task_raw)

    # BCD-AUC
    bcd_rows = []
    task_keys = list(tasks.keys())
    for mn, info in auc_map.items():
        u = info.get(task_keys[0], 0.5)
        d = info.get(task_keys[1], 0.5)
        bcd_rows.append(dict(model=mn, up=u, down=d, gmean=np.sqrt(u * d),
                              amean=(u + d) / 2, role=info["role"]))
    bcd_rows.sort(key=lambda x: x["gmean"], reverse=True)

    # ── SHAP ─────────────────────────────────────────────────────
    print("\n[4/7] SHAP values ...")
    shap_dict, fn_dict = {}, {}
    for task_name, target in tasks.items():
        label = "Rally" if "Rally" in task_name else "Crash"
        print(f"  {task_name} ...")
        sv, xt, fn = compute_shap_values(combined, target, cfg, n_folds=5)
        if sv is not None:
            shap_dict[label] = sv
            fn_dict[label] = fn
            plot_shap_beeswarm(sv, xt, fn, label, top_k=20)
    if shap_dict:
        plot_shap_group_bar(shap_dict, fn_dict)

    # ── Curves + Comparison ──────────────────────────────────────
    print("\n[5/7] ROC, PR, model comparison ...")
    plot_roc_pub(all_results)
    plot_pr_pub(all_results)
    plot_model_comparison(all_results, bcd_rows)

    # ── Ablation ─────────────────────────────────────────────────
    print("\n[6/7] Ablation heatmap ...")
    ablation = run_ablation(spectral, traditional, tasks, cfg)
    plot_ablation_heatmap(ablation)

    # ── Event profiles + Equity ──────────────────────────────────
    print("\n[7/7] Event profiles + equity curve ...")
    crash_target = tasks[task_keys[1]]
    profiles = compute_event_profiles(combined, crash_target)
    plot_event_profiles(profiles)

    bt = simple_backtest(combined, prices, tasks[task_keys[0]], tasks[task_keys[1]], cfg)
    plot_equity(bt)

    summary = {"generated_at": str(datetime.now()),
               "figures": sorted([f.name for f in OUT.iterdir()])}
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! {len(list(OUT.iterdir()))} files in {OUT}/")
    print(f"Elapsed: {datetime.now() - t0}")


if __name__ == "__main__":
    main()
