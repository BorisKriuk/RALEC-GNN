#!/usr/bin/env python3
"""
investigate_v2.py — Unbiased Feature Impact & Threshold Analysis
================================================================

Key improvements over Boris's investigate.py:
A. No MI-selection bias — SHAP & permutation importance on ALL 85 features
B. Event fingerprinting — what do pre-UP/DOWN days actually look like
C. Concrete value thresholds — specific trigger levels with precision/recall
D. Joint thresholds — feature PAIRS that create the strongest signals
E. Regime-conditional importance — does the mechanism change by market state
F. Non-linear SHAP dependence — binned mean curves reveal threshold effects

Output → results/investigate_v2/
"""

import warnings

warnings.filterwarnings("ignore")

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("shap not installed — SHAP analysis will be skipped.")

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
)

OUT = Path("results/investigate_v2")
OUT.mkdir(parents=True, exist_ok=True)


# ── Feature grouping ────────────────────────────────────────────
def feature_group(name: str) -> str:
    n = name.lower()
    if "_zscore_" in n or "_roc_" in n or "_accel_" in n:
        return "Dynamics"
    if any(
        k in n
        for k in (
            "lambda",
            "spectral_gap",
            "absorption_ratio",
            "eigenvalue_entropy",
            "effective_rank",
            "mp_excess",
        )
    ):
        return "Eigenvalue"
    if "edge_density" in n:
        return "Topology"
    if "corr" in n:
        return "Correlation"
    if "drawdown" in n:
        return "Drawdown"
    if any(
        k in n
        for k in ("volatility", "vol_ratio", "vol_change", "vol_of_vol", "garch")
    ):
        return "Volatility"
    if "return" in n:
        return "Returns"
    if any(k in n for k in ("sma", "rsi", "momentum")):
        return "Momentum"
    if any(
        k in n
        for k in (
            "skew",
            "kurtosis",
            "max_loss",
            "downside_vol",
            "down_up_vol",
            "neg_days",
        )
    ):
        return "TailRisk"
    if any(
        k in n for k in ("credit", "flight", "breadth", "dispersion", "em_stress")
    ):
        return "CrossAsset"
    return "Other"


PALETTE = {
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


def _safe(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(">", "gt")
        .replace("%", "pct")
    )


# =================================================================
# A. UNBIASED IMPORTANCE (ALL FEATURES)
# =================================================================
def unbiased_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feat_names: List[str],
    config: Config,
) -> Dict:
    scaler = RobustScaler()
    Xs_tr = scaler.fit_transform(X_train)
    Xs_te = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_leaf=config.rf_min_samples_leaf,
        min_samples_split=config.rf_min_samples_split,
        class_weight="balanced_subsample",
        random_state=config.random_seed,
        oob_score=True,
        n_jobs=-1,
    )
    rf.fit(Xs_tr, y_train)

    rf_imp = rf.feature_importances_

    try:
        oob_auc = roc_auc_score(y_train, rf.oob_decision_function_[:, 1])
    except Exception:
        oob_auc = 0.5

    try:
        test_probs = rf.predict_proba(Xs_te)[:, 1]
        test_auc = roc_auc_score(y_test, test_probs)
    except Exception:
        test_auc = 0.5
        test_probs = np.full(len(y_test), 0.5)

    perm = permutation_importance(
        rf,
        Xs_te,
        y_test,
        n_repeats=10,
        scoring="roc_auc",
        random_state=config.random_seed,
        n_jobs=-1,
    )

    shap_values = None
    if HAS_SHAP:
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(Xs_te)
        if isinstance(sv, list):
            sv = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]
        shap_values = sv

    return {
        "rf": rf,
        "scaler": scaler,
        "rf_imp": rf_imp,
        "perm_imp": perm.importances_mean,
        "perm_std": perm.importances_std,
        "shap_values": shap_values,
        "shap_X": Xs_te,
        "oob_auc": oob_auc,
        "test_auc": test_auc,
        "test_probs": test_probs,
    }


# =================================================================
# B. EVENT FINGERPRINTING
# =================================================================
def event_fingerprinting(
    features: pd.DataFrame,
    target: pd.Series,
    feat_names: List[str],
) -> pd.DataFrame:
    valid = target.dropna().index
    X = features.reindex(valid).ffill().fillna(0)
    y = target.loc[valid]

    event_mask = y.values == 1
    n_events = event_mask.sum()

    results = []
    for name in feat_names:
        vals = X[name].values
        event_vals = vals[event_mask]
        non_vals = vals[~event_mask]

        event_mean = np.mean(event_vals)
        non_mean = np.mean(non_vals)
        pop_std = np.std(vals)
        z_score = (event_mean - non_mean) / (pop_std + 1e-10)

        ks_stat, ks_pval = sp_stats.ks_2samp(event_vals, non_vals)

        quintile_edges = np.percentile(vals, [0, 20, 40, 60, 80, 100])
        quintile_pct = np.zeros(5)
        for q in range(5):
            low = quintile_edges[q]
            high = quintile_edges[q + 1]
            if q == 4:
                mask = (event_vals >= low) & (event_vals <= high)
            else:
                mask = (event_vals >= low) & (event_vals < high)
            quintile_pct[q] = np.sum(mask) / max(n_events, 1) * 100

        peak_q = np.argmax(quintile_pct)

        results.append(
            {
                "feature": name,
                "group": feature_group(name),
                "event_mean": event_mean,
                "non_event_mean": non_mean,
                "z_score": z_score,
                "ks_stat": ks_stat,
                "ks_pval": ks_pval,
                "Q1_pct": quintile_pct[0],
                "Q2_pct": quintile_pct[1],
                "Q3_pct": quintile_pct[2],
                "Q4_pct": quintile_pct[3],
                "Q5_pct": quintile_pct[4],
                "peak_quintile": peak_q + 1,
                "peak_concentration": quintile_pct[peak_q],
            }
        )

    return pd.DataFrame(results)


# =================================================================
# C. VALUE THRESHOLDS
# =================================================================
def value_thresholds(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feat_names: List[str],
    top_k: int = 25,
    feature_ranking: np.ndarray = None,
) -> pd.DataFrame:
    if feature_ranking is None:
        feature_ranking = np.ones(len(feat_names))

    top_idx = np.argsort(feature_ranking)[::-1][:top_k]

    results = []
    for feat_i in top_idx:
        name = feat_names[feat_i]

        tree = DecisionTreeClassifier(
            max_depth=1, class_weight="balanced", random_state=42
        )
        tree.fit(X_train[:, feat_i : feat_i + 1], y_train)

        threshold = tree.tree_.threshold[0]
        left_val = tree.tree_.value[1]
        right_val = tree.tree_.value[2]
        left_pos = left_val[0][1] / left_val[0].sum() if left_val[0].sum() > 0 else 0
        right_pos = (
            right_val[0][1] / right_val[0].sum() if right_val[0].sum() > 0 else 0
        )

        if left_pos > right_pos:
            direction = "<"
            test_pred = (X_test[:, feat_i] <= threshold).astype(int)
        else:
            direction = ">"
            test_pred = (X_test[:, feat_i] > threshold).astype(int)

        if test_pred.sum() > 0 and y_test.sum() > 0:
            prec = precision_score(y_test, test_pred, zero_division=0)
            rec = recall_score(y_test, test_pred, zero_division=0)
        else:
            prec = 0
            rec = 0

        n_captured = int(np.sum((test_pred == 1) & (y_test == 1)))
        n_total_ev = int(y_test.sum())
        n_signals = int(test_pred.sum())
        pctile = sp_stats.percentileofscore(X_train[:, feat_i], threshold)

        results.append(
            {
                "feature": name,
                "group": feature_group(name),
                "threshold": threshold,
                "direction": direction,
                "threshold_percentile": pctile,
                "precision": prec,
                "recall": rec,
                "events_captured": n_captured,
                "total_events": n_total_ev,
                "total_signals": n_signals,
                "importance": feature_ranking[feat_i],
            }
        )

    return pd.DataFrame(results)


# =================================================================
# D. JOINT THRESHOLDS
# =================================================================
def joint_thresholds(
    X_test: np.ndarray,
    y_test: np.ndarray,
    feat_names: List[str],
    feature_ranking: np.ndarray,
    X_train: np.ndarray,
    top_k: int = 10,
    n_best: int = 30,
) -> pd.DataFrame:
    top_idx = np.argsort(feature_ranking)[::-1][:top_k]
    base_rate = y_test.mean()

    percentiles = [10, 25, 50, 75, 90]
    results = []

    for i in range(len(top_idx)):
        for j in range(i + 1, len(top_idx)):
            fi, fj = top_idx[i], top_idx[j]

            for pi in percentiles:
                thresh_i = np.percentile(X_train[:, fi], pi)
                for dir_i in ["<", ">"]:
                    mask_i = (
                        (X_test[:, fi] < thresh_i)
                        if dir_i == "<"
                        else (X_test[:, fi] > thresh_i)
                    )

                    for pj in percentiles:
                        thresh_j = np.percentile(X_train[:, fj], pj)
                        for dir_j in ["<", ">"]:
                            mask_j = (
                                (X_test[:, fj] < thresh_j)
                                if dir_j == "<"
                                else (X_test[:, fj] > thresh_j)
                            )

                            joint = mask_i & mask_j
                            n_sig = joint.sum()
                            if n_sig < 3:
                                continue

                            n_hits = int((joint & (y_test == 1)).sum())
                            prec = n_hits / n_sig
                            rec = n_hits / max(y_test.sum(), 1)
                            lift = prec / max(base_rate, 1e-10)

                            results.append(
                                {
                                    "feature_A": feat_names[fi],
                                    "feature_B": feat_names[fj],
                                    "thresh_A": float(thresh_i),
                                    "dir_A": dir_i,
                                    "pctile_A": pi,
                                    "thresh_B": float(thresh_j),
                                    "dir_B": dir_j,
                                    "pctile_B": pj,
                                    "precision": prec,
                                    "recall": rec,
                                    "lift": lift,
                                    "n_signals": int(n_sig),
                                    "n_hits": n_hits,
                                }
                            )

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values("precision", ascending=False)
        .head(n_best)
        .reset_index(drop=True)
    )


# =================================================================
# E. REGIME-CONDITIONAL IMPORTANCE
# =================================================================
def regime_conditional_importance(
    features: pd.DataFrame,
    target: pd.Series,
    feat_names: List[str],
    config: Config,
    split_feature: str = "volatility_20d",
) -> Dict:
    valid = target.dropna().index
    X = features.reindex(valid).ffill().fillna(0)
    y = target.loc[valid]

    split_col = X[split_feature].values
    median_val = np.median(split_col)

    results = {}
    for regime_name, regime_mask in [
        ("high_vol", split_col > median_val),
        ("low_vol", split_col <= median_val),
    ]:
        X_r = X.values[regime_mask]
        y_r = y.values[regime_mask]

        if len(np.unique(y_r)) < 2 or y_r.sum() < 5:
            continue

        split_pt = int(len(X_r) * 0.7)
        X_tr = np.nan_to_num(X_r[:split_pt], nan=0, posinf=0, neginf=0)
        y_tr = y_r[:split_pt]
        X_te = np.nan_to_num(X_r[split_pt:], nan=0, posinf=0, neginf=0)
        y_te = y_r[split_pt:]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        scaler = RobustScaler()
        Xs_tr = scaler.fit_transform(X_tr)
        Xs_te = scaler.transform(X_te)

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            min_samples_split=config.rf_min_samples_split,
            class_weight="balanced_subsample",
            random_state=config.random_seed,
            n_jobs=-1,
        )
        rf.fit(Xs_tr, y_tr)

        try:
            perm = permutation_importance(
                rf,
                Xs_te,
                y_te,
                n_repeats=10,
                scoring="roc_auc",
                random_state=config.random_seed,
                n_jobs=-1,
            )
            results[regime_name] = {
                "perm_imp": perm.importances_mean,
                "n_events": int(y_r.sum()),
                "n_total": len(y_r),
                "event_rate": float(y_r.mean()),
            }
        except Exception:
            pass

    return results


# =================================================================
# PLOTS
# =================================================================
def plot_unbiased_importance(
    perm_imp: np.ndarray,
    shap_imp: np.ndarray,
    feat_names: List[str],
    task: str,
    top_k: int = 30,
) -> None:
    # Rank by SHAP (more granular than permutation)
    idx = np.argsort(shap_imp)[::-1][:top_k]

    names = [feat_names[i] for i in idx]
    groups = [feature_group(n) for n in names]
    colors = [PALETTE.get(g, "#95a5a6") for g in groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle(
        f"Unbiased Feature Importance (ALL {len(feat_names)} features) — {task}",
        fontsize=14,
        fontweight="bold",
    )

    ax1.barh(range(len(names)), perm_imp[idx], color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Permutation Importance (AUC drop when shuffled)")
    ax1.set_title("Permutation Importance")
    ax1.axvline(0, color="grey", linewidth=0.5)

    ax2.barh(range(len(names)), shap_imp[idx], color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean |SHAP value|")
    ax2.set_title("SHAP Importance")

    seen = set()
    handles = []
    for g in groups:
        if g not in seen:
            seen.add(g)
            handles.append(Rectangle((0, 0), 1, 1, fc=PALETTE.get(g, "#95a5a6"), label=g))
    ax2.legend(handles=handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / f"unbiased_importance_{_safe(task)}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_event_fingerprint(
    fp: pd.DataFrame,
    task: str,
    top_k: int = 25,
) -> None:
    fp_sorted = fp.sort_values("ks_stat", ascending=False).head(top_k)

    names = fp_sorted["feature"].values
    groups = fp_sorted["group"].values
    q_data = fp_sorted[["Q1_pct", "Q2_pct", "Q3_pct", "Q4_pct", "Q5_pct"]].values

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(20, 10), gridspec_kw={"width_ratios": [3, 1]}
    )
    fig.suptitle(
        f"Event Fingerprint — {task}\nQuintile Concentration (20% = random)",
        fontsize=14,
        fontweight="bold",
    )

    im = ax1.imshow(q_data, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=60)
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"], fontsize=10)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels([f"{n}  [{g}]" for n, g in zip(names, groups)], fontsize=8)
    ax1.set_title("% of events in each quintile")

    for i in range(len(names)):
        for j in range(5):
            val = q_data[i, j]
            color = "white" if val > 40 else "black"
            ax1.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax1, shrink=0.8, label="% of events")

    z_scores = fp_sorted["z_score"].values
    z_colors = ["#e74c3c" if z > 0 else "#3498db" for z in z_scores]
    ax2.barh(range(len(names)), z_scores, color=z_colors, edgecolor="white", linewidth=0.5)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Z-score\n(event mean vs population)")
    ax2.set_title("Direction")
    ax2.axvline(0, color="grey", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUT / f"event_fingerprint_{_safe(task)}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_threshold_analysis(thresholds: pd.DataFrame, task: str) -> None:
    n = len(thresholds)
    fig, ax = plt.subplots(figsize=(14, 10))

    names = [
        f"{r['feature']} {r['direction']} {r['threshold']:.3f}" for _, r in thresholds.iterrows()
    ]
    colors = [PALETTE.get(r["group"], "#95a5a6") for _, r in thresholds.iterrows()]

    ax.barh(
        np.arange(n) - 0.2,
        thresholds["precision"].values * 100,
        0.35,
        color=colors,
        alpha=0.9,
        label="Precision %",
        edgecolor="white",
    )
    ax.barh(
        np.arange(n) + 0.2,
        thresholds["recall"].values * 100,
        0.35,
        color=colors,
        alpha=0.4,
        label="Recall %",
        edgecolor="white",
    )

    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Percentage (%)")
    ax.set_title(
        f"Single-Feature Thresholds — {task}\n(DecisionStump optimal splits)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right")

    for i, (_, r) in enumerate(thresholds.iterrows()):
        ax.text(
            max(r["precision"], r["recall"]) * 100 + 1,
            i,
            f"{r['events_captured']}/{r['total_events']} events, {r['total_signals']} signals",
            va="center",
            fontsize=6,
        )

    plt.tight_layout()
    plt.savefig(OUT / f"thresholds_{_safe(task)}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_joint_thresholds(jt: pd.DataFrame, task: str, top_k: int = 20) -> None:
    if jt.empty:
        return

    top = jt.head(top_k)
    labels = []
    for _, r in top.iterrows():
        labels.append(
            f"{r['feature_A']} {r['dir_A']} P{r['pctile_A']:.0f}  "
            f"AND  {r['feature_B']} {r['dir_B']} P{r['pctile_B']:.0f}"
        )

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.barh(range(len(labels)), top["precision"].values * 100, color="#e74c3c", alpha=0.7, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Precision (%)")
    ax.set_title(
        f"Joint Threshold Rules — {task}\n(Feature A AND Feature B)",
        fontsize=14,
        fontweight="bold",
    )

    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(
            r["precision"] * 100 + 0.5,
            i,
            f"lift={r['lift']:.1f}x  {r['n_hits']}/{r['n_signals']} signals  recall={r['recall']:.0%}",
            va="center",
            fontsize=6,
        )

    plt.tight_layout()
    plt.savefig(OUT / f"joint_thresholds_{_safe(task)}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_regime_comparison(
    regime_data: Dict,
    feat_names: List[str],
    task: str,
    top_k: int = 25,
) -> None:
    if "high_vol" not in regime_data or "low_vol" not in regime_data:
        return

    high = regime_data["high_vol"]["perm_imp"]
    low = regime_data["low_vol"]["perm_imp"]
    diff = high - low
    combined = np.abs(high) + np.abs(low)
    idx = np.argsort(combined)[::-1][:top_k]

    names = [feat_names[i] for i in idx]
    groups = [feature_group(n) for n in names]
    colors = [PALETTE.get(g, "#95a5a6") for g in groups]
    y_pos = np.arange(len(names))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 10))
    fig.suptitle(f"Regime-Conditional Importance — {task}", fontsize=14, fontweight="bold")

    ax1.barh(y_pos, high[idx], color=colors, alpha=0.8, edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{n}  [{g}]" for n, g in zip(names, groups)], fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel("Permutation Importance")
    ax1.set_title(f"HIGH Vol ({regime_data['high_vol']['n_events']} events)")
    ax1.axvline(0, color="grey", linewidth=0.5)

    ax2.barh(y_pos, low[idx], color=colors, alpha=0.8, edgecolor="white")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Permutation Importance")
    ax2.set_title(f"LOW Vol ({regime_data['low_vol']['n_events']} events)")
    ax2.axvline(0, color="grey", linewidth=0.5)

    diff_colors = ["#e74c3c" if d > 0 else "#3498db" for d in diff[idx]]
    ax3.barh(y_pos, diff[idx], color=diff_colors, alpha=0.7, edgecolor="white")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([])
    ax3.invert_yaxis()
    ax3.set_xlabel("Shift (High - Low Vol)")
    ax3.set_title("Regime Shift\nRed = more important in high vol")
    ax3.axvline(0, color="grey", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUT / f"regime_comparison_{_safe(task)}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_dependence_v2(
    shap_values: np.ndarray,
    X: np.ndarray,
    feat_names: List[str],
    task: str,
    top_k: int = 9,
) -> None:
    if shap_values is None:
        return

    mas = np.mean(np.abs(shap_values), axis=0)
    idx = np.argsort(mas)[::-1][:top_k]

    ncols = 3
    nrows = (top_k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    fig.suptitle(
        f"SHAP Dependence with Binned Mean — {task}",
        fontsize=14,
        fontweight="bold",
    )

    for ax_i, feat_i in enumerate(idx):
        r, c = ax_i // ncols, ax_i % ncols
        ax = axes[r][c] if nrows > 1 else axes[c]

        sv = shap_values[:, feat_i]
        xv = X[:, feat_i]

        ax.scatter(xv, sv, c=sv, cmap="coolwarm", s=3, alpha=0.3, rasterized=True)

        n_bins = 20
        valid = ~(np.isnan(xv) | np.isnan(sv))
        if valid.sum() > n_bins * 3:
            bin_edges = np.percentile(xv[valid], np.linspace(0, 100, n_bins + 1))
            bin_centers = []
            bin_means = []
            for b in range(n_bins):
                if b < n_bins - 1:
                    mask = (xv >= bin_edges[b]) & (xv < bin_edges[b + 1])
                else:
                    mask = (xv >= bin_edges[b]) & (xv <= bin_edges[b + 1])
                if mask.sum() > 2:
                    bin_centers.append(np.mean(xv[mask]))
                    bin_means.append(np.mean(sv[mask]))
            ax.plot(bin_centers, bin_means, "k-", linewidth=2.5, alpha=0.9)

        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xlabel(feat_names[feat_i], fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(
            f"{feat_names[feat_i]}  [{feature_group(feat_names[feat_i])}]",
            fontsize=9,
            fontweight="bold",
        )

    for ax_i in range(top_k, nrows * ncols):
        r, c = ax_i // ncols, ax_i % ncols
        (axes[r][c] if nrows > 1 else axes[c]).set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / f"shap_dependence_v2_{_safe(task)}.png", dpi=150, bbox_inches="tight")
    plt.close()


# =================================================================
# CONSOLE REPORTS
# =================================================================
def print_unbiased_importance(
    perm_imp: np.ndarray,
    shap_imp: np.ndarray,
    feat_names: List[str],
    task: str,
    top_k: int = 25,
) -> None:
    idx = np.argsort(shap_imp)[::-1][:top_k]

    print(f"\n  {'='*100}")
    print(f"  UNBIASED IMPORTANCE (ALL {len(feat_names)} features) — {task}")
    print(f"  {'='*100}")
    print(
        f"  {'Rank':<5} {'Feature':<35s} {'|SHAP|':>10} {'Perm Imp':>10} "
        f"{'Group':<14}"
    )
    print(f"  {'-'*100}")

    for rank, i in enumerate(idx, 1):
        print(
            f"  {rank:<5} {feat_names[i]:<35s} {shap_imp[i]:>10.6f} {perm_imp[i]:>10.6f} "
            f"{feature_group(feat_names[i]):<14}"
        )

    # Group totals
    print(f"\n  GROUP IMPORTANCE (sum of |SHAP|):")
    group_shap: Dict[str, float] = {}
    for i, nm in enumerate(feat_names):
        g = feature_group(nm)
        group_shap[g] = group_shap.get(g, 0) + shap_imp[i]
    for g, v in sorted(group_shap.items(), key=lambda x: x[1], reverse=True):
        print(f"    {g:<16s}: {v:.6f}")


def print_event_fingerprint(fp: pd.DataFrame, task: str, top_k: int = 20):
    fp_sorted = fp.sort_values("ks_stat", ascending=False).head(top_k)

    print(f"\n  {'='*115}")
    print(f"  EVENT FINGERPRINT — {task}")
    print(f"  {'='*115}")
    print(
        f"  {'Feature':<35s} {'Group':<14s} {'Z-score':>8s} {'KS':>6s} "
        f"{'Q1':>5s} {'Q2':>5s} {'Q3':>5s} {'Q4':>5s} {'Q5':>5s} {'Peak':>8s}"
    )
    print(f"  {'-'*115}")

    for _, r in fp_sorted.iterrows():
        z = r["z_score"]
        arrow = "+" if z > 0.3 else ("-" if z < -0.3 else " ")
        print(
            f"  {r['feature']:<35s} {r['group']:<14s} {arrow}{abs(z):>6.2f} {r['ks_stat']:>6.3f} "
            f"{r['Q1_pct']:>4.0f}% {r['Q2_pct']:>4.0f}% {r['Q3_pct']:>4.0f}% "
            f"{r['Q4_pct']:>4.0f}% {r['Q5_pct']:>4.0f}% "
            f"Q{r['peak_quintile']:.0f}={r['peak_concentration']:.0f}%"
        )


def print_thresholds(thresholds: pd.DataFrame, task: str, base_rate: float):
    print(f"\n  {'='*115}")
    print(f"  VALUE THRESHOLDS — {task}  (base rate: {base_rate:.1%})")
    print(f"  {'='*115}")
    print(
        f"  {'Feature':<32s} {'Threshold':>10s} {'Dir':>4s} {'Pctile':>7s} "
        f"{'Precision':>10s} {'Recall':>8s} {'Events':>10s} {'Lift':>6s}"
    )
    print(f"  {'-'*115}")

    for _, r in thresholds.iterrows():
        lift = r["precision"] / max(base_rate, 1e-10)
        print(
            f"  {r['feature']:<32s} {r['threshold']:>10.4f} {r['direction']:>4s} "
            f"{r['threshold_percentile']:>6.1f}% "
            f"{r['precision']:>9.1%} {r['recall']:>7.1%} "
            f"{r['events_captured']:>3d}/{r['total_events']:<3d}    "
            f"{lift:>5.1f}x"
        )


def print_joint_thresholds(
    jt: pd.DataFrame, task: str, base_rate: float, top_k: int = 15
):
    if jt.empty:
        print(f"\n  No joint threshold rules found for {task}")
        return

    print(f"\n  {'='*130}")
    print(f"  JOINT THRESHOLDS — {task}  (base rate: {base_rate:.1%})")
    print(f"  {'='*130}")

    for i, (_, r) in enumerate(jt.head(top_k).iterrows(), 1):
        rule_a = f"{r['feature_A']} {r['dir_A']} P{r['pctile_A']:.0f}"
        rule_b = f"{r['feature_B']} {r['dir_B']} P{r['pctile_B']:.0f}"
        print(
            f"  #{i:<3d} {rule_a:<38s} AND {rule_b:<38s} "
            f"prec={r['precision']:.1%}  recall={r['recall']:.1%}  "
            f"lift={r['lift']:.1f}x  ({r['n_hits']}/{r['n_signals']} signals)"
        )


def print_regime_comparison(
    regime_data: Dict, feat_names: List[str], task: str, top_k: int = 15
):
    if "high_vol" not in regime_data or "low_vol" not in regime_data:
        print(f"\n  Regime analysis: insufficient data for {task}")
        return

    high = regime_data["high_vol"]
    low = regime_data["low_vol"]
    diff = high["perm_imp"] - low["perm_imp"]
    idx = np.argsort(np.abs(diff))[::-1][:top_k]

    print(f"\n  {'='*105}")
    print(f"  REGIME-CONDITIONAL IMPORTANCE — {task}")
    print(
        f"  Split on volatility_20d median.  "
        f"High-vol: {high['n_events']} events ({high['event_rate']:.1%}),  "
        f"Low-vol: {low['n_events']} events ({low['event_rate']:.1%})"
    )
    print(f"  {'='*105}")
    print(
        f"  {'Feature':<35s} {'Group':<14s} {'High Vol':>9s} {'Low Vol':>9s} "
        f"{'Shift':>9s} {'Note':<20s}"
    )
    print(f"  {'-'*105}")

    for i in idx:
        h = high["perm_imp"][i]
        l = low["perm_imp"][i]
        d = diff[i]
        note = ""
        if d > 0.01:
            note = "HIGH-VOL ONLY"
        elif d < -0.01:
            note = "LOW-VOL ONLY"
        elif h > 0.005 and l > 0.005:
            note = "regime-stable"
        print(
            f"  {feat_names[i]:<35s} {feature_group(feat_names[i]):<14s} "
            f"{h:>9.4f} {l:>9.4f} {d:>+9.4f} {note:<20s}"
        )


# =================================================================
# MAIN
# =================================================================
def main():
    t0 = datetime.now()
    print("=" * 80)
    print("INVESTIGATE v2 — Unbiased Feature Impact & Thresholds")
    print("=" * 80)
    print(f"Start: {t0:%Y-%m-%d %H:%M:%S}")
    print(f"SHAP available: {HAS_SHAP}")
    print(f"Output: {OUT}/\n")

    config = Config()

    # ── 1. Data ──────────────────────────────────────────────────
    print("[1/6] Loading data …")
    prices, returns = load_data(config)

    # ── 2. Features ──────────────────────────────────────────────
    print("\n[2/6] Building features …")
    spectral = build_spectral_features(returns, config)
    traditional = build_traditional_features(prices, returns)
    combined = pd.concat([spectral, traditional], axis=1)
    feat_names = list(combined.columns)
    n_feat = len(feat_names)
    print(f"  Combined: {n_feat} features")

    # ── 3. Targets ───────────────────────────────────────────────
    print("\n[3/6] Computing targets …")
    market = prices["SP500"] if "SP500" in prices.columns else prices.iloc[:, 0]
    fut_3d = market.pct_change(3).shift(-3)

    tasks = {
        "Large UP move 3d (>3%)": (fut_3d > 0.03).astype(int),
        "Large DOWN move 3d (>3%)": (fut_3d < -0.03).astype(int),
    }
    for name, tgt in tasks.items():
        print(
            f"  {name}: {tgt.dropna().mean():.1%} positive "
            f"({int(tgt.dropna().sum())} events)"
        )

    summary = {}

    for task_name, target in tasks.items():
        print(f"\n{'='*80}")
        print(f"  TASK: {task_name}")
        print(f"{'='*80}")

        valid_dates = target.dropna().index
        X = combined.reindex(valid_dates).ffill().fillna(0)
        y = target.loc[valid_dates]
        base_rate = float(y.mean())

        train_size = int(config.train_years * 252)
        test_size = int(config.test_months * 21)
        gap = config.gap_days
        schedule = _compute_fold_schedule(
            len(X), train_size, test_size, gap, config.n_splits
        )

        # ── 4. Unbiased importance (last 3 folds) ───────────────
        print("\n  [A] Unbiased importance (SHAP + Permutation on ALL features) …")
        n_use = min(3, len(schedule))
        folds_to_use = schedule[-n_use:]

        all_perm = np.zeros(n_feat)
        all_shap = np.zeros(n_feat)
        all_rf_imp = np.zeros(n_feat)
        shap_concat = []
        X_concat = []
        fold_count = 0
        perm_fold_count = 0  # Track folds with valid permutation importance

        # Also accumulate raw test data across folds for threshold analysis
        pooled_X_te_raw = []
        pooled_y_te_raw = []
        last_X_tr_raw = None  # Keep last train set for threshold fitting

        for fi, (train_end, test_start, test_end) in enumerate(folds_to_use):
            X_tr = np.nan_to_num(X.iloc[:train_end].values, nan=0, posinf=0, neginf=0)
            y_tr = y.iloc[:train_end].values
            X_te = np.nan_to_num(
                X.iloc[test_start:test_end].values, nan=0, posinf=0, neginf=0
            )
            y_te = y.iloc[test_start:test_end].values

            if len(np.unique(y_tr)) < 2:
                continue

            # Pool raw test data for threshold evaluation later
            pooled_X_te_raw.append(X.iloc[test_start:test_end].values)
            pooled_y_te_raw.append(y.iloc[test_start:test_end].values)
            last_X_tr_raw = X.iloc[:train_end].values

            res = unbiased_importance(X_tr, y_tr, X_te, y_te, feat_names, config)

            # Only accumulate permutation importance if it's not NaN
            perm_imp = res["perm_imp"]
            if not np.all(np.isnan(perm_imp)):
                all_perm += np.nan_to_num(perm_imp, nan=0)
                perm_fold_count += 1

            all_rf_imp += res["rf_imp"]
            if res["shap_values"] is not None:
                all_shap += np.mean(np.abs(res["shap_values"]), axis=0)
                shap_concat.append(res["shap_values"])
                X_concat.append(res["shap_X"])
            fold_count += 1

            print(
                f"    Fold {fi}: OOB={res['oob_auc']:.3f}  "
                f"Test={res['test_auc']:.3f}  "
                f"test_events={int(y_te.sum())}"
            )

        if fold_count > 0:
            all_shap /= fold_count
            all_rf_imp /= fold_count
        if perm_fold_count > 0:
            all_perm /= perm_fold_count
        else:
            print("    WARNING: No folds had valid permutation importance (no events in test sets)")

        # Pool test data across folds for robust threshold evaluation
        if pooled_X_te_raw:
            pooled_X_te = np.nan_to_num(
                np.vstack(pooled_X_te_raw), nan=0, posinf=0, neginf=0
            )
            pooled_y_te = np.concatenate(pooled_y_te_raw)
            pooled_X_tr = np.nan_to_num(last_X_tr_raw, nan=0, posinf=0, neginf=0)
            pooled_n_events = int(pooled_y_te.sum())
            print(
                f"    Pooled test data: {len(pooled_y_te)} samples, "
                f"{pooled_n_events} events ({pooled_n_events/len(pooled_y_te):.1%})"
            )

        shap_full = np.vstack(shap_concat) if shap_concat else None
        X_full = np.vstack(X_concat) if X_concat else None

        print_unbiased_importance(all_perm, all_shap, feat_names, task_name)
        print("    Plotting …")
        plot_unbiased_importance(all_perm, all_shap, feat_names, task_name)
        if shap_full is not None:
            plot_shap_dependence_v2(shap_full, X_full, feat_names, task_name)

        # ── 5. Event fingerprinting ──────────────────────────────
        print("\n  [B] Event fingerprinting …")
        fp = event_fingerprinting(combined, target, feat_names)
        print_event_fingerprint(fp, task_name)
        plot_event_fingerprint(fp, task_name)

        # ── 6. Value thresholds (pooled across folds) ─────────────
        print("\n  [C] Value thresholds (pooled test data) …")
        thresh_df = value_thresholds(
            pooled_X_tr,
            y.iloc[: len(pooled_X_tr)].values,
            pooled_X_te,
            pooled_y_te,
            feat_names,
            top_k=25,
            feature_ranking=all_shap,
        )
        print_thresholds(thresh_df, task_name, base_rate)
        plot_threshold_analysis(thresh_df, task_name)

        # ── 7. Joint thresholds (pooled across folds) ────────────
        print("\n  [D] Joint thresholds (pooled test data) …")
        jt = joint_thresholds(
            pooled_X_te, pooled_y_te, feat_names, all_shap, pooled_X_tr,
            top_k=10, n_best=30,
        )
        print_joint_thresholds(jt, task_name, base_rate)
        if not jt.empty:
            plot_joint_thresholds(jt, task_name)

        # ── 8. Regime-conditional ────────────────────────────────
        print("\n  [E] Regime-conditional importance …")
        regime_data = regime_conditional_importance(
            combined, target, feat_names, config
        )
        print_regime_comparison(regime_data, feat_names, task_name)
        plot_regime_comparison(regime_data, feat_names, task_name)

        # ── Store ────────────────────────────────────────────────
        top_idx = np.argsort(all_shap)[::-1][:30]
        summary[task_name] = {
            "base_rate": base_rate,
            "top_30_features": [
                {
                    "name": feat_names[i],
                    "group": feature_group(feat_names[i]),
                    "shap_importance": float(all_shap[i]),
                    "perm_importance": float(all_perm[i]),
                }
                for i in top_idx
            ],
            "event_fingerprint_top10": (
                fp.sort_values("ks_stat", ascending=False)
                .head(10)
                .to_dict("records")
            ),
            "thresholds_top10": thresh_df.head(10).to_dict("records"),
            "joint_thresholds_top5": (
                jt.head(5).to_dict("records") if not jt.empty else []
            ),
        }

    # ── Save ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  Saving results …")
    with open(OUT / "summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  All outputs saved to {OUT}/")
    print(f"  Runtime: {datetime.now() - t0}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
