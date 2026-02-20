#!/usr/bin/env python3
"""
investigate.py — Deep SHAP & feature diagnostics for CGECD
===========================================================

For each task (UP / DOWN) independently:
  1. SHAP values via TreeExplainer on the RF component
  2. Vol-LR coefficient analysis  &  per-fold α routing
  3. Feature-group SHAP breakdown
  4. MI selection frequency across folds
  5. SHAP-directed effect (which direction each feature pushes)
  6. Feature-count sweep  (AUC vs top-k SHAP features)
  7. Exhaustive group-combination search  (2^G combos)
  8. Cross-task comparison (unique UP / DOWN drivers)

Run:
    pip install shap          # if not already installed
    python investigate.py

Output → results/investigate/
"""

import warnings

warnings.filterwarnings("ignore")

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

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
    print("⚠  shap not installed — falling back to RF importances only.")
    print("   Install with:  pip install shap")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from config import Config
from algorithm import (
    load_data,
    build_spectral_features,
    build_traditional_features,
    _VOL_FEATURES,
    _compute_fold_schedule,
)
from metrics import compute_metrics


# ── Output ───────────────────────────────────────────────────────
OUT = Path("results/investigate")
OUT.mkdir(parents=True, exist_ok=True)


# ── Feature grouping (mirrors visualizations.py) ────────────────
def feature_group(name: str) -> str:
    n = name.lower()
    if "_zscore_" in n or "_roc_" in n or "_accel_" in n:
        return "Dynamics"
    if "_ratio" in n and any(
        k in n
        for k in (
            "lambda",
            "absorption",
            "effective_rank",
            "edge_density",
            "mean_abs_corr",
        )
    ):
        return "Eigenvalue"
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
        k in n
        for k in ("credit", "flight", "breadth", "dispersion", "em_stress")
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
# FOLD DIAGNOSTICS
# =================================================================
@dataclass
class FoldInfo:
    fold_idx: int = 0
    train_end: int = 0
    test_start: int = 0
    test_end: int = 0
    rf_oob_auc: float = 0.5
    vol_lr_cv_auc: float = 0.5
    alpha: float = 0.0
    selected_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    shap_values: Optional[np.ndarray] = None
    X_test_selected: Optional[np.ndarray] = None
    vol_lr_coefs: Optional[np.ndarray] = None
    vol_feature_idx: List[int] = field(default_factory=list)
    rf_importances_full: Optional[np.ndarray] = None
    mi_scores: Optional[np.ndarray] = None
    test_probs: Optional[np.ndarray] = None
    test_actuals: Optional[np.ndarray] = None


# =================================================================
# DIAGNOSTIC WALK-FORWARD
# =================================================================
def diagnostic_walk_forward(
    features: pd.DataFrame,
    target: pd.Series,
    config: Config,
    task_label: str = "",
) -> Tuple[List[FoldInfo], List[str], np.ndarray, np.ndarray]:
    """Walk-forward capturing per-fold models, SHAP, α, coefficients."""

    print(f"\n  ── {task_label} ──")
    valid_dates = target.dropna().index
    X = features.reindex(valid_dates).ffill().fillna(0)
    y = target.loc[valid_dates]
    feat_names = list(X.columns)
    n_feat = len(feat_names)

    train_size = int(config.train_years * 252)
    test_size = int(config.test_months * 21)
    gap = config.gap_days

    schedule = _compute_fold_schedule(
        len(X), train_size, test_size, gap, config.n_splits
    )
    folds: List[FoldInfo] = []
    all_probs: List[float] = []
    all_actuals: List[int] = []

    for fold_idx, (train_end, test_start, test_end) in enumerate(schedule):
        fi = FoldInfo(
            fold_idx=fold_idx,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        X_tr = np.nan_to_num(
            X.iloc[:train_end].values, nan=0, posinf=0, neginf=0
        )
        y_tr = y.iloc[:train_end].values
        X_te = np.nan_to_num(
            X.iloc[test_start:test_end].values, nan=0, posinf=0, neginf=0
        )
        y_te = y.iloc[test_start:test_end].values

        if len(np.unique(y_tr)) < 2:
            continue

        try:
            scaler = RobustScaler()
            Xs_tr = scaler.fit_transform(X_tr)
            Xs_te = scaler.transform(X_te)

            # MI selection
            k = min(config.feature_selection_k, Xs_tr.shape[1])
            mi = mutual_info_classif(
                Xs_tr, y_tr, random_state=config.random_seed, n_neighbors=5
            )
            mi = np.nan_to_num(mi, nan=0.0)
            sel = np.argsort(mi)[-k:]
            fi.selected_idx = sel
            fi.mi_scores = mi.copy()

            Xk_tr = Xs_tr[:, sel]
            Xk_te = Xs_te[:, sel]

            # Vol features
            vol_idx = [
                i for i, nm in enumerate(feat_names) if nm in _VOL_FEATURES
            ]
            fi.vol_feature_idx = vol_idx
            has_vol = len(vol_idx) >= 3
            if has_vol:
                Xv_check = Xs_tr[:, vol_idx]
                if np.all(np.std(Xv_check, axis=0) < 1e-10):
                    has_vol = False

            # ── Train RF (with OOB) ────────────────────────────
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
            rf.fit(Xk_tr, y_tr)

            imp_full = np.zeros(n_feat)
            imp_full[sel] = rf.feature_importances_
            fi.rf_importances_full = imp_full

            try:
                fi.rf_oob_auc = roc_auc_score(
                    y_tr, rf.oob_decision_function_[:, 1]
                )
            except Exception:
                fi.rf_oob_auc = 0.5

            # ── SHAP ───────────────────────────────────────────
            if HAS_SHAP:
                explainer = shap.TreeExplainer(rf)
                sv = explainer.shap_values(Xk_te)
                if isinstance(sv, list):
                    sv = sv[1]  # positive class
                elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                    sv = sv[:, :, 1]  # (n_samples, n_features, n_classes) → positive class
                fi.shap_values = sv
                fi.X_test_selected = Xk_te

            # ── Vol-LR ─────────────────────────────────────────
            fi.alpha = 0.0
            pred_rf = rf.predict_proba(Xk_te)[:, 1]

            if has_vol:
                Xv_tr = Xs_tr[:, vol_idx]
                Xv_te = Xs_te[:, vol_idx]

                vol_auc = 0.5
                try:
                    tscv = TimeSeriesSplit(n_splits=3)
                    cv_p = np.full(len(y_tr), np.nan)
                    for tr_i, te_i in tscv.split(Xv_tr):
                        if len(np.unique(y_tr[tr_i])) < 2:
                            continue
                        lr_t = LogisticRegression(
                            penalty="l1",
                            solver="saga",
                            C=0.5,
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=config.random_seed,
                        )
                        lr_t.fit(Xv_tr[tr_i], y_tr[tr_i])
                        cv_p[te_i] = lr_t.predict_proba(Xv_tr[te_i])[:, 1]
                    valid = ~np.isnan(cv_p)
                    if (
                        np.sum(valid) > 20
                        and len(np.unique(y_tr[valid])) >= 2
                    ):
                        vol_auc = roc_auc_score(y_tr[valid], cv_p[valid])
                except Exception:
                    vol_auc = 0.5

                fi.vol_lr_cv_auc = vol_auc
                diff = (
                    vol_auc - fi.rf_oob_auc - config.vol_correction_threshold
                )
                alpha = min(
                    max(0.0, diff * config.vol_correction_scale),
                    config.vol_correction_max,
                )
                fi.alpha = alpha

                vol_lr = LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=0.5,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=config.random_seed,
                )
                vol_lr.fit(Xv_tr, y_tr)
                fi.vol_lr_coefs = vol_lr.coef_[0].copy()

                pred_vol = vol_lr.predict_proba(Xv_te)[:, 1]
                fi.test_probs = (1.0 - alpha) * pred_rf + alpha * pred_vol
            else:
                fi.test_probs = pred_rf

            fi.test_actuals = y_te
            all_probs.extend(fi.test_probs.tolist())
            all_actuals.extend(y_te.tolist())
            folds.append(fi)

            print(
                f"    Fold {fold_idx:2d}  RF_OOB={fi.rf_oob_auc:.3f}  "
                f"Vol_CV={fi.vol_lr_cv_auc:.3f}  α={fi.alpha:.3f}  "
                f"n_test={len(y_te)}"
            )

        except Exception as e:
            print(f"    Fold {fold_idx} FAILED: {e}")

    return folds, feat_names, np.array(all_probs), np.array(all_actuals)


# =================================================================
# SHAP AGGREGATION
# =================================================================
def aggregate_shap(
    folds: List[FoldInfo], n_features: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Map per-fold SHAP to full feature space.

    Returns
        shap_full  (N_test_total, n_features)
        X_full     (N_test_total, n_features)   — 0 where not selected
    """
    shap_rows, x_rows = [], []
    for fi in folds:
        if fi.shap_values is None:
            continue
        n_test = fi.shap_values.shape[0]
        s = np.zeros((n_test, n_features))
        x = np.zeros((n_test, n_features))
        s[:, fi.selected_idx] = fi.shap_values
        if fi.X_test_selected is not None:
            x[:, fi.selected_idx] = fi.X_test_selected
        shap_rows.append(s)
        x_rows.append(x)

    if not shap_rows:
        return np.zeros((1, n_features)), np.zeros((1, n_features))
    return np.vstack(shap_rows), np.vstack(x_rows)


def mean_abs_shap(shap_full: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(shap_full), axis=0)


# =================================================================
# PLOTS
# =================================================================
def plot_shap_bar(
    mas: np.ndarray, feat_names: List[str], task: str, top_k: int = 30
) -> None:
    idx = np.argsort(mas)[::-1][:top_k]
    names = [feat_names[i] for i in idx]
    vals = mas[idx]
    groups = [feature_group(n) for n in names]
    colors = [PALETTE.get(g, "#95a5a6") for g in groups]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(
        range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.5
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(
        f"SHAP Feature Importance — {task}", fontsize=14, fontweight="bold"
    )

    seen = set()
    handles = []
    for g in groups:
        if g not in seen:
            seen.add(g)
            handles.append(
                Rectangle((0, 0), 1, 1, fc=PALETTE.get(g, "#95a5a6"), label=g)
            )
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / f"shap_bar_{_safe(task)}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved shap_bar_{_safe(task)}.png")


def plot_shap_beeswarm(
    shap_full: np.ndarray,
    X_full: np.ndarray,
    feat_names: List[str],
    task: str,
    top_k: int = 25,
) -> None:
    mas = mean_abs_shap(shap_full)
    idx = np.argsort(mas)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(12, 10))
    rng = np.random.RandomState(42)

    for row, feat_i in enumerate(idx):
        sv = shap_full[:, feat_i]
        xv = X_full[:, feat_i]

        xmin, xmax = np.percentile(xv, [2, 98])
        if xmax - xmin < 1e-10:
            xnorm = np.full_like(xv, 0.5)
        else:
            xnorm = np.clip((xv - xmin) / (xmax - xmin), 0, 1)

        jitter = rng.uniform(-0.35, 0.35, len(sv))
        ax.scatter(
            sv,
            row + jitter,
            c=xnorm,
            cmap="coolwarm",
            s=3,
            alpha=0.5,
            rasterized=True,
        )

    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feat_names[i] for i in idx], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
    ax.set_title(f"SHAP Beeswarm — {task}", fontsize=14, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Feature value (normalized)", fontsize=9)

    plt.tight_layout()
    plt.savefig(
        OUT / f"shap_beeswarm_{_safe(task)}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"      Saved shap_beeswarm_{_safe(task)}.png")


def plot_shap_by_group(
    mas: np.ndarray, feat_names: List[str], task: str
) -> None:
    group_shap: Dict[str, float] = {}
    group_count: Dict[str, int] = {}
    for i, name in enumerate(feat_names):
        g = feature_group(name)
        group_shap[g] = group_shap.get(g, 0) + mas[i]
        group_count[g] = group_count.get(g, 0) + 1

    sg = sorted(group_shap.items(), key=lambda x: x[1], reverse=True)
    groups = [g for g, _ in sg]
    vals = [v for _, v in sg]
    counts = [group_count[g] for g in groups]
    colors = [PALETTE.get(g, "#95a5a6") for g in groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"SHAP by Feature Group — {task}", fontsize=14, fontweight="bold"
    )

    bars = ax1.barh(range(len(groups)), vals, color=colors, edgecolor="white")
    ax1.set_yticks(range(len(groups)))
    ax1.set_yticklabels(groups, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Sum of mean |SHAP|")
    ax1.set_title("Total Group Importance")
    for bar, v, c in zip(bars, vals, counts):
        ax1.text(
            bar.get_width() + 0.0005,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f} ({c} feat)",
            va="center",
            fontsize=8,
        )

    avg = [v / max(c, 1) for v, c in zip(vals, counts)]
    ax2.barh(range(len(groups)), avg, color=colors, edgecolor="white")
    ax2.set_yticks(range(len(groups)))
    ax2.set_yticklabels(groups, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean |SHAP| per feature in group")
    ax2.set_title("Average Per-Feature Importance")

    plt.tight_layout()
    plt.savefig(
        OUT / f"shap_groups_{_safe(task)}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"      Saved shap_groups_{_safe(task)}.png")


def plot_shap_dependence(
    shap_full: np.ndarray,
    X_full: np.ndarray,
    feat_names: List[str],
    task: str,
    top_k: int = 6,
) -> None:
    mas = mean_abs_shap(shap_full)
    idx = np.argsort(mas)[::-1][:top_k]

    rows_plot = (top_k + 2) // 3
    fig, axes = plt.subplots(rows_plot, 3, figsize=(18, 5 * rows_plot))
    fig.suptitle(
        f"SHAP Dependence — {task}", fontsize=14, fontweight="bold"
    )
    if rows_plot == 1:
        axes = [axes]

    for ax_i, feat_i in enumerate(idx):
        ax = axes[ax_i // 3][ax_i % 3] if rows_plot > 1 else axes[0][ax_i % 3]
        sv = shap_full[:, feat_i]
        xv = X_full[:, feat_i]
        ax.scatter(
            xv, sv, c=sv, cmap="coolwarm", s=3, alpha=0.5, rasterized=True
        )
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xlabel(feat_names[feat_i], fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(feat_names[feat_i], fontsize=10, fontweight="bold")

    # hide unused axes
    for ax_i in range(top_k, rows_plot * 3):
        r, c = ax_i // 3, ax_i % 3
        if rows_plot > 1:
            axes[r][c].set_visible(False)
        else:
            axes[0][c].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        OUT / f"shap_dependence_{_safe(task)}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"      Saved shap_dependence_{_safe(task)}.png")


def plot_shap_direction(
    shap_full: np.ndarray,
    X_full: np.ndarray,
    feat_names: List[str],
    task: str,
    top_k: int = 25,
) -> np.ndarray:
    """Directional effect: correlation(feature_value, SHAP_value).

    Positive → higher feature → higher crisis probability.
    Negative → higher feature → lower crisis probability.
    """
    mas = mean_abs_shap(shap_full)
    idx = np.argsort(mas)[::-1][:top_k]

    directions = np.zeros(len(feat_names))
    for i in range(len(feat_names)):
        sv = shap_full[:, i]
        xv = X_full[:, i]
        if np.std(sv) > 1e-12 and np.std(xv) > 1e-12:
            directions[i] = np.corrcoef(xv, sv)[0, 1]

    fig, ax = plt.subplots(figsize=(12, 10))
    names = [feat_names[i] for i in idx]
    vals = directions[idx]
    groups = [feature_group(n) for n in names]
    colors_bar = ["#e74c3c" if v > 0 else "#3498db" for v in vals]

    ax.barh(range(len(names)), vals, color=colors_bar, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(
        [f"{n}  [{g}]" for n, g in zip(names, groups)], fontsize=7
    )
    ax.invert_yaxis()
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel(
        "Corr(feature_value, SHAP_value)\n"
        "← Higher value lowers prediction     Higher value raises prediction →",
        fontsize=10,
    )
    ax.set_title(
        f"SHAP Effect Direction — {task}", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(
        OUT / f"shap_direction_{_safe(task)}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"      Saved shap_direction_{_safe(task)}.png")
    return directions


def plot_vol_lr_analysis(
    folds: List[FoldInfo], feat_names: List[str], task: str
) -> None:
    vol_idx = folds[0].vol_feature_idx if folds else []
    vol_names = [feat_names[i] for i in vol_idx]

    coef_matrix = []
    alphas = []
    rf_aucs = []
    vol_aucs = []
    for fi in folds:
        alphas.append(fi.alpha)
        rf_aucs.append(fi.rf_oob_auc)
        vol_aucs.append(fi.vol_lr_cv_auc)
        if fi.vol_lr_coefs is not None:
            coef_matrix.append(fi.vol_lr_coefs)

    if not coef_matrix:
        print("      No Vol-LR coefficients to plot.")
        return

    coefs = np.array(coef_matrix)
    mean_c = np.mean(coefs, axis=0)
    std_c = np.std(coefs, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"Vol-LR & Ensemble Routing — {task}", fontsize=14, fontweight="bold"
    )

    # 1 — coefficients
    ax = axes[0, 0]
    order = np.argsort(np.abs(mean_c))[::-1]
    n_sorted = [vol_names[i] for i in order]
    c_sorted = mean_c[order]
    s_sorted = std_c[order]
    cbar = ["#e74c3c" if c > 0 else "#3498db" for c in c_sorted]
    ax.barh(
        range(len(n_sorted)),
        c_sorted,
        xerr=s_sorted,
        color=cbar,
        edgecolor="white",
        capsize=3,
    )
    ax.set_yticks(range(len(n_sorted)))
    ax.set_yticklabels(n_sorted, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("Coefficient (mean ± std)")
    ax.set_title("Vol-LR Coefficients")

    # 2 — α per fold
    ax = axes[0, 1]
    ax.bar(range(len(alphas)), alphas, color="#e74c3c", alpha=0.7)
    ax.set_xlabel("Fold")
    ax.set_ylabel("α (Vol-LR weight)")
    ax.set_title(f"Routing Weight  (mean α = {np.mean(alphas):.3f})")
    ax.axhline(0, color="grey", linewidth=0.5)

    # 3 — RF vs Vol-LR AUC
    ax = axes[1, 0]
    x = range(len(rf_aucs))
    ax.plot(x, rf_aucs, "o-", color="#2ecc71", label="RF (OOB)", linewidth=2)
    ax.plot(x, vol_aucs, "s-", color="#e74c3c", label="Vol-LR (CV)", linewidth=2)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC")
    ax.set_title("Per-Fold AUC: RF vs Vol-LR")
    ax.legend()

    # 4 — coefficient heatmap
    ax = axes[1, 1]
    if coefs.shape[0] > 1:
        vmax = np.percentile(np.abs(coefs), 95) or 1.0
        im = ax.imshow(
            coefs[:, order].T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_yticks(range(len(n_sorted)))
        ax.set_yticklabels(n_sorted, fontsize=7)
        ax.set_xlabel("Fold")
        ax.set_title("Coefficient Stability")
        plt.colorbar(im, ax=ax, shrink=0.8)
    else:
        ax.text(0.5, 0.5, "Single fold", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(
        OUT / f"vol_lr_analysis_{_safe(task)}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"      Saved vol_lr_analysis_{_safe(task)}.png")


def plot_selection_frequency(
    folds: List[FoldInfo], feat_names: List[str], task: str
) -> np.ndarray:
    counts = np.zeros(len(feat_names))
    for fi in folds:
        counts[fi.selected_idx] += 1
    freq = counts / max(len(folds), 1)

    idx = np.argsort(freq)[::-1][:40]
    names = [feat_names[i] for i in idx]
    vals = freq[idx]
    groups = [feature_group(n) for n in names]
    colors = [PALETTE.get(g, "#95a5a6") for g in groups]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(
        range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.5
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Selection Frequency (fraction of folds)", fontsize=11)
    ax.set_title(
        f"MI Selection Frequency — {task}", fontsize=14, fontweight="bold"
    )
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(
        OUT / f"selection_freq_{_safe(task)}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"      Saved selection_freq_{_safe(task)}.png")
    return freq


def plot_mi_vs_shap(
    folds: List[FoldInfo],
    mas: np.ndarray,
    feat_names: List[str],
    task: str,
) -> None:
    """Compare MI ranking with SHAP ranking."""
    mi_avg = np.zeros(len(feat_names))
    cnt = 0
    for fi in folds:
        if fi.mi_scores is not None:
            mi_avg += fi.mi_scores
            cnt += 1
    if cnt > 0:
        mi_avg /= cnt

    fig, ax = plt.subplots(figsize=(10, 8))
    groups = [feature_group(n) for n in feat_names]
    for g in set(groups):
        mask = [i for i, gg in enumerate(groups) if gg == g]
        ax.scatter(
            mi_avg[mask],
            mas[mask],
            c=PALETTE.get(g, "#95a5a6"),
            label=g,
            alpha=0.6,
            s=30,
        )

    ax.set_xlabel("Mutual Information (avg across folds)", fontsize=11)
    ax.set_ylabel("Mean |SHAP|", fontsize=11)
    ax.set_title(f"MI vs SHAP — {task}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # annotate top-5 by SHAP
    top5 = np.argsort(mas)[::-1][:5]
    for i in top5:
        ax.annotate(
            feat_names[i],
            (mi_avg[i], mas[i]),
            fontsize=6,
            alpha=0.8,
            textcoords="offset points",
            xytext=(5, 5),
        )

    plt.tight_layout()
    plt.savefig(
        OUT / f"mi_vs_shap_{_safe(task)}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"      Saved mi_vs_shap_{_safe(task)}.png")


# =================================================================
# FEATURE SELECTION SWEEP
# =================================================================
def feature_selection_sweep(
    features: pd.DataFrame,
    target: pd.Series,
    feat_names: List[str],
    shap_ranking: np.ndarray,
    config: Config,
    task: str,
    ks: Optional[List[int]] = None,
) -> Dict[int, float]:
    """AUC for top-k SHAP features (last fold for speed)."""

    valid_dates = target.dropna().index
    X = features.reindex(valid_dates).ffill().fillna(0)
    y = target.loc[valid_dates]

    train_size = int(config.train_years * 252)
    test_size = int(config.test_months * 21)
    gap = config.gap_days

    train_end = len(X) - gap - test_size
    test_start = train_end + gap
    test_end = len(X)
    if train_end < train_size:
        train_end = train_size

    X_tr = np.nan_to_num(
        X.iloc[:train_end].values, nan=0, posinf=0, neginf=0
    )
    y_tr = y.iloc[:train_end].values
    X_te = np.nan_to_num(
        X.iloc[test_start:test_end].values, nan=0, posinf=0, neginf=0
    )
    y_te = y.iloc[test_start:test_end].values

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return {}

    scaler = RobustScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_te = scaler.transform(X_te)

    ranked = np.argsort(shap_ranking)[::-1]

    if ks is None:
        ks = sorted(
            set(
                [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80]
                + [len(feat_names)]
            )
        )
        ks = [k for k in ks if k <= len(feat_names)]

    results: Dict[int, float] = {}
    for k in ks:
        sel = ranked[:k]
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            min_samples_split=config.rf_min_samples_split,
            class_weight="balanced_subsample",
            random_state=config.random_seed,
            n_jobs=-1,
        )
        rf.fit(Xs_tr[:, sel], y_tr)
        probs = rf.predict_proba(Xs_te[:, sel])[:, 1]
        try:
            auc = roc_auc_score(y_te, probs)
        except Exception:
            auc = 0.5
        results[k] = auc
    return results


def plot_selection_sweep(
    sweep_up: Dict[int, float], sweep_down: Dict[int, float]
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if sweep_up:
        ks = sorted(sweep_up.keys())
        ax.plot(
            ks,
            [sweep_up[k] for k in ks],
            "o-",
            color="#2ecc71",
            label="UP task",
            linewidth=2,
            markersize=5,
        )
        best_k = max(sweep_up, key=sweep_up.get)
        ax.axvline(best_k, color="#2ecc71", linestyle=":", alpha=0.5)
    if sweep_down:
        ks = sorted(sweep_down.keys())
        ax.plot(
            ks,
            [sweep_down[k] for k in ks],
            "s-",
            color="#e74c3c",
            label="DOWN task",
            linewidth=2,
            markersize=5,
        )
        best_k = max(sweep_down, key=sweep_down.get)
        ax.axvline(best_k, color="#e74c3c", linestyle=":", alpha=0.5)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Number of Top SHAP Features", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title(
        "Feature Count Sweep — AUC vs # Features",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        OUT / "feature_selection_sweep.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"    Saved feature_selection_sweep.png")


# =================================================================
# GROUP COMBINATION SEARCH
# =================================================================
def group_combination_search(
    features: pd.DataFrame,
    target: pd.Series,
    feat_names: List[str],
    config: Config,
    task: str,
) -> pd.DataFrame:
    """Exhaustive search over 2^G group combinations."""

    valid_dates = target.dropna().index
    X = features.reindex(valid_dates).ffill().fillna(0)
    y = target.loc[valid_dates]

    train_size = int(config.train_years * 252)
    test_size = int(config.test_months * 21)
    gap = config.gap_days

    train_end = len(X) - gap - test_size
    test_start = train_end + gap
    test_end = len(X)
    if train_end < train_size:
        train_end = train_size

    X_tr = np.nan_to_num(
        X.iloc[:train_end].values, nan=0, posinf=0, neginf=0
    )
    y_tr = y.iloc[:train_end].values
    X_te = np.nan_to_num(
        X.iloc[test_start:test_end].values, nan=0, posinf=0, neginf=0
    )
    y_te = y.iloc[test_start:test_end].values

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return pd.DataFrame()

    scaler = RobustScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_te = scaler.transform(X_te)

    fg = [feature_group(n) for n in feat_names]
    unique_g = sorted(set(fg))
    g_idx = {g: [i for i, gg in enumerate(fg) if gg == g] for g in unique_g}
    n_g = len(unique_g)

    print(f"    Searching {2**n_g - 1} group combinations …")
    rows = []
    for mask in range(1, 2**n_g):
        sel_groups = [unique_g[i] for i in range(n_g) if mask & (1 << i)]
        sel = []
        for g in sel_groups:
            sel.extend(g_idx[g])
        sel = sorted(sel)
        if len(sel) < 2:
            continue
        try:
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=config.rf_max_depth,
                min_samples_leaf=config.rf_min_samples_leaf,
                min_samples_split=config.rf_min_samples_split,
                class_weight="balanced_subsample",
                random_state=config.random_seed,
                n_jobs=-1,
            )
            rf.fit(Xs_tr[:, sel], y_tr)
            probs = rf.predict_proba(Xs_te[:, sel])[:, 1]
            auc = roc_auc_score(y_te, probs)
        except Exception:
            auc = 0.5

        rows.append(
            {
                "groups": " + ".join(sel_groups),
                "n_groups": len(sel_groups),
                "n_features": len(sel),
                "auc": auc,
            }
        )

    return (
        pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    )


def plot_group_combinations(gc_up: pd.DataFrame, gc_down: pd.DataFrame) -> None:
    """Top-15 group combos side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
    fig.suptitle(
        "Best Feature-Group Combinations", fontsize=14, fontweight="bold"
    )

    for ax, gc, title, color in [
        (ax1, gc_up, "UP Task", "#2ecc71"),
        (ax2, gc_down, "DOWN Task", "#e74c3c"),
    ]:
        if gc.empty:
            ax.text(0.5, 0.5, "No data", ha="center")
            continue
        top = gc.head(15)
        ax.barh(
            range(len(top)),
            top["auc"].values,
            color=color,
            alpha=0.7,
            edgecolor="white",
        )
        ax.set_yticks(range(len(top)))
        labels = [
            f"{r['groups']}  ({r['n_features']}f)"
            for _, r in top.iterrows()
        ]
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("AUC-ROC")
        ax.set_title(title)
        ax.axvline(0.5, color="grey", linestyle="--", alpha=0.5)
        for i, (_, r) in enumerate(top.iterrows()):
            ax.text(
                r["auc"] + 0.002,
                i,
                f"{r['auc']:.3f}",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(
        OUT / "group_combinations.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"    Saved group_combinations.png")


# =================================================================
# CROSS-TASK COMPARISON
# =================================================================
def plot_cross_task(
    mas_up: np.ndarray, mas_down: np.ndarray, feat_names: List[str]
) -> None:
    up_n = mas_up / (mas_up.sum() + 1e-10)
    down_n = mas_down / (mas_down.sum() + 1e-10)

    combined = up_n + down_n
    top_idx = np.argsort(combined)[::-1][:30]

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    fig.suptitle(
        "Cross-Task Feature Comparison (UP vs DOWN)",
        fontsize=14,
        fontweight="bold",
    )

    # 1 — side-by-side bars
    ax = axes[0]
    names = [feat_names[i] for i in top_idx]
    y_pos = np.arange(len(names))
    ax.barh(y_pos - 0.2, up_n[top_idx], 0.35, color="#2ecc71", label="UP", alpha=0.8)
    ax.barh(
        y_pos + 0.2, down_n[top_idx], 0.35, color="#e74c3c", label="DOWN", alpha=0.8
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized mean |SHAP|")
    ax.set_title("Top Features: UP vs DOWN")
    ax.legend()

    # 2 — scatter
    ax = axes[1]
    groups = [feature_group(n) for n in feat_names]
    for g in sorted(set(groups)):
        mask = [i for i, gg in enumerate(groups) if gg == g]
        ax.scatter(
            up_n[mask],
            down_n[mask],
            c=PALETTE.get(g, "#95a5a6"),
            label=g,
            alpha=0.6,
            s=30,
        )
    lim = max(up_n.max(), down_n.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
    ax.set_xlabel("Importance for UP", fontsize=11)
    ax.set_ylabel("Importance for DOWN", fontsize=11)
    ax.set_title("Feature Importance Scatter")
    ax.legend(fontsize=7, ncol=2)

    # 3 — group-level
    ax = axes[2]
    g_up: Dict[str, float] = {}
    g_down: Dict[str, float] = {}
    for i, nm in enumerate(feat_names):
        g = feature_group(nm)
        g_up[g] = g_up.get(g, 0) + up_n[i]
        g_down[g] = g_down.get(g, 0) + down_n[i]
    all_g = sorted(
        set(list(g_up.keys()) + list(g_down.keys())),
        key=lambda g: g_up.get(g, 0) + g_down.get(g, 0),
        reverse=True,
    )
    y_pos = np.arange(len(all_g))
    ax.barh(
        y_pos - 0.2,
        [g_up.get(g, 0) for g in all_g],
        0.35,
        color="#2ecc71",
        label="UP",
        alpha=0.8,
    )
    ax.barh(
        y_pos + 0.2,
        [g_down.get(g, 0) for g in all_g],
        0.35,
        color="#e74c3c",
        label="DOWN",
        alpha=0.8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_g, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Sum of normalized |SHAP|")
    ax.set_title("Group Importance: UP vs DOWN")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        OUT / "cross_task_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"    Saved cross_task_comparison.png")


# =================================================================
# CONSOLE REPORT HELPERS
# =================================================================
def print_task_report(
    task: str,
    folds: List[FoldInfo],
    feat_names: List[str],
    mas: np.ndarray,
    sel_freq: np.ndarray,
    directions: np.ndarray,
) -> None:
    print(f"\n{'='*80}")
    print(f"  DETAILED REPORT — {task}")
    print(f"{'='*80}")

    # Top features
    print(f"\n  TOP 20 FEATURES by mean |SHAP|:")
    print(f"  {'Rank':<5} {'Feature':<42} {'|SHAP|':>10} {'Direction':>10} "
          f"{'Group':<14} {'MI sel%':>8}")
    print(f"  {'-'*95}")
    top_idx = np.argsort(mas)[::-1][:20]
    for rank, i in enumerate(top_idx, 1):
        g = feature_group(feat_names[i])
        d = directions[i]
        d_str = f"{'↑' if d > 0.1 else '↓' if d < -0.1 else '—'} {d:+.2f}"
        print(
            f"  {rank:<5} {feat_names[i]:<42} {mas[i]:>10.6f} {d_str:>10} "
            f"{g:<14} {sel_freq[i]:>7.0%}"
        )

    # Group summary
    print(f"\n  GROUP IMPORTANCE (sum of |SHAP|):")
    group_shap: Dict[str, float] = {}
    for i, nm in enumerate(feat_names):
        g = feature_group(nm)
        group_shap[g] = group_shap.get(g, 0) + mas[i]
    for g, v in sorted(group_shap.items(), key=lambda x: x[1], reverse=True):
        print(f"    {g:<16s}: {v:.6f}")

    # Ensemble routing
    alphas = [fi.alpha for fi in folds]
    rf_aucs = [fi.rf_oob_auc for fi in folds]
    vol_aucs = [fi.vol_lr_cv_auc for fi in folds]
    n_corr = sum(1 for a in alphas if a > 0)

    print(f"\n  ENSEMBLE ROUTING (α):")
    print(f"    α  mean={np.mean(alphas):.3f}  min={np.min(alphas):.3f}  "
          f"max={np.max(alphas):.3f}")
    print(f"    RF OOB AUC   mean={np.mean(rf_aucs):.3f}  "
          f"(range {np.min(rf_aucs):.3f}–{np.max(rf_aucs):.3f})")
    print(f"    Vol-LR CV AUC mean={np.mean(vol_aucs):.3f}  "
          f"(range {np.min(vol_aucs):.3f}–{np.max(vol_aucs):.3f})")
    print(f"    Folds with Vol-LR active: {n_corr}/{len(folds)}")

    if n_corr > 0:
        active_alphas = [a for a in alphas if a > 0]
        print(f"    When active: mean α = {np.mean(active_alphas):.3f}")

    # Vol-LR coefficients summary
    coefs_all = [fi.vol_lr_coefs for fi in folds if fi.vol_lr_coefs is not None]
    if coefs_all:
        vol_idx = folds[0].vol_feature_idx
        vol_names = [feat_names[i] for i in vol_idx]
        mean_c = np.mean(coefs_all, axis=0)
        order = np.argsort(np.abs(mean_c))[::-1]
        print(f"\n  VOL-LR COEFFICIENTS (mean across folds):")
        for i in order:
            if abs(mean_c[i]) > 0.01:
                print(f"    {vol_names[i]:<30s}: {mean_c[i]:+.4f}")


# =================================================================
# MAIN
# =================================================================
def main():
    t0 = datetime.now()
    print("=" * 80)
    print("CGECD INVESTIGATION — SHAP & Feature Diagnostics")
    print("=" * 80)
    print(f"Start: {t0:%Y-%m-%d %H:%M:%S}")
    print(f"SHAP available: {HAS_SHAP}")
    print(f"Output: {OUT}/\n")

    config = Config()

    # ── 1. Data ──────────────────────────────────────────────────
    print("[1/7] Loading data …")
    prices, returns = load_data(config)

    # ── 2. Features ──────────────────────────────────────────────
    print("\n[2/7] Building features …")
    spectral = build_spectral_features(returns, config)
    traditional = build_traditional_features(prices, returns)
    combined = pd.concat([spectral, traditional], axis=1)
    print(f"\n  Combined: {combined.shape[1]} features")

    # ── 3. Targets ───────────────────────────────────────────────
    print("\n[3/7] Computing targets …")
    market = prices["SP500"] if "SP500" in prices.columns else prices.iloc[:, 0]
    fut_3d = market.pct_change(3).shift(-3)

    tasks = {
        "Large UP move 3d (>3%)": (fut_3d > 0.03).astype(int),
        "Large DOWN move 3d (>3%)": (fut_3d < -0.03).astype(int),
    }
    for name, tgt in tasks.items():
        print(f"  {name}: {tgt.dropna().mean():.1%} positive "
              f"({int(tgt.dropna().sum())} events)")

    # ── 4. Diagnostic walk-forward ───────────────────────────────
    print("\n[4/7] Diagnostic walk-forward …")
    task_data: Dict[str, dict] = {}

    for task_name, target in tasks.items():
        folds, feat_names, all_probs, all_actuals = diagnostic_walk_forward(
            combined, target, config, task_name
        )
        if not folds:
            print(f"  !! No successful folds for {task_name}")
            continue

        if len(all_probs) > 0 and len(np.unique(all_actuals)) >= 2:
            m = compute_metrics(
                all_actuals, (all_probs >= 0.5).astype(int), all_probs
            )
            print(f"  → AUC={m.auc_roc:.3f}  F1={m.f1:.3f}  "
                  f"Prec={m.precision:.3f}  Rec={m.recall:.3f}")

        task_data[task_name] = {
            "folds": folds,
            "feat_names": feat_names,
            "all_probs": all_probs,
            "all_actuals": all_actuals,
        }

    # ── 5. SHAP analysis & per-task plots ────────────────────────
    print("\n[5/7] SHAP analysis & plots …")
    shap_rankings: Dict[str, np.ndarray] = {}
    directions_map: Dict[str, np.ndarray] = {}
    sel_freq_map: Dict[str, np.ndarray] = {}

    for task_name, tdata in task_data.items():
        folds = tdata["folds"]
        feat_names = tdata["feat_names"]
        n_feat = len(feat_names)

        print(f"\n  ── {task_name} ──")

        # Aggregate SHAP
        shap_full, X_full = aggregate_shap(folds, n_feat)
        mas = mean_abs_shap(shap_full)
        shap_rankings[task_name] = mas

        has_shap_data = HAS_SHAP and shap_full.shape[0] > 1

        if has_shap_data:
            print("    Plotting SHAP bar …")
            plot_shap_bar(mas, feat_names, task_name)

            print("    Plotting SHAP beeswarm …")
            plot_shap_beeswarm(shap_full, X_full, feat_names, task_name)

            print("    Plotting SHAP dependence …")
            plot_shap_dependence(shap_full, X_full, feat_names, task_name)

            print("    Plotting SHAP direction …")
            directions = plot_shap_direction(
                shap_full, X_full, feat_names, task_name
            )
            directions_map[task_name] = directions
        else:
            directions_map[task_name] = np.zeros(n_feat)

        print("    Plotting SHAP by group …")
        plot_shap_by_group(mas, feat_names, task_name)

        print("    Plotting Vol-LR analysis …")
        plot_vol_lr_analysis(folds, feat_names, task_name)

        print("    Plotting MI selection frequency …")
        sel_freq = plot_selection_frequency(folds, feat_names, task_name)
        sel_freq_map[task_name] = sel_freq

        print("    Plotting MI vs SHAP …")
        plot_mi_vs_shap(folds, mas, feat_names, task_name)

        # Console report
        print_task_report(
            task_name,
            folds,
            feat_names,
            mas,
            sel_freq,
            directions_map[task_name],
        )

    # ── 6. Feature selection sweep & group search ────────────────
    print("\n[6/7] Feature sweep & group combination search …")

    sweep_results: Dict[str, Dict[int, float]] = {}
    gc_results: Dict[str, pd.DataFrame] = {}

    for task_name, target in tasks.items():
        if task_name not in shap_rankings:
            continue

        print(f"\n  Feature sweep: {task_name}")
        sweep = feature_selection_sweep(
            combined,
            target,
            list(combined.columns),
            shap_rankings[task_name],
            config,
            task_name,
        )
        sweep_results[task_name] = sweep
        if sweep:
            best_k = max(sweep, key=sweep.get)
            print(f"    Best: k={best_k} → AUC={sweep[best_k]:.3f}")
            # Print sweep table
            print(f"    {'k':>5}  {'AUC':>8}")
            for k in sorted(sweep.keys()):
                marker = " ◀" if k == best_k else ""
                print(f"    {k:>5}  {sweep[k]:>8.3f}{marker}")

        print(f"\n  Group combination search: {task_name}")
        gc = group_combination_search(
            combined, target, list(combined.columns), config, task_name
        )
        gc_results[task_name] = gc
        if not gc.empty:
            gc.to_csv(OUT / f"group_combos_{_safe(task_name)}.csv", index=False)
            print(f"\n    TOP 10 GROUP COMBINATIONS:")
            print(f"    {'AUC':>7}  {'#G':>3}  {'#F':>4}  Groups")
            print(f"    {'─'*70}")
            for _, row in gc.head(10).iterrows():
                print(
                    f"    {row['auc']:>7.3f}  {row['n_groups']:>3}  "
                    f"{row['n_features']:>4}  {row['groups']}"
                )

    # Plot sweep
    plot_selection_sweep(
        sweep_results.get("Large UP move 3d (>3%)", {}),
        sweep_results.get("Large DOWN move 3d (>3%)", {}),
    )

    # Plot group combos
    plot_group_combinations(
        gc_results.get("Large UP move 3d (>3%)", pd.DataFrame()),
        gc_results.get("Large DOWN move 3d (>3%)", pd.DataFrame()),
    )

    # ── 7. Cross-task comparison ─────────────────────────────────
    print("\n[7/7] Cross-task comparison …")

    if len(shap_rankings) == 2:
        tnames = list(shap_rankings.keys())
        feat_names = task_data[tnames[0]]["feat_names"]

        plot_cross_task(
            shap_rankings[tnames[0]], shap_rankings[tnames[1]], feat_names
        )

        mas_up = shap_rankings[tnames[0]]
        mas_down = shap_rankings[tnames[1]]
        up_n = mas_up / (mas_up.sum() + 1e-10)
        down_n = mas_down / (mas_down.sum() + 1e-10)
        ratio = up_n / (down_n + 1e-10)

        print(f"\n  {'─'*70}")
        print("  UNIQUE UP DRIVERS (high UP importance, low DOWN):")
        for i in np.argsort(ratio)[::-1][:10]:
            if up_n[i] > 0.003:
                print(
                    f"    {feat_names[i]:<42s}  UP={up_n[i]:.4f}  "
                    f"DOWN={down_n[i]:.4f}  ratio={ratio[i]:.1f}"
                )

        print(f"\n  UNIQUE DOWN DRIVERS (high DOWN importance, low UP):")
        for i in np.argsort(ratio)[:10]:
            if down_n[i] > 0.003:
                print(
                    f"    {feat_names[i]:<42s}  UP={up_n[i]:.4f}  "
                    f"DOWN={down_n[i]:.4f}  ratio={ratio[i]:.1f}"
                )

        # Shared drivers
        print(f"\n  SHARED DRIVERS (important for both):")
        product = up_n * down_n
        for i in np.argsort(product)[::-1][:10]:
            if product[i] > 1e-6:
                print(
                    f"    {feat_names[i]:<42s}  UP={up_n[i]:.4f}  "
                    f"DOWN={down_n[i]:.4f}"
                )

    # ── Save summary ─────────────────────────────────────────────
    summary: Dict = {}
    for task_name in task_data:
        mas = shap_rankings[task_name]
        fn = task_data[task_name]["feat_names"]
        folds = task_data[task_name]["folds"]
        top_idx = np.argsort(mas)[::-1][:30]

        sweep = sweep_results.get(task_name, {})
        best_k = max(sweep, key=sweep.get) if sweep else None

        gc = gc_results.get(task_name, pd.DataFrame())
        best_combo = gc.iloc[0].to_dict() if not gc.empty else None

        summary[task_name] = {
            "top_30_features": [
                {
                    "name": fn[i],
                    "mean_abs_shap": float(mas[i]),
                    "group": feature_group(fn[i]),
                    "direction": float(directions_map.get(task_name, np.zeros(len(fn)))[i]),
                    "selection_freq": float(sel_freq_map.get(task_name, np.zeros(len(fn)))[i]),
                }
                for i in top_idx
            ],
            "alpha_mean": float(np.mean([fi.alpha for fi in folds])),
            "alpha_max": float(np.max([fi.alpha for fi in folds])),
            "rf_auc_mean": float(np.mean([fi.rf_oob_auc for fi in folds])),
            "vol_lr_auc_mean": float(
                np.mean([fi.vol_lr_cv_auc for fi in folds])
            ),
            "folds_with_vol_lr": int(
                sum(1 for fi in folds if fi.alpha > 0)
            ),
            "optimal_k": best_k,
            "optimal_k_auc": float(sweep[best_k]) if best_k else None,
            "best_group_combo": best_combo,
        }

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Final summary table ──────────────────────────────────────
    print(f"\n{'='*80}")
    print("  INVESTIGATION SUMMARY")
    print(f"{'='*80}")

    for task_name, s in summary.items():
        print(f"\n  {task_name}:")
        print(f"    RF mean AUC:        {s['rf_auc_mean']:.3f}")
        print(f"    Vol-LR mean AUC:    {s['vol_lr_auc_mean']:.3f}")
        print(f"    Mean α:             {s['alpha_mean']:.3f}  "
              f"(active in {s['folds_with_vol_lr']} folds)")
        if s["optimal_k"]:
            print(f"    Optimal # features: {s['optimal_k']}  "
                  f"(AUC={s['optimal_k_auc']:.3f})")
        if s["best_group_combo"]:
            bc = s["best_group_combo"]
            print(f"    Best group combo:   {bc['groups']}  "
                  f"(AUC={bc['auc']:.3f})")
        print(f"    Top-5 SHAP features:")
        for i, feat in enumerate(s["top_30_features"][:5], 1):
            d = feat["direction"]
            arrow = "↑" if d > 0.1 else ("↓" if d < -0.1 else "—")
            print(
                f"      {i}. {feat['name']:<38s} "
                f"|SHAP|={feat['mean_abs_shap']:.5f}  "
                f"{arrow}  [{feat['group']}]"
            )

    print(f"\n  All outputs saved to {OUT}/")
    print(f"  Runtime: {datetime.now() - t0}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()