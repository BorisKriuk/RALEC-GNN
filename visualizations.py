#!/usr/bin/env python3
"""All plots for the CGECD paper."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


# =====================================================================
# HELPERS
# =====================================================================
def _group(name: str) -> str:
    n = name.lower()
    # Dynamics / acceleration always takes priority
    if "_zscore_" in n or "_roc_" in n or "_accel_" in n:
        return "Dynamics"
    # Cross-window ratios stay with their parent group
    if "_ratio" in n and any(
        k in n
        for k in ("lambda", "absorption", "effective_rank", "edge_density", "mean_abs_corr")
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


_PALETTE = {
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


# =====================================================================
# 1. FEATURE IMPORTANCE
# =====================================================================
def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    top_k: int = 30,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 14))
    fig.suptitle("Feature Importance Analysis", fontsize=16, fontweight="bold")

    idx = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]
    groups = [_group(n) for n in names]
    colors = [_PALETTE.get(g, "#95a5a6") for g in groups]

    ax1.barh(
        range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.5
    )
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Feature Importance")
    ax1.set_title(f"Top {top_k} Features")

    seen = set()
    handles = []
    for g in groups:
        if g not in seen:
            seen.add(g)
            handles.append(
                plt.Rectangle(
                    (0, 0), 1, 1, fc=_PALETTE.get(g, "#95a5a6"), label=g
                )
            )
    ax1.legend(handles=handles, loc="lower right", fontsize=8)

    # By group
    gmap: Dict[str, float] = {}
    for i, fn in enumerate(feature_names):
        g = _group(fn)
        gmap[g] = gmap.get(g, 0) + importances[i]
    sg = sorted(gmap.items(), key=lambda x: x[1], reverse=True)
    ax2.barh(
        range(len(sg)),
        [v for _, v in sg],
        color=[_PALETTE.get(g, "#95a5a6") for g, _ in sg],
        edgecolor="white",
    )
    ax2.set_yticks(range(len(sg)))
    ax2.set_yticklabels([g for g, _ in sg], fontsize=11)
    ax2.invert_yaxis()
    ax2.set_xlabel("Sum of Importance")
    ax2.set_title("Importance by Feature Group")

    plt.tight_layout()
    path = output_dir / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# =====================================================================
# 2. MODEL COMPARISON
# =====================================================================
def plot_model_comparison(
    results: List[Dict], output_dir: Path, suffix: str = ""
) -> None:
    fig, ax = plt.subplots(figsize=(12, max(6, len(results) * 0.8)))
    fig.suptitle("Model Comparison: AUC-ROC", fontsize=16, fontweight="bold")

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

    bars = ax.barh(
        range(len(names)), aucs, color=colors, edgecolor="white", height=0.6
    )
    for bar, auc in zip(bars, aucs):
        ax.text(
            bar.get_width() + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{auc:.3f}",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("AUC-ROC", fontsize=12)
    ax.set_xlim(0.4, 1.0)

    plt.tight_layout()
    path = output_dir / f"model_comparison{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# =====================================================================
# 3. ABLATION
# =====================================================================
def plot_ablation(ablation: List[Dict], output_dir: Path) -> None:
    metrics_list = [
        ("auc_roc", "AUC-ROC"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score"),
    ]
    modules = [r["module"] for r in ablation]
    pal = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12"][: len(modules)]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        "Performance Comparison: Feature Modules",
        fontsize=16,
        fontweight="bold",
    )
    for ax, (key, title) in zip(axes, metrics_list):
        vals = [r[key] for r in ablation]
        bars = ax.bar(
            range(len(modules)), vals, color=pal, edgecolor="white", width=0.6
        )
        if key == "auc_roc":
            ax.axhline(
                y=0.5, color="red", linestyle="--", linewidth=1, alpha=0.7
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )
        ax.set_xticks(range(len(modules)))
        ax.set_xticklabels(modules, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 1.0)

    plt.tight_layout()
    path = output_dir / "ablation_study.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# =====================================================================
# 4. ROC CURVES
# =====================================================================
def plot_roc_curves(raw: Dict, output_dir: Path, suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")

    for name, res in raw.items():
        if "actuals" not in res:
            continue
        fpr, tpr, _ = roc_curve(res["actuals"], res["probabilities"])
        auc = res["metrics"].auc_roc
        lw = 3 if res.get("is_ours") else 1.5
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=lw)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"roc_curves{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# =====================================================================
# 5. PRECISION-RECALL CURVES
# =====================================================================
def plot_pr_curves(raw: Dict, output_dir: Path, suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Precision–Recall Curves", fontsize=14, fontweight="bold")

    for name, res in raw.items():
        if "actuals" not in res:
            continue
        prec, rec, _ = precision_recall_curve(
            res["actuals"], res["probabilities"]
        )
        ap = res["metrics"].avg_precision
        lw = 3 if res.get("is_ours") else 1.5
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", linewidth=lw)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"pr_curves{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")