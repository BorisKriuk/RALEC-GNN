#!/usr/bin/env python3
"""
CGECD — Main Experiment  v2
============================
Tasks:
  1. Large UP   move 3 d  (> 3 %)
  2. Large DOWN move 3 d  (> 3 %)

Comparisons:
  • 4 ML baselines  (LR / SVM / GB / RF on traditional features)
  • 4 signal-based  (Absorption Ratio / Turbulence / GARCH / HAR-RV)

Ablation:
  • Spectral Only  •  Traditional Only
  • Combined no sel  •  Combined + MI selection (CGECD-RF)

Outputs:  tables, CSV, 7+ PNG visualisations
"""

import warnings

warnings.filterwarnings("ignore")

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from config import Config
from metrics import compute_metrics, significance_test
from algorithm import (
    load_data,
    build_spectral_features,
    build_traditional_features,
    CGECDModel,
    CGECDRFModel,
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
from visualizations import (
    plot_feature_importance,
    plot_model_comparison,
    plot_ablation,
    plot_roc_curves,
    plot_pr_curves,
)


# ------------------------------------------------------------------
def _safe_suffix(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace(">", "gt")
        .replace("<", "lt")
    )


def _print_table(task: str, results: List[Dict], pos_rate: float) -> None:
    print(f"\n{'='*95}")
    print(f"TASK: {task}")
    print(f"Positive rate: {pos_rate:.1%}")
    print(f"{'='*95}")
    hdr = (
        f"{'Model':<35} {'AUC-ROC':>9} {'AvgPrec':>9}"
        f" {'Prec':>9} {'Recall':>9} {'F1':>9}"
    )
    print(hdr)
    print("-" * 95)
    for r in sorted(results, key=lambda x: x["auc_roc"], reverse=True):
        m = "★" if r.get("is_ours") else " "
        print(
            f"{m} {r['model']:<33} {r['auc_roc']:>9.3f}"
            f" {r['avg_precision']:>9.3f} {r['precision']:>8.1%}"
            f" {r['recall']:>8.1%} {r['f1']:>8.1%}"
        )
    print("-" * 95)
    print("★ = Our method (CGECD)\n")


# ==================================================================
def run_experiment():
    t0 = datetime.now()
    print("=" * 95)
    print("CORRELATION GRAPH EIGENVALUE CRISIS DETECTOR  (CGECD)  v2")
    print("Extended Comparison against SOTA Benchmarks")
    print("=" * 95)
    print(f"Start: {t0:%Y-%m-%d %H:%M:%S}\n")

    config = Config()

    # ==============================================================
    # 1. DATA
    # ==============================================================
    print("[1/5] Loading data …")
    prices, returns = load_data(config)

    # ==============================================================
    # 2. FEATURES
    # ==============================================================
    print("\n[2/5] Building features …")

    print("\n  — Spectral features")
    spectral_features = build_spectral_features(returns, config)

    print("\n  — Traditional features")
    traditional_features = build_traditional_features(prices, returns)

    cgecd_features = pd.concat([spectral_features, traditional_features], axis=1)

    print("\n  — Benchmark features")
    bench_features = prepare_benchmark_features(prices, returns)

    print(
        f"\n  Dimensions:  spectral={spectral_features.shape[1]}  "
        f"traditional={traditional_features.shape[1]}  "
        f"combined={cgecd_features.shape[1]}"
    )

    # ==============================================================
    # 3. TARGETS
    # ==============================================================
    print("\n[3/5] Computing targets …")
    market = prices["SP500"] if "SP500" in prices.columns else prices.iloc[:, 0]
    fut_3d = market.pct_change(3).shift(-3)

    tasks = {
        "Large UP move 3d (>3%)": (fut_3d > 0.03).astype(int),
        "Large DOWN move 3d (>3%)": (fut_3d < -0.03).astype(int),
    }
    for name, tgt in tasks.items():
        n_pos = int(tgt.dropna().sum())
        print(f"  {name}: {tgt.dropna().mean():.1%} positive ({n_pos} events)")

    # ==============================================================
    # 4. MAIN COMPARISON  +  ABLATION
    # ==============================================================
    print("\n[4/5] Running experiments …")

    all_results: Dict[str, Dict] = {}

    for task_name, target in tasks.items():
        print(f"\n{'─'*60}")
        print(f"  TASK: {task_name}")
        print(f"{'─'*60}")

        pos_rate = float(target.dropna().mean())
        task_results: List[Dict] = []
        task_raw: Dict[str, Dict] = {}

        # ---- Our method: CGECD (Ensemble) ----
        print("    CGECD (Ours) …")
        res = walk_forward_evaluate(cgecd_features, target, CGECDModel, config)
        if "error" not in res:
            m = res["metrics"]
            task_results.append(
                dict(
                    model="CGECD (Ours)",
                    auc_roc=m.auc_roc,
                    avg_precision=m.avg_precision,
                    precision=m.precision,
                    recall=m.recall,
                    f1=m.f1,
                    is_ours=True,
                )
            )
            task_raw["CGECD (Ours)"] = {**res, "is_ours": True}
            print(f"      AUC={m.auc_roc:.3f}")
        else:
            print(f"      FAILED: {res['error']}")

        # ---- ML baselines (traditional features) ----
        ml_baselines = [
            ("Logistic Regression", LogisticRegressionModel),
            ("SVM", SVMBaselineModel),
            ("Gradient Boosting", GradientBoostingModel),
            ("Random Forest", RandomForestBaselineModel),
        ]
        for bname, bclass in ml_baselines:
            print(f"    {bname} …")
            res = walk_forward_evaluate(
                traditional_features, target, bclass, config
            )
            if "error" not in res:
                m = res["metrics"]
                task_results.append(
                    dict(
                        model=bname,
                        auc_roc=m.auc_roc,
                        avg_precision=m.avg_precision,
                        precision=m.precision,
                        recall=m.recall,
                        f1=m.f1,
                        is_ours=False,
                    )
                )
                task_raw[bname] = {**res, "is_ours": False}
                print(f"      AUC={m.auc_roc:.3f}")
            else:
                print(f"      FAILED: {res['error']}")

        # ---- Signal-based benchmarks ----
        signal_benchmarks = [
            (
                "Absorption Ratio",
                bench_features["absorption_ratio"],
                AbsorptionRatioModel,
            ),
            ("Turbulence Index", bench_features["turbulence"], TurbulenceModel),
            ("GARCH(1,1)", bench_features["garch"], GARCHModel),
            ("HAR-RV", bench_features["har_rv"], HARRVModel),
        ]
        for bname, bfeat, bclass in signal_benchmarks:
            print(f"    {bname} …")
            res = walk_forward_evaluate(bfeat, target, bclass, config)
            if "error" not in res:
                m = res["metrics"]
                task_results.append(
                    dict(
                        model=bname,
                        auc_roc=m.auc_roc,
                        avg_precision=m.avg_precision,
                        precision=m.precision,
                        recall=m.recall,
                        f1=m.f1,
                        is_ours=False,
                    )
                )
                task_raw[bname] = {**res, "is_ours": False}
                print(f"      AUC={m.auc_roc:.3f}")
            else:
                print(f"      FAILED: {res['error']}")

        all_results[task_name] = dict(
            results=task_results, pos_rate=pos_rate, raw=task_raw
        )

    # ---- Ablation study (first task only) ----
    print(f"\n{'─'*60}")
    print("  ABLATION STUDY")
    print(f"{'─'*60}")

    abl_target = list(tasks.values())[0]
    ablation: List[Dict] = []

    abl_configs = [
        ("Spectral Only", spectral_features, CGECDNoSelModel),
        ("Traditional Only", traditional_features, CGECDNoSelModel),
        ("Combined (no sel)", cgecd_features, CGECDNoSelModel),
        ("CGECD-RF (selected)", cgecd_features, CGECDRFModel),
        ("CGECD (ensemble)", cgecd_features, CGECDModel),
    ]
    for aname, afeat, acls in abl_configs:
        print(f"    {aname} …")
        res = walk_forward_evaluate(afeat, abl_target, acls, config)
        if "error" not in res:
            m = res["metrics"]
            ablation.append(
                dict(
                    module=aname,
                    auc_roc=m.auc_roc,
                    precision=m.precision,
                    recall=m.recall,
                    f1=m.f1,
                )
            )
            print(f"      AUC={m.auc_roc:.3f}")
        else:
            print(f"      FAILED: {res['error']}")

    # ==============================================================
    # 5. OUTPUT
    # ==============================================================
    print("\n[5/5] Generating outputs …\n")

    # ---- Tables ----
    for task_name, data in all_results.items():
        _print_table(task_name, data["results"], data["pos_rate"])

    if ablation:
        print("ABLATION (first task):")
        for a in ablation:
            print(
                f"  {a['module']:<25} AUC={a['auc_roc']:.3f}  F1={a['f1']:.1%}"
            )
        print()

    # ---- Visualisations ----
    first_raw = list(all_results.values())[0]["raw"]
    if (
        "CGECD (Ours)" in first_raw
        and "feature_importances" in first_raw["CGECD (Ours)"]
    ):
        plot_feature_importance(
            first_raw["CGECD (Ours)"]["feature_importances"],
            first_raw["CGECD (Ours)"]["feature_names"],
            config.output_dir,
        )

    for task_name, data in all_results.items():
        sfx = "_" + _safe_suffix(task_name)

        comparison = data["results"] + [
            dict(
                model="Random Baseline",
                auc_roc=0.5,
                avg_precision=data["pos_rate"],
                precision=0,
                recall=0,
                f1=0,
                is_ours=False,
            )
        ]
        plot_model_comparison(comparison, config.output_dir, suffix=sfx)
        plot_roc_curves(data["raw"], config.output_dir, suffix=sfx)
        plot_pr_curves(data["raw"], config.output_dir, suffix=sfx)

    if ablation:
        plot_ablation(ablation, config.output_dir)

    # ---- Summary ----
    print("=" * 95)
    print("SUMMARY")
    print("=" * 95)

    for task_name, data in all_results.items():
        ours = next((r for r in data["results"] if r.get("is_ours")), None)
        if not ours:
            continue
        others = [r for r in data["results"] if not r.get("is_ours")]
        if not others:
            continue
        best = max(others, key=lambda x: x["auc_roc"])
        diff = ours["auc_roc"] - best["auc_roc"]
        sign = "✓" if diff > 0.02 else ("≈" if abs(diff) <= 0.02 else "✗")
        print(f"\n{sign} {task_name}")
        print(f"  CGECD:          AUC = {ours['auc_roc']:.3f}")
        print(f"  Best benchmark: {best['model']}  AUC = {best['auc_roc']:.3f}")
        print(f"  Δ = {diff:+.3f}")

    # ---- Significance tests ----
    print("\nSignificance tests (bootstrap, 2 000 resamples):")
    for task_name, data in all_results.items():
        raw = data["raw"]
        if "CGECD (Ours)" not in raw:
            continue
        o = raw["CGECD (Ours)"]
        for bname, b in raw.items():
            if bname == "CGECD (Ours)":
                continue
            # With aligned folds, actuals lengths and values match
            if len(o["actuals"]) != len(b["actuals"]):
                print(
                    f"  [{task_name}] CGECD vs {bname}: SKIPPED "
                    f"(mismatched lengths {len(o['actuals'])} vs {len(b['actuals'])})"
                )
                continue
            try:
                st = significance_test(
                    o["actuals"], o["probabilities"], b["probabilities"]
                )
                tag = "p<0.05 ✓" if st["significant_05"] else "n.s."
                print(
                    f"  [{task_name}] CGECD vs {bname}: "
                    f"Δ={st['auc_diff']:+.3f}  ({tag})"
                )
            except Exception as e:
                print(f"  [{task_name}] CGECD vs {bname}: ERROR ({e})")

    # ---- Save CSV ----
    rows = []
    for task_name, data in all_results.items():
        for r in data["results"]:
            rows.append({"Task": task_name, **r})
    pd.DataFrame(rows).to_csv(config.output_dir / "results.csv", index=False)

    print(f"\nAll results saved to {config.output_dir}/")
    print(f"Runtime: {datetime.now() - t0}")
    return all_results


# ==================================================================
if __name__ == "__main__":
    run_experiment()