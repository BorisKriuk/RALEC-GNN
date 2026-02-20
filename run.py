#!/usr/bin/env python3
"""
CGECD — Main Experiment  v4
============================
Tasks:
  1. Large UP   move 3 d  (> 3 %)
  2. Large DOWN move 3 d  (> 3 %)

Evaluation: Bidirectional Reliability
  Primary sort:  Worst-Case Rank  (no directional blind spots)
  Secondary:     Average Rank     (overall performance)
  Tertiary:      Net Dominance    (statistical proof)
"""

import warnings

warnings.filterwarnings("ignore")

from collections import defaultdict
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
    print("CORRELATION GRAPH EIGENVALUE CRISIS DETECTOR  (CGECD)  v4")
    print("Bidirectional Reliability Evaluation")
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

        # ---- Our method: CGECD ----
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
        ("CGECD (full)", cgecd_features, CGECDModel),
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

    # ---- Per-task tables ----
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

    # ---- Per-task summary ----
    print("=" * 95)
    print("PER-TASK SUMMARY")
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

    # ---- Significance tests (CGECD vs each) ----
    print("\nSignificance tests (bootstrap, 2 000 resamples):")
    for task_name, data in all_results.items():
        raw = data["raw"]
        if "CGECD (Ours)" not in raw:
            continue
        o = raw["CGECD (Ours)"]
        for bname, b in raw.items():
            if bname == "CGECD (Ours)":
                continue
            if len(o["actuals"]) != len(b["actuals"]):
                print(
                    f"  [{task_name}] CGECD vs {bname}: SKIPPED "
                    f"(mismatched lengths)"
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

    # ---- All-pairs significance ----
    task_names = list(all_results.keys())
    n_tasks = len(task_names)

    model_sig_wins = defaultdict(int)
    model_sig_losses = defaultdict(int)

    print(f"\nAll-pairs significance testing …")
    for task_name, data in all_results.items():
        raw = data["raw"]
        models_in_task = list(raw.keys())
        for i in range(len(models_in_task)):
            for j in range(i + 1, len(models_in_task)):
                a, b = models_in_task[i], models_in_task[j]
                if len(raw[a]["actuals"]) != len(raw[b]["actuals"]):
                    continue
                try:
                    st = significance_test(
                        raw[a]["actuals"],
                        raw[a]["probabilities"],
                        raw[b]["probabilities"],
                    )
                    if st["p_value"] < 0.05:
                        model_sig_wins[a] += 1
                        model_sig_losses[b] += 1
                    elif st["p_value"] > 0.95:
                        model_sig_wins[b] += 1
                        model_sig_losses[a] += 1
                except Exception:
                    pass

    # ---- Composite rankings ----
    model_task_auc = defaultdict(dict)
    model_task_rank = defaultdict(dict)

    for task_name, data in all_results.items():
        sorted_r = sorted(
            data["results"], key=lambda x: x["auc_roc"], reverse=True
        )
        for rank, r in enumerate(sorted_r, 1):
            model_task_auc[r["model"]][task_name] = r["auc_roc"]
            model_task_rank[r["model"]][task_name] = rank

    composite = {}
    for model in model_task_rank:
        if len(model_task_rank[model]) == n_tasks:
            ranks = list(model_task_rank[model].values())
            aucs = list(model_task_auc[model].values())
            sw = model_sig_wins.get(model, 0)
            sl = model_sig_losses.get(model, 0)
            composite[model] = {
                "avg_rank": np.mean(ranks),
                "worst_rank": max(ranks),
                "top2_count": sum(1 for r in ranks if r <= 2),
                "balanced_auc": np.mean(aucs),
                "min_auc": min(aucs),
                "sig_wins": sw,
                "sig_losses": sl,
                "net_dom": sw - sl,
            }

    # ── Sort: worst_rank (no blind spots), avg_rank, net_dom ───
    sorted_models = sorted(
        composite.keys(),
        key=lambda m: (
            composite[m]["worst_rank"],
            composite[m]["avg_rank"],
            -composite[m]["net_dom"],
        ),
    )

    # ---- Print composite table ----
    print(f"\n{'='*120}")
    print("COMPOSITE EVALUATION — Bidirectional Reliability")
    print(f"{'='*120}")
    print(
        "A risk manager deploys ONE model for both upside and downside"
        " tail events.  A model"
    )
    print(
        "that excels on one direction but fails on the other creates a"
        " dangerous blind spot."
    )
    print(
        "Primary sort: Worst-Case Rank (lower = no blind spots)."
        "  Tiebreaker: Avg Rank, Net Dom.\n"
    )

    hdr = f"  {'#':>2}  {'Model':<26s}"
    for t in task_names:
        short = (
            t.replace("Large ", "")
            .replace(" move", "")
            .replace("(>3%)", ">3%")
            .replace("(", "")
            .replace(")", "")
        )
        if len(short) > 14:
            short = short[:14]
        hdr += f"  {short:>14s}"
    hdr += (
        f"  {'Worst':>5s}  {'Avg':>5s}  {'Top2':>5s}"
        f"  {'W':>3s}  {'L':>3s}  {'Net':>4s}"
        f"  {'BalAUC':>6s}"
    )
    print(hdr)
    print("-" * 120)

    for table_rank, model in enumerate(sorted_models, 1):
        c = composite[model]
        marker = "★" if model == "CGECD (Ours)" else " "
        line = f"{marker} {table_rank:>2}  {model:<26s}"
        for t in task_names:
            auc = model_task_auc[model][t]
            rank = model_task_rank[model][t]
            line += f"  {auc:.3f} (#{rank:<2d})"
        line += f"  {c['worst_rank']:>5d}"
        line += f"  {c['avg_rank']:>5.1f}"
        line += f"  {c['top2_count']:>3d}/{n_tasks}"
        line += f"  {c['sig_wins']:>3d}"
        line += f"  {c['sig_losses']:>3d}"
        line += f"  {c['net_dom']:>+4d}"
        line += f"  {c['balanced_auc']:>6.3f}"
        print(line)

    print("-" * 120)
    print(
        "Sorted by: Worst Rank (primary), Avg Rank (secondary),"
        " Net Dominance (tertiary)."
    )
    print(
        "W/L/Net = pairwise significant wins / losses / (W−L)"
        " across all model pairs and tasks."
    )

    # ---- Per-dimension analysis ----
    ours_name = "CGECD (Ours)"

    dims = [
        ("Worst-Case Rank", "worst_rank", False,
         "Absence of directional blind spots"),
        ("Average Rank", "avg_rank", False,
         "Overall discriminative power"),
        ("Top-2 Consistency", "top2_count", True,
         "Fraction of tasks with near-optimal performance"),
        ("Net Stat. Dominance", "net_dom", True,
         "Significant wins minus losses (all pairs)"),
        ("Balanced AUC", "balanced_auc", True,
         "Mean AUC across tasks"),
    ]

    print(f"\n{'='*90}")
    print("PER-DIMENSION ANALYSIS")
    print(f"{'='*90}")

    cgecd_dim_wins = 0
    for dim_name, dim_key, higher_better, description in dims:
        if higher_better:
            best_val = max(composite[m][dim_key] for m in composite)
            best_models = [
                m for m in composite if composite[m][dim_key] == best_val
            ]
        else:
            best_val = min(composite[m][dim_key] for m in composite)
            best_models = [
                m for m in composite if composite[m][dim_key] == best_val
            ]

        ours_val = composite.get(ours_name, {}).get(dim_key, None)
        wins = ours_name in best_models
        if wins:
            cgecd_dim_wins += 1

        tag = "✓" if wins else " "
        best_str = ", ".join(best_models)
        print(
            f"\n  {tag} {dim_name} — {description}"
            f"\n    Best: {best_str} = {best_val}"
            f"\n    CGECD: {ours_val}"
        )

    print(f"\n  CGECD wins {cgecd_dim_wins}/{len(dims)} dimensions")

    # ---- CGECD profile ----
    if ours_name in composite:
        c_ours = composite[ours_name]

        print(f"\n{'─'*90}")
        print("★ CGECD RELIABILITY PROFILE")
        print(f"{'─'*90}")
        print(f"  Worst-Case Rank:  {c_ours['worst_rank']}")
        print(f"  Average Rank:     {c_ours['avg_rank']:.1f}")
        print(f"  Top-2 Count:      {c_ours['top2_count']}/{n_tasks}")
        print(f"  Balanced AUC:     {c_ours['balanced_auc']:.3f}")
        print(
            f"  Stat. Dominance:  {c_ours['sig_wins']}W"
            f" / {c_ours['sig_losses']}L"
            f" = {c_ours['net_dom']:+d} net"
        )

        # Head-to-head vs top competitors
        others = [
            (m, c) for m, c in composite.items() if m != ours_name
        ]
        others_sorted = sorted(
            others,
            key=lambda x: (x[1]["worst_rank"], x[1]["avg_rank"]),
        )

        if others_sorted:
            for comp_name, comp_c in others_sorted[:3]:
                print(f"\n    vs {comp_name}:")

                task_wins = 0
                for t in task_names:
                    o_r = model_task_rank[ours_name][t]
                    c_r = model_task_rank[comp_name][t]
                    o_a = model_task_auc[ours_name][t]
                    c_a = model_task_auc[comp_name][t]
                    w = "✓" if o_r < c_r else ("=" if o_r == c_r else "✗")
                    if o_r < c_r:
                        task_wins += 1
                    short_t = (
                        t.replace("Large ", "")
                        .replace(" move 3d (>3%)", "")
                    )
                    print(
                        f"      {short_t:<8s}: CGECD #{o_r}"
                        f" ({o_a:.3f}) vs #{c_r} ({c_a:.3f})  {w}"
                    )

                dim_wins = 0
                for _, dk, hb, _ in dims:
                    ov = c_ours[dk]
                    cv = comp_c[dk]
                    if hb:
                        if ov > cv:
                            dim_wins += 1
                    else:
                        if ov < cv:
                            dim_wins += 1
                print(
                    f"      CGECD wins {task_wins}/{n_tasks} tasks,"
                    f" {dim_wins}/{len(dims)} composite dimensions"
                )

                for t in task_names:
                    r = model_task_rank[comp_name][t]
                    if r >= 4:
                        short_t = (
                            t.replace("Large ", "")
                            .replace(" 3d (>3%)", "")
                        )
                        print(
                            f"      ⚠ {comp_name} drops to #{r}"
                            f" on {short_t}"
                        )

    # ---- Save CSV ----
    rows = []
    for task_name, data in all_results.items():
        for r in data["results"]:
            rows.append({"Task": task_name, **r})
    pd.DataFrame(rows).to_csv(config.output_dir / "results.csv", index=False)

    comp_rows = []
    for model in sorted_models:
        c = composite[model]
        row = {"Model": model}
        for t in task_names:
            row[f"AUC_{_safe_suffix(t)}"] = model_task_auc[model][t]
            row[f"Rank_{_safe_suffix(t)}"] = model_task_rank[model][t]
        row["Worst_Rank"] = c["worst_rank"]
        row["Avg_Rank"] = c["avg_rank"]
        row["Top2_Count"] = c["top2_count"]
        row["Balanced_AUC"] = c["balanced_auc"]
        row["Sig_Wins"] = c["sig_wins"]
        row["Sig_Losses"] = c["sig_losses"]
        row["Net_Dominance"] = c["net_dom"]
        comp_rows.append(row)
    pd.DataFrame(comp_rows).to_csv(
        config.output_dir / "composite_evaluation.csv", index=False
    )

    print(f"\nAll results saved to {config.output_dir}/")
    print(f"Runtime: {datetime.now() - t0}")
    return all_results


# ==================================================================
if __name__ == "__main__":
    run_experiment()