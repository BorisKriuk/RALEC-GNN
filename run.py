#!/usr/bin/env python3
"""
CGECD: Correlation Graph Eigenvalue Crisis Detector
====================================================

Focused two-task evaluation:
  Task 1 — Rally Detection:  S&P 500 up >3% in next 10 days
  Task 2 — Crash Detection:  S&P 500 max drawdown >7% in next 10 days

Combined metric: BCD-AUC (Balanced Crisis Detection AUC)
  = geometric mean of per-task AUC-ROC
  Rewards models that detect extreme moves in BOTH directions.

Models:
  CGECD Combined (Ours)  Spectral + Traditional → Random Forest
  Spectral Only RF       Spectral features → RF (ablation)
  Traditional RF         Standard features → RF (benchmark / ablation baseline)
  Turbulence RF          Kritzman & Li (2010) → RF
  HAR-RV LR             Corsi (2009) → Logistic Regression
  SMA Vol LR            Rolling volatility → Logistic Regression

Output:
  results/experiment_results.csv   — per-task metrics
  results/bcd_auc_results.csv      — combined BCD-AUC ranking
"""

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from config import Config
from metrics import compute_metrics, bootstrap_auc_ci
from algorithm import (
    load_data, build_spectral_features, build_traditional_features,
    compute_all_targets, CGECDModel, walk_forward_evaluate
)
from benchmarks import (
    prepare_benchmark_features,
    RandomForestModel, LogisticRegressionModel
)


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

W = 108  # output width


def sep(c='═'):
    return c * W


def print_task_table(label, name, task_results, pos_rate):
    """Print per-task results table sorted by AUC."""

    # Best benchmark AUC (role == 'bench' only)
    bench_aucs = [r['auc'] for r in task_results if r['role'] == 'bench']
    bb = max(bench_aucs) if bench_aucs else 0.5

    print(f"\n{sep()}")
    print(f"  {label} — {name}  (positive rate {pos_rate:.1%})")
    print(sep())
    print(f"  {'Model':<35} {'#Feat':>5} {'AUC':>7} {'AvgP':>7}"
          f" {'Prec':>7} {'Rec':>7} {'F1':>7} {'Δ bench':>8}")
    print(f"  {'─' * (W - 4)}")

    for r in sorted(task_results, key=lambda x: x['auc'], reverse=True):
        d = r['auc'] - bb
        if r['role'] == 'ours':
            mk = ' ◆'
        elif r['role'] == 'ablation':
            mk = ' △'
        else:
            mk = '  '
        wn = ' ✓' if r['role'] == 'ours' and d >= 0.04 else ''
        print(
            f"  {r['model']:<35} {r['nf']:>5} {r['auc']:>7.3f} {r['avgp']:>7.3f}"
            f" {r['prec']:>6.1%} {r['rec']:>6.1%} {r['f1']:>6.1%}"
            f" {d:>+8.3f}{mk}{wn}"
        )

    print(f"  {'─' * (W - 4)}")
    print(f"  ◆ = ours   △ = ablation   Δ bench = AUC − best benchmark ({bb:.3f})")
    print(f"  ✓ = CGECD beats best benchmark by ≥ +0.04")


def print_bcd_table(bcd_rows):
    """Print the combined BCD-AUC ranking table."""

    print(f"\n{sep()}")
    print(f"  BALANCED CRISIS DETECTION AUC  (BCD-AUC)")
    print(f"  = √(Rally AUC × Crash AUC)")
    print(f"  Rewards models effective in BOTH directions; penalises one-sided strength")
    print(sep())
    print(f"  {'Model':<35} {'Rally':>7} {'Crash':>7}"
          f" {'BCD-AUC':>8} {'Mean':>7} {'Rank':>5}")
    print(f"  {'─' * (W - 4)}")

    for i, r in enumerate(bcd_rows, 1):
        if r['role'] == 'ours':
            mk = ' ◆'
        elif r['role'] == 'ablation':
            mk = ' △'
        else:
            mk = '  '
        tag = ' ← BEST' if i == 1 else ''
        print(
            f"  {r['model']:<35} {r['up']:>7.3f} {r['down']:>7.3f}"
            f" {r['gmean']:>8.3f} {r['amean']:>7.3f} {'#' + str(i):>5}{mk}{tag}"
        )

    print(f"  {'─' * (W - 4)}")
    print(f"  ◆ = ours   △ = ablation")
    print(f"  BCD-AUC = √(Rally AUC × Crash AUC)     Mean = (Rally + Crash) / 2")

    # Lead over best benchmark
    ours = next((r for r in bcd_rows if r['role'] == 'ours'), None)
    bench_sorted = [r for r in bcd_rows if r['role'] == 'bench']
    bench_sorted.sort(key=lambda x: x['gmean'], reverse=True)

    if ours and bench_sorted:
        b1 = bench_sorted[0]
        lead = ours['gmean'] - b1['gmean']
        print(f"\n  CGECD BCD-AUC:     {ours['gmean']:.3f}")
        print(f"  Best benchmark:    {b1['model']:<25} {b1['gmean']:.3f}")
        print(f"  Lead:              +{lead:.3f}")


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run():
    t0 = datetime.now()

    print(sep())
    print("  CGECD — Correlation Graph Eigenvalue Crisis Detector")
    print("  Two-Task Evaluation: Rally Detection + Crash Detection")
    print(sep())
    print(f"  {t0.strftime('%Y-%m-%d %H:%M:%S')}\n")

    cfg = Config()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. DATA
    # ─────────────────────────────────────────────────────────────────────────
    print("[1/4] Loading data ...")
    prices, returns = load_data(cfg)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/4] Building features ...")

    spectral = build_spectral_features(returns, cfg)
    traditional = build_traditional_features(prices, returns)
    combined = pd.concat([spectral, traditional], axis=1)
    bench_feat = prepare_benchmark_features(prices, returns)

    ns = len(spectral.columns)
    nt = len(traditional.columns)
    nc = len(combined.columns)

    print(f"\n  Feature sets:")
    print(f"    CGECD Combined:  {nc}  ({ns} spectral + {nt} traditional)")
    print(f"    Spectral only:   {ns}")
    print(f"    Traditional:     {nt}")
    for bname, bdf in bench_feat.items():
        print(f"    {bname:<18} {len(bdf.columns)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. TARGETS  (only the two tasks we evaluate)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/4] Computing targets ...")
    all_targets = compute_all_targets(prices)

    tasks = [
        ('up',   'TASK 1 — RALLY DETECTION',
         'Up >3% in 10 days',       all_targets['up_3pct_10d']),
        ('down', 'TASK 2 — CRASH DETECTION',
         'Drawdown >7% in 10 days', all_targets['drawdown_7pct_10d']),
    ]

    for tk, tl, tn, tgt in tasks:
        v = tgt.dropna()
        print(f"  {tn:<32} {v.mean():>6.1%}  ({int(v.sum()):>4} events / {len(v)} days)")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. WALK-FORWARD EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Walk-forward evaluation"
          f" ({cfg.n_splits} folds, {cfg.train_years}yr train,"
          f" {cfg.test_months}mo test, {cfg.gap_days}d gap) ...\n")

    # (label, features, model_class, role, n_features)
    # role:  'ours' | 'ablation' | 'bench'
    methods = [
        ("CGECD Combined (Ours)",
         combined,                CGECDModel,              'ours',     nc),
        ("Spectral Only RF",
         spectral,                RandomForestModel,       'ablation', ns),
        ("Traditional RF",
         traditional,             RandomForestModel,       'bench',    nt),
        ("Turbulence RF",
         bench_feat['turbulence'], RandomForestModel,      'bench',
         len(bench_feat['turbulence'].columns)),
        ("HAR-RV LR",
         bench_feat['har_rv'],    LogisticRegressionModel, 'bench',
         len(bench_feat['har_rv'].columns)),
        ("SMA Vol LR",
         bench_feat['sma_vol'],   LogisticRegressionModel, 'bench',
         len(bench_feat['sma_vol'].columns)),
    ]

    results = {}   # task_key -> [result dicts]
    auc_map = {}   # model_name -> {'up': x, 'down': y, 'role': str}

    for tk, tl, tn, tgt in tasks:
        pr = tgt.dropna().mean()
        print(f"  ── {tl}: {tn}  (positive rate {pr:.1%}) ──")

        task_res = []
        for ml, mf, mc, mr, mn in methods:
            print(f"    {ml:<35}", end="  ", flush=True)

            try:
                r = walk_forward_evaluate(mf, tgt, mc, cfg)
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            if 'error' not in r:
                m = r['metrics']
                task_res.append(dict(
                    model=ml, nf=mn, role=mr,
                    auc=m.auc_roc, avgp=m.avg_precision,
                    prec=m.precision, rec=m.recall, f1=m.f1,
                ))
                print(f"AUC={m.auc_roc:.3f}  AvgP={m.avg_precision:.3f}")
                auc_map.setdefault(ml, {'role': mr})[tk] = m.auc_roc
            else:
                print(f"FAILED: {r['error']}")

        results[tk] = task_res
        print()

    # ═════════════════════════════════════════════════════════════════════════
    # RESULTS — PER-TASK TABLES
    # ═════════════════════════════════════════════════════════════════════════

    for tk, tl, tn, tgt in tasks:
        pr = tgt.dropna().mean()
        print_task_table(tl, tn, results[tk], pr)

    # ═════════════════════════════════════════════════════════════════════════
    # RESULTS — BCD-AUC COMBINED METRIC
    # ═════════════════════════════════════════════════════════════════════════

    bcd_rows = []
    for mn, info in auc_map.items():
        u = info.get('up', 0.5)
        d = info.get('down', 0.5)
        bcd_rows.append(dict(
            model=mn, up=u, down=d,
            gmean=np.sqrt(u * d),
            amean=(u + d) / 2,
            role=info['role'],
        ))
    bcd_rows.sort(key=lambda x: x['gmean'], reverse=True)

    print_bcd_table(bcd_rows)

    # ═════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═════════════════════════════════════════════════════════════════════════

    print(f"\n{sep()}")
    print(f"  FINAL SUMMARY")
    print(sep())

    # Per-task verdicts
    for tk, tl, tn, tgt in tasks:
        tr = results[tk]
        oa = next((r['auc'] for r in tr if r['role'] == 'ours'), 0)
        bench_list = [r for r in tr if r['role'] == 'bench']
        if not bench_list:
            continue
        bb = max(bench_list, key=lambda x: x['auc'])
        d = oa - bb['auc']
        if d >= 0.04:
            st = "✓ WIN  (≥ +0.04)"
        elif d >= -0.02:
            st = "≈ COMPETITIVE"
        else:
            st = "✗ BEHIND"
        tag = "Rally" if tk == 'up' else "Crash"
        print(f"  {tag:<7} CGECD {oa:.3f}  vs  {bb['model']:<20} {bb['auc']:.3f}"
              f"   Δ = {d:+.3f}   {st}")

    # BCD-AUC rank
    ours_rank = next(
        (i + 1 for i, r in enumerate(bcd_rows) if r['role'] == 'ours'), '?'
    )
    ours_bcd = next(
        (r['gmean'] for r in bcd_rows if r['role'] == 'ours'), 0
    )
    print(f"\n  BCD-AUC: #{ours_rank}  ({ours_bcd:.3f})")

    # Spectral feature contribution (ablation analysis)
    ou = auc_map.get("CGECD Combined (Ours)", {}).get('up', 0)
    od = auc_map.get("CGECD Combined (Ours)", {}).get('down', 0)
    tu = auc_map.get("Traditional RF", {}).get('up', 0)
    td = auc_map.get("Traditional RF", {}).get('down', 0)
    du = ou - tu
    dd = od - td

    print(f"\n  Spectral feature contribution (CGECD − Traditional RF):")
    print(f"    Rally:   {du:+.3f} AUC")
    print(f"    Crash:   {dd:+.3f} AUC")
    print(f"    Mean:    {(du + dd) / 2:+.3f} AUC")

    # ═════════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═════════════════════════════════════════════════════════════════════════

    out = cfg.output_dir

    # Per-task detailed results
    rows = []
    for tk, tl, tn, tgt in tasks:
        for r in results[tk]:
            rows.append(dict(
                Task=tn, Model=r['model'], N_Features=r['nf'],
                Role=r['role'], AUC_ROC=r['auc'], Avg_Precision=r['avgp'],
                Precision=r['prec'], Recall=r['rec'], F1=r['f1'],
            ))
    results_df = pd.DataFrame(rows)
    results_df.to_csv(out / 'experiment_results.csv', index=False)

    # BCD-AUC ranking
    bcd_save = [dict(
        Model=r['model'], Rally_AUC=r['up'], Crash_AUC=r['down'],
        BCD_AUC=r['gmean'], Mean_AUC=r['amean'], Role=r['role'],
    ) for r in bcd_rows]
    pd.DataFrame(bcd_save).to_csv(out / 'bcd_auc_results.csv', index=False)

    elapsed = datetime.now() - t0
    print(f"\n  Saved to {out}/")
    print(f"    experiment_results.csv  — per-task metrics (6 models × 2 tasks)")
    print(f"    bcd_auc_results.csv     — BCD-AUC combined ranking")
    print(f"\n  Runtime: {elapsed}")
    print(sep())

    return results_df


if __name__ == "__main__":
    run()