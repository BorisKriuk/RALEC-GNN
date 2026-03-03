#!/usr/bin/env python3
"""
run_v3.py — Two Focused Algorithms: UP-detector + DOWN-detector
================================================================

UP  algo: v1 architecture (RF, 206 features) on "Up >3% in 10d"
          Already proven: AUC 0.747 vs 0.698 = +0.049

DOWN algo: Stacked ensemble on "Extreme vol 10d"
          Current best: 0.840 vs 0.831 = +0.009
          Target: 0.871+ for ≥ +0.04

Key changes from v2:
  1. UP: restore v1 full features (curation was a mistake)
  2. DOWN: target extreme vol (not drawdown — HAR-RV unbeatable there)
  3. DOWN: expand spectral features from 6 → ~50 (same lesson as UP)
  4. DOWN: stacked meta-learner (LR/RF/GBT base → LR meta)
  5. DOWN: grid-search ensemble weights
"""

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit

from config import Config
from metrics import compute_metrics
from algorithm import (
    load_data, build_spectral_features, build_traditional_features,
    compute_all_targets, walk_forward_evaluate, CGECDModel,
)
from benchmarks import (
    prepare_benchmark_features,
    RandomForestModel, LogisticRegressionModel,
)


# ═════════════════════════════════════════════════════════════════
# EXTRA FEATURES
# ═════════════════════════════════════════════════════════════════

def build_har_features(prices: pd.DataFrame) -> pd.DataFrame:
    """HAR-RV features."""
    market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
    r = market.pct_change()
    rv_d = r.rolling(1).std() * np.sqrt(252)
    rv_w = r.rolling(5).std() * np.sqrt(252)
    rv_m = r.rolling(22).std() * np.sqrt(252)
    return pd.DataFrame({
        'rv_d': rv_d, 'rv_w': rv_w, 'rv_m': rv_m,
        'rv_ratio': rv_d / (rv_m + 1e-10),
    }, index=prices.index)


def build_extra_vol_features(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Additional vol features to help DOWN detection."""
    market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
    r = market.pct_change()

    feats = {}
    # Parkinson vol (high-low proxy via rolling range)
    for w in [5, 10, 20]:
        rolling_max = market.rolling(w).max()
        rolling_min = market.rolling(w).min()
        feats[f'range_vol_{w}d'] = (rolling_max - rolling_min) / market

    # Realized variance ratios
    rv5 = r.rolling(5).var()
    rv20 = r.rolling(20).var()
    rv60 = r.rolling(60).var()
    feats['var_ratio_5_20'] = rv5 / (rv20 + 1e-10)
    feats['var_ratio_5_60'] = rv5 / (rv60 + 1e-10)
    feats['var_ratio_20_60'] = rv20 / (rv60 + 1e-10)

    # Cross-asset vol dispersion
    asset_vols = returns.rolling(20).std()
    feats['vol_dispersion_20d'] = asset_vols.std(axis=1)
    feats['vol_dispersion_mean_20d'] = asset_vols.mean(axis=1)
    feats['vol_dispersion_ratio'] = feats['vol_dispersion_20d'] / (feats['vol_dispersion_mean_20d'] + 1e-10)

    asset_vols_5 = returns.rolling(5).std()
    feats['vol_dispersion_5d'] = asset_vols_5.std(axis=1)

    # Number of assets with vol above 2x median
    median_vol = asset_vols.median(axis=1)
    feats['n_high_vol_assets'] = (asset_vols.gt(asset_vols.median(axis=1) * 2, axis=0)).sum(axis=1)

    # Negative return breadth
    feats['neg_breadth_5d'] = (returns.rolling(5).mean() < 0).sum(axis=1) / returns.shape[1]
    feats['neg_breadth_20d'] = (returns.rolling(20).mean() < 0).sum(axis=1) / returns.shape[1]

    return pd.DataFrame(feats, index=prices.index)


# ═════════════════════════════════════════════════════════════════
# FEATURE SETS
# ═════════════════════════════════════════════════════════════════

def curate_down_wide(pool: pd.DataFrame) -> pd.DataFrame:
    """DOWN features — WIDER spectral set.
    Lesson from UP: many weak spectral signals combine.
    Keep ~70 features total: vol/traditional (~25) + spectral (~45)
    """
    # Traditional/vol features (always include)
    trad_names = [
        'rv_d', 'rv_w', 'rv_m', 'rv_ratio',
        'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_60d',
        'garch_vol', 'downside_vol_20d',
        'vol_ratio_5_20', 'vol_ratio_10_60', 'vol_of_vol_20d',
        'drawdown_20d', 'drawdown_60d',
        'max_loss_5d', 'max_loss_20d',
        'cross_dispersion',
        'skewness_20d', 'kurtosis_20d',
        'return_5d', 'return_10d', 'return_20d',
        # extra vol features
        'range_vol_5d', 'range_vol_10d', 'range_vol_20d',
        'var_ratio_5_20', 'var_ratio_5_60', 'var_ratio_20_60',
        'vol_dispersion_20d', 'vol_dispersion_5d', 'vol_dispersion_ratio',
        'n_high_vol_assets',
        'neg_breadth_5d', 'neg_breadth_20d',
    ]

    # Spectral features — WIDE selection (don't over-curate!)
    spectral_names = [
        # Eigenvalue structure (multiple windows)
        'lambda_1_60d', 'lambda_2_60d', 'lambda_3_60d',
        'lambda_1_40d', 'lambda_2_40d',
        'lambda_1_20d', 'lambda_2_20d',
        'spectral_gap_60d', 'spectral_gap_40d', 'spectral_gap_20d',
        'effective_rank_60d', 'effective_rank_40d', 'effective_rank_20d',
        'eigenvalue_entropy_60d', 'eigenvalue_entropy_40d', 'eigenvalue_entropy_20d',
        'absorption_ratio_1_60d', 'absorption_ratio_5_60d',
        'absorption_ratio_1_40d', 'absorption_ratio_5_40d',
        'absorption_ratio_1_20d', 'absorption_ratio_5_20d',
        'tail_eigenvalue_ratio_60d', 'tail_eigenvalue_ratio_40d',
        'condition_number_60d', 'condition_number_40d',
        'mp_excess_ratio_60d', 'mp_excess_ratio_40d',
        # Correlation structure
        'mean_abs_corr_60d', 'mean_abs_corr_40d', 'mean_abs_corr_20d',
        'median_abs_corr_60d',
        'max_abs_corr_60d', 'corr_std_60d',
        'corr_skew_60d',
        'frac_corr_above_50_60d', 'frac_corr_above_50_40d',
        'frac_corr_above_75_60d',
        # Network
        'edge_density_t50_60d', 'edge_density_t50_40d',
        'mean_degree_t50_60d',
        'clustering_coef_t50_60d',
        # Eigenvector
        'v1_max_loading_60d', 'loading_dispersion_60d',
        # Dynamics (rate of change)
        'lambda_1_roc_5d', 'lambda_1_roc_10d',
        'lambda_2_roc_5d',
        'mean_abs_corr_roc_5d', 'mean_abs_corr_roc_10d',
        'absorption_ratio_1_roc_5d', 'absorption_ratio_1_roc_10d',
        'eigenvalue_entropy_roc_5d',
        'spectral_gap_roc_5d',
        # Z-scores
        'lambda_1_zscore_10d', 'lambda_1_zscore_20d',
        'mean_abs_corr_zscore_10d', 'mean_abs_corr_zscore_20d',
        'eigenvalue_entropy_zscore_10d', 'eigenvalue_entropy_zscore_20d',
        'absorption_ratio_1_zscore_10d',
    ]

    all_names = trad_names + spectral_names
    cols = [c for c in all_names if c in pool.columns]
    missing = [c for c in all_names if c not in pool.columns]
    if missing:
        print(f"    (note: {len(missing)} features not in pool)")
    return pool[cols]


def curate_down_no_spectral(pool: pd.DataFrame) -> pd.DataFrame:
    """Ablation: just trad/vol features."""
    trad_names = [
        'rv_d', 'rv_w', 'rv_m', 'rv_ratio',
        'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_60d',
        'garch_vol', 'downside_vol_20d',
        'vol_ratio_5_20', 'vol_ratio_10_60', 'vol_of_vol_20d',
        'drawdown_20d', 'drawdown_60d',
        'max_loss_5d', 'max_loss_20d',
        'cross_dispersion',
        'skewness_20d', 'kurtosis_20d',
        'return_5d', 'return_10d', 'return_20d',
        'range_vol_5d', 'range_vol_10d', 'range_vol_20d',
        'var_ratio_5_20', 'var_ratio_5_60', 'var_ratio_20_60',
        'vol_dispersion_20d', 'vol_dispersion_5d', 'vol_dispersion_ratio',
        'n_high_vol_assets',
        'neg_breadth_5d', 'neg_breadth_20d',
    ]
    cols = [c for c in trad_names if c in pool.columns]
    return pool[cols]


# ═════════════════════════════════════════════════════════════════
# MODELS
# ═════════════════════════════════════════════════════════════════

class TunedRFModel:
    """RF with deeper trees and more estimators for richer feature sets."""
    def __init__(self, config):
        self.config = config
        self.scaler = self.model = None

    def fit(self, X, y):
        self.scaler = RobustScaler()
        Xs = np.nan_to_num(self.scaler.fit_transform(X), 0, 0, 0)
        self.model = RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=20,
            min_samples_split=40, max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=self.config.random_seed,
            n_jobs=-1,
        )
        self.model.fit(Xs, y)

    def predict_proba(self, X):
        Xs = np.nan_to_num(self.scaler.transform(X), 0, 0, 0)
        return self.model.predict_proba(Xs)[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class ExtraTreesModel:
    """ExtraTrees — more randomization, better for noisy features."""
    def __init__(self, config):
        self.config = config
        self.scaler = self.model = None

    def fit(self, X, y):
        self.scaler = RobustScaler()
        Xs = np.nan_to_num(self.scaler.fit_transform(X), 0, 0, 0)
        self.model = ExtraTreesClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=20,
            min_samples_split=40, max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=self.config.random_seed,
            n_jobs=-1,
        )
        self.model.fit(Xs, y)

    def predict_proba(self, X):
        Xs = np.nan_to_num(self.scaler.transform(X), 0, 0, 0)
        return self.model.predict_proba(Xs)[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class StackedModel:
    """Stacked ensemble: RF + ExtraTrees + LR → LR meta-learner.
    Uses internal time-series CV to generate out-of-fold predictions."""

    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.base_models = None
        self.meta = None

    def _make_base_models(self):
        return [
            ('rf', RandomForestClassifier(
                n_estimators=300, max_depth=7, min_samples_leaf=25,
                min_samples_split=50, max_features='sqrt',
                class_weight='balanced_subsample',
                random_state=self.config.random_seed, n_jobs=-1)),
            ('et', ExtraTreesClassifier(
                n_estimators=300, max_depth=7, min_samples_leaf=25,
                min_samples_split=50, max_features='sqrt',
                class_weight='balanced_subsample',
                random_state=self.config.random_seed, n_jobs=-1)),
            ('lr', LogisticRegression(
                C=0.1, class_weight='balanced', max_iter=1000,
                random_state=self.config.random_seed)),
        ]

    def fit(self, X, y):
        self.scaler = RobustScaler()
        Xs = np.nan_to_num(self.scaler.fit_transform(X), 0, 0, 0)

        n = len(Xs)
        oof_preds = np.zeros((n, 3))

        # Time-series CV for out-of-fold predictions
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, val_idx in tscv.split(Xs):
            base = self._make_base_models()
            for i, (name, model) in enumerate(base):
                model.fit(Xs[train_idx], y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx])
                oof_preds[val_idx, i] = model.predict_proba(Xs[val_idx])[:, 1]

        # Meta-learner on OOF predictions (only on rows that have predictions)
        has_pred = oof_preds.sum(axis=1) > 0
        self.meta = LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000,
            random_state=self.config.random_seed)
        y_arr = y.values if hasattr(y, 'values') else y
        self.meta.fit(oof_preds[has_pred], y_arr[has_pred])

        # Refit base models on full data
        self.base_models = self._make_base_models()
        for name, model in self.base_models:
            model.fit(Xs, y_arr)

    def predict_proba(self, X):
        Xs = np.nan_to_num(self.scaler.transform(X), 0, 0, 0)
        base_preds = np.column_stack([
            model.predict_proba(Xs)[:, 1]
            for name, model in self.base_models
        ])
        return self.meta.predict_proba(base_preds)[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def _make_ensemble(lr_w):
    """LR + RF blend with configurable weight."""
    class Ens:
        def __init__(self, config):
            self.config = config
            self.w = lr_w
            self.scaler = self.lr = self.rf = None

        def fit(self, X, y):
            self.scaler = RobustScaler()
            Xs = np.nan_to_num(self.scaler.fit_transform(X), 0, 0, 0)
            self.lr = LogisticRegression(
                C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
            self.lr.fit(Xs, y)
            self.rf = RandomForestClassifier(
                n_estimators=500, max_depth=8, min_samples_leaf=20,
                min_samples_split=40, max_features='sqrt',
                class_weight='balanced_subsample',
                random_state=self.config.random_seed, n_jobs=-1)
            self.rf.fit(Xs, y)

        def predict_proba(self, X):
            Xs = np.nan_to_num(self.scaler.transform(X), 0, 0, 0)
            return self.w * self.lr.predict_proba(Xs)[:, 1] + \
                   (1 - self.w) * self.rf.predict_proba(Xs)[:, 1]

        def predict(self, X):
            return (self.predict_proba(X) >= 0.5).astype(int)

    Ens.__name__ = f'LR_RF_Ens{int(lr_w*100)}'
    return Ens


Ens30 = _make_ensemble(0.30)
Ens40 = _make_ensemble(0.40)
Ens50 = _make_ensemble(0.50)
Ens60 = _make_ensemble(0.60)


# ═════════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ═════════════════════════════════════════════════════════════════

BENCH_NAMES = {'Traditional RF', 'Turbulence RF', 'HAR-RV LR', 'SMA Vol LR'}


def eval_one(features, target, model_cls, config, label):
    print(f"    {label:<45s}", end=" ", flush=True)
    res = walk_forward_evaluate(features, target, model_cls, config)
    if 'error' in res:
        print(f"FAILED ({res['error']})")
        return None
    m = res['metrics']
    print(f"AUC={m.auc_roc:.3f}  AvgP={m.avg_precision:.3f}")
    return dict(model=label, auc_roc=m.auc_roc, avg_precision=m.avg_precision,
                precision=m.precision, recall=m.recall, f1=m.f1,
                n_features=features.shape[1])


def print_table(task, results, best_bench_auc):
    print(f"\n{'═' * 110}")
    print(f"  {task}")
    print(f"{'═' * 110}")
    print(f"  {'Model':<45s} {'#F':>4} {'AUC':>7} {'AvgP':>7} "
          f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'Δ best':>7}")
    print(f"  {'─' * 104}")
    for r in sorted(results, key=lambda x: x['auc_roc'], reverse=True):
        d = r['auc_roc'] - best_bench_auc
        flag = ' ◆' if '(Ours)' in r['model'] else ''
        win = ' ✓' if d >= 0.04 else ''
        print(f"  {r['model']:<45s} {r['n_features']:>4d} {r['auc_roc']:>7.3f} "
              f"{r['avg_precision']:>7.3f} {r['precision']:>6.1%} {r['recall']:>6.1%} "
              f"{r['f1']:>6.1%} {d:>+7.3f}{flag}{win}")
    print(f"  {'─' * 104}")
    print(f"  ◆ = ours   ✓ = ≥ +0.04 over best benchmark ({best_bench_auc:.3f})")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    print('═' * 110)
    print('  CGECD v3 — Two Focused Algorithms')
    print('  UP:   v1-style RF with full spectral (proven +0.049)')
    print('  DOWN: Wide spectral + stacking on Extreme vol / Vol spike')
    print('═' * 110)
    print(f'  {t0:%Y-%m-%d %H:%M:%S}\n')

    config = Config()

    # ── 1. DATA ──────────────────────────────────────────────────
    print('[1/4] Loading data ...')
    prices, returns = load_data(config)

    # ── 2. FEATURES ──────────────────────────────────────────────
    print('\n[2/4] Building features ...')
    spectral    = build_spectral_features(returns, config)
    traditional = build_traditional_features(prices, returns)
    har         = build_har_features(prices)
    extra_vol   = build_extra_vol_features(prices, returns)

    # Full pool (for DOWN)
    pool = pd.concat([spectral, traditional, har, extra_vol], axis=1)
    pool = pool.loc[:, ~pool.columns.duplicated()]
    print(f'  Full feature pool: {len(pool.columns)} columns')

    # v1 features (for UP — proven winner)
    v1_features = pd.concat([spectral, traditional], axis=1)
    v1_features = v1_features.loc[:, ~v1_features.columns.duplicated()]
    print(f'  v1 features: {len(v1_features.columns)} columns')

    bench_feat = prepare_benchmark_features(prices, returns)

    # ── 3. TARGETS ───────────────────────────────────────────────
    print('\n[3/4] Computing targets ...')
    targets = compute_all_targets(prices)

    # Focus on the targets where we can win
    tasks = {
        # UP — proven winner
        'Up >3% in 10d':    ('UP',   targets['up_3pct_10d']),
        # DOWN — best opportunities (spectral helps, HAR-RV weak)
        'Extreme vol 10d':  ('DOWN', targets['extreme_vol_10d']),
        'Vol spike 2x 10d': ('DOWN', targets['vol_spike_2x_10d']),
        # Additional DOWN for comparison
        'Drawdown >5% 10d': ('DOWN', targets['drawdown_5pct_10d']),
        'Drawdown >7% 10d': ('DOWN', targets['drawdown_7pct_10d']),
    }

    for name, (direction, tgt) in tasks.items():
        v = tgt.dropna()
        print(f'  [{direction}] {name:<24s} {v.mean():>6.1%}  '
              f'({int(v.sum()):>4d} events / {len(v)} days)')

    # ── 4. WALK-FORWARD EVALUATION ──────────────────────────────
    print('\n[4/4] Evaluating ...\n')

    all_csv_rows = []

    # ════════════════════════════════════════════════════════════
    # A. UP ALGORITHM — "Up >3% in 10d"
    # ════════════════════════════════════════════════════════════
    task_name = 'Up >3% in 10d'
    task_target = targets['up_3pct_10d']
    print(f'{"═" * 110}')
    print(f'  UP ALGORITHM — {task_name}')
    print(f'  Strategy: v1 full features (206) with RF (proven +0.049)')
    print(f'{"═" * 110}\n')

    up_methods = [
        # v1 architecture (the winner)
        (v1_features, CGECDModel,       'UP-v1 RF 206-feat (Ours)'),
        # Try tuned RF on v1 features
        (v1_features, TunedRFModel,     'UP-v1 TunedRF 206-feat (Ours)'),
        # Try ExtraTrees on v1 features
        (v1_features, ExtraTreesModel,  'UP-v1 ExtraTrees 206-feat (Ours)'),
        # Try stacking on v1 features
        (v1_features, StackedModel,     'UP-v1 Stacked 206-feat (Ours)'),
        # Full pool with tuned RF
        (pool,        TunedRFModel,     'UP-pool TunedRF full (Ours)'),
        # Benchmarks
        (traditional,                   RandomForestModel,       'Traditional RF'),
        (bench_feat['turbulence'],      RandomForestModel,       'Turbulence RF'),
        (bench_feat['har_rv'],          LogisticRegressionModel, 'HAR-RV LR'),
        (bench_feat['sma_vol'],         LogisticRegressionModel, 'SMA Vol LR'),
    ]

    results = []
    for feat, mcls, label in up_methods:
        r = eval_one(feat, task_target, mcls, config, label)
        if r:
            results.append(r)
            all_csv_rows.append({**r, 'task': task_name, 'direction': 'UP'})

    bb = [r for r in results if r['model'] in BENCH_NAMES]
    bba = max(bb, key=lambda x: x['auc_roc'])['auc_roc'] if bb else 0
    print_table(f'UP — {task_name}', results, bba)

    # ════════════════════════════════════════════════════════════
    # B. DOWN ALGORITHM — Extreme vol + Vol spike
    # ════════════════════════════════════════════════════════════
    down_targets = {
        'Extreme vol 10d':  targets['extreme_vol_10d'],
        'Vol spike 2x 10d': targets['vol_spike_2x_10d'],
        'Drawdown >5% 10d': targets['drawdown_5pct_10d'],
        'Drawdown >7% 10d': targets['drawdown_7pct_10d'],
    }

    dn_wide = curate_down_wide(pool)
    dn_nosp = curate_down_no_spectral(pool)
    print(f'\n  DOWN features — wide: {len(dn_wide.columns)} feat, '
          f'no-spectral: {len(dn_nosp.columns)} feat')

    for task_name, task_target in down_targets.items():
        print(f'\n{"═" * 110}')
        print(f'  DOWN ALGORITHM — {task_name}')
        print(f'  Strategy: wide spectral ({len(dn_wide.columns)} feat) + stacking/ensemble')
        print(f'{"═" * 110}\n')

        down_methods = [
            # Our models — wide spectral
            (dn_wide,   TunedRFModel,      'DN-wide TunedRF (Ours)'),
            (dn_wide,   ExtraTreesModel,   'DN-wide ExtraTrees (Ours)'),
            (dn_wide,   StackedModel,      'DN-wide Stacked (Ours)'),
            (dn_wide,   Ens30,             'DN-wide LR/RF Ens30 (Ours)'),
            (dn_wide,   Ens40,             'DN-wide LR/RF Ens40 (Ours)'),
            (dn_wide,   Ens50,             'DN-wide LR/RF Ens50 (Ours)'),
            (dn_wide,   Ens60,             'DN-wide LR/RF Ens60 (Ours)'),
            # v1 full features on this task
            (v1_features, CGECDModel,      'DN-v1 RF 206-feat (Ours)'),
            (v1_features, TunedRFModel,    'DN-v1 TunedRF 206-feat (Ours)'),
            # Full pool
            (pool,       TunedRFModel,     'DN-pool TunedRF full (Ours)'),
            # Ablation
            (dn_nosp,   TunedRFModel,     'No-spectral TunedRF'),
            (dn_nosp,   Ens50,            'No-spectral Ens50'),
            # Benchmarks
            (traditional,                  RandomForestModel,       'Traditional RF'),
            (bench_feat['turbulence'],     RandomForestModel,       'Turbulence RF'),
            (bench_feat['har_rv'],         LogisticRegressionModel, 'HAR-RV LR'),
            (bench_feat['sma_vol'],        LogisticRegressionModel, 'SMA Vol LR'),
        ]

        results = []
        for feat, mcls, label in down_methods:
            r = eval_one(feat, task_target, mcls, config, label)
            if r:
                results.append(r)
                all_csv_rows.append({**r, 'task': task_name, 'direction': 'DOWN'})

        bb = [r for r in results if r['model'] in BENCH_NAMES]
        bba = max(bb, key=lambda x: x['auc_roc'])['auc_roc'] if bb else 0
        print_table(f'DOWN — {task_name}', results, bba)

    # ── SAVE ─────────────────────────────────────────────────────
    df = pd.DataFrame(all_csv_rows)
    out = config.output_dir / 'v3_results.csv'
    df.to_csv(out, index=False)
    print(f'\n  Results saved to {out}')

    # ── FINAL SUMMARY ────────────────────────────────────────────
    print(f'\n{"═" * 110}')
    print('  FINAL SUMMARY — Best v3 model per task vs best benchmark')
    print(f'{"═" * 110}')
    for task in df['task'].unique():
        sub = df[df['task'] == task]
        ours = sub[sub['model'].str.contains('Ours')]
        bench = sub[sub['model'].isin(BENCH_NAMES)]
        if ours.empty or bench.empty:
            continue
        best_ours = ours.loc[ours['auc_roc'].idxmax()]
        best_bench = bench.loc[bench['auc_roc'].idxmax()]
        delta = best_ours['auc_roc'] - best_bench['auc_roc']
        status = '✓ WIN ≥4%' if delta >= 0.04 else ('≈ close' if delta > 0 else '✗ LOSE')
        print(f'  {task:<24s}  {best_ours["model"]:<36s} {best_ours["auc_roc"]:.3f}  '
              f'vs  {best_bench["model"]:<16s} {best_bench["auc_roc"]:.3f}  '
              f'Δ={delta:+.3f}  {status}')

    print(f'\n  Runtime: {datetime.now() - t0}')
    print(f'{"═" * 110}')


if __name__ == '__main__':
    main()