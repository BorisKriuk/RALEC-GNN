#!/usr/bin/env python3
"""
CGECD Comprehensive Benchmarks
==============================

Compare Boris's Correlation Graph Eigenvalue Crisis Detector against:
1. Naive baselines (random, always-positive, historical rate)
2. Traditional ML models (Logistic Regression, SVM, Gradient Boosting)
3. Standard financial indicators (VIX-based, momentum-based)
4. Statistical tests for significance

Generates publication-quality visualizations and metrics.
"""

import os
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')
np.random.seed(42)

load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY") or os.getenv("API_KEY")

OUTPUT_DIR = Path("cgecd_benchmarks")
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache")


# =============================================================================
# DATA LOADING (reuse from algorithm-boris.py)
# =============================================================================
class DataLoader:
    BASE_URL = "https://eodhd.com/api"

    def __init__(self, api_key: str):
        self.api_key = api_key
        import requests
        self.session = requests.Session()

    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_file = symbol.replace('.', '_').replace('/', '_')
        cache_path = CACHE_DIR / f"{cache_file}.pkl"

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass

        try:
            params = {
                'api_token': self.api_key,
                'fmt': 'json',
                'from': start_date,
                'to': end_date
            }
            url = f"{self.BASE_URL}/eod/{symbol}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()

            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)

            return df
        except Exception as e:
            return pd.DataFrame()


def load_multi_asset_data(years: int = 15) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load diverse asset universe for correlation analysis."""
    loader = DataLoader(API_KEY)

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')

    symbols = {
        'SPY.US': 'SP500', 'QQQ.US': 'Nasdaq100', 'IWM.US': 'Russell2000',
        'XLF.US': 'Financials', 'XLE.US': 'Energy', 'XLK.US': 'Technology',
        'XLV.US': 'Healthcare', 'XLU.US': 'Utilities', 'XLP.US': 'ConsumerStaples',
        'XLY.US': 'ConsumerDisc', 'XLI.US': 'Industrials', 'XLB.US': 'Materials',
        'XLRE.US': 'RealEstate', 'EFA.US': 'DevIntl', 'EEM.US': 'EmergingMkts',
        'VGK.US': 'Europe', 'EWJ.US': 'Japan', 'TLT.US': 'LongTreasury',
        'IEF.US': 'IntermTreasury', 'LQD.US': 'InvGradeCorp', 'HYG.US': 'HighYield',
        'GLD.US': 'Gold', 'USO.US': 'Oil', 'UUP.US': 'USDollar', 'VNQ.US': 'REITs',
    }

    print(f"Loading {len(symbols)} assets...")

    data_dict = {}
    for symbol, name in symbols.items():
        df = loader.get_data(symbol, start_date, end_date)
        if not df.empty and 'adjusted_close' in df.columns:
            data_dict[name] = df['adjusted_close']

    prices = pd.DataFrame(data_dict)
    prices = prices.dropna(how='all').ffill(limit=5).dropna()

    print(f"Combined dataset: {len(prices)} days, {len(prices.columns)} assets")
    return prices, {v: k for k, v in symbols.items()}


# =============================================================================
# SPECTRAL FEATURE EXTRACTION (from algorithm-boris.py)
# =============================================================================
class SpectralFeatureExtractor:
    """Extract features from correlation matrix eigenvalue decomposition."""

    def __init__(self, n_assets: int):
        self.n_assets = n_assets

    def extract_all_features(self, corr_matrix: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive spectral and topology features."""
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)

        features = {}

        try:
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues, 1e-10)
        except:
            return self._default_features()

        n = len(eigenvalues)
        total_var = np.sum(eigenvalues)

        # Primary eigenvalue features
        features['lambda_1'] = eigenvalues[0]
        features['lambda_1_ratio'] = eigenvalues[0] / total_var
        features['lambda_2'] = eigenvalues[1] if n > 1 else 0
        features['spectral_gap'] = eigenvalues[0] / (eigenvalues[1] + 1e-10)

        # Absorption ratios
        for k in [1, 3, 5]:
            features[f'absorption_ratio_{k}'] = np.sum(eigenvalues[:min(k, n)]) / total_var

        # Entropy and effective rank
        normalized_eig = eigenvalues / total_var
        entropy = -np.sum(normalized_eig * np.log(normalized_eig + 1e-10))
        features['eigenvalue_entropy'] = entropy
        features['effective_rank'] = np.exp(entropy)

        # Higher moments
        features['eigenvalue_std'] = np.std(eigenvalues)
        features['eigenvalue_skew'] = stats.skew(eigenvalues)
        features['eigenvalue_kurt'] = stats.kurtosis(eigenvalues)

        # Graph topology features
        upper_tri = corr_matrix[np.triu_indices(self.n_assets, k=1)]
        features['mean_abs_corr'] = np.mean(np.abs(upper_tri))
        features['max_abs_corr'] = np.max(np.abs(upper_tri))
        features['corr_std'] = np.std(upper_tri)
        features['frac_corr_above_50'] = np.mean(np.abs(upper_tri) > 0.5)
        features['frac_corr_above_70'] = np.mean(np.abs(upper_tri) > 0.7)

        # Edge density at thresholds
        for thresh in [0.3, 0.5, 0.7]:
            adj = (np.abs(corr_matrix) > thresh).astype(float)
            np.fill_diagonal(adj, 0)
            n_edges = np.sum(adj) / 2
            max_edges = self.n_assets * (self.n_assets - 1) / 2
            features[f'edge_density_t{int(thresh*100)}'] = n_edges / max_edges

        return features

    def _default_features(self):
        return {
            'lambda_1': 1.0, 'lambda_1_ratio': 1.0/self.n_assets,
            'lambda_2': 0.0, 'spectral_gap': 1.0,
            'absorption_ratio_1': 1.0/self.n_assets,
            'absorption_ratio_3': 3.0/self.n_assets,
            'absorption_ratio_5': 5.0/self.n_assets,
            'eigenvalue_entropy': np.log(self.n_assets),
            'effective_rank': self.n_assets,
            'eigenvalue_std': 0.0, 'eigenvalue_skew': 0.0, 'eigenvalue_kurt': 0.0,
            'mean_abs_corr': 0.0, 'max_abs_corr': 0.0, 'corr_std': 0.0,
            'frac_corr_above_50': 0.0, 'frac_corr_above_70': 0.0,
            'edge_density_t30': 0.0, 'edge_density_t50': 0.0, 'edge_density_t70': 0.0,
        }


# =============================================================================
# TRADITIONAL FEATURE BUILDERS
# =============================================================================
class TraditionalFeatureBuilder:
    """Build standard technical analysis features."""

    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
        self.market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
        self.returns = self.market.pct_change()

    def compute_features(self) -> pd.DataFrame:
        features = pd.DataFrame(index=self.prices.index)

        # Returns
        for window in [1, 5, 10, 20, 60]:
            features[f'return_{window}d'] = self.market.pct_change(window)

        # Volatility
        for window in [5, 10, 20, 60]:
            features[f'volatility_{window}d'] = self.returns.rolling(window).std() * np.sqrt(252)

        features['vol_ratio_5_20'] = features['volatility_5d'] / (features['volatility_20d'] + 1e-8)

        # Momentum
        for window in [10, 20, 50]:
            sma = self.market.rolling(window).mean()
            features[f'price_to_sma_{window}'] = self.market / sma - 1

        # RSI
        delta = self.market.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        features['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

        # Drawdown
        for window in [20, 60]:
            rolling_max = self.market.rolling(window).max()
            features[f'drawdown_{window}d'] = (self.market - rolling_max) / rolling_max

        # Higher moments
        for window in [20, 60]:
            features[f'skewness_{window}d'] = self.returns.rolling(window).skew()
            features[f'kurtosis_{window}d'] = self.returns.rolling(window).kurt()

        return features


class VIXBasedPredictor:
    """Simple VIX-based crisis prediction baseline."""

    def __init__(self, prices: pd.DataFrame):
        self.market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
        self.returns = self.market.pct_change()

    def compute_features(self) -> pd.DataFrame:
        """Use realized volatility as VIX proxy."""
        features = pd.DataFrame(index=self.market.index)

        # Realized vol (VIX proxy)
        vol_20 = self.returns.rolling(20).std() * np.sqrt(252) * 100
        features['vix_proxy'] = vol_20

        # Vol percentile
        features['vix_percentile'] = vol_20.rolling(252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 20 else 0.5
        )

        # Vol spike
        features['vix_spike'] = vol_20 / vol_20.rolling(60).mean()

        # Vol acceleration
        features['vix_accel'] = vol_20.diff(5)

        return features


# =============================================================================
# TARGET BUILDER
# =============================================================================
class TargetBuilder:
    """Build crisis/stress prediction targets."""

    def __init__(self, prices: pd.DataFrame):
        self.market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
        self.returns = self.market.pct_change()

    def compute_targets(self) -> pd.DataFrame:
        targets = pd.DataFrame(index=self.market.index)

        # Drawdown targets
        for horizon in [5, 10, 20]:
            future_dd = self._compute_future_drawdown(horizon)
            for threshold in [0.03, 0.05, 0.07, 0.10]:
                targets[f'drawdown_{int(threshold*100)}pct_{horizon}d'] = (future_dd < -threshold).astype(int)

        # Volatility spike targets
        realized_vol = self.returns.rolling(20).std() * np.sqrt(252)
        for horizon in [5, 10, 20]:
            future_vol = realized_vol.shift(-horizon)
            targets[f'vol_spike_2x_{horizon}d'] = (future_vol > realized_vol * 2).astype(int)
            vol_threshold = realized_vol.rolling(252).quantile(0.9)
            targets[f'vol_extreme_{horizon}d'] = (future_vol > vol_threshold).astype(int)

        # Large down moves
        for horizon in [1, 3, 5, 10]:
            future_ret = self.market.pct_change(horizon).shift(-horizon)
            for threshold in [0.02, 0.03, 0.05]:
                targets[f'down_{int(threshold*100)}pct_{horizon}d'] = (future_ret < -threshold).astype(int)

        return targets

    def _compute_future_drawdown(self, horizon: int) -> pd.Series:
        future_dd = pd.Series(index=self.market.index, dtype=float)
        for i in range(len(self.market) - horizon):
            current = self.market.iloc[i]
            future_min = self.market.iloc[i+1:i+horizon+1].min()
            future_dd.iloc[i] = (future_min - current) / current
        return future_dd


# =============================================================================
# BENCHMARK MODELS
# =============================================================================
@dataclass
class BenchmarkResult:
    """Store results for a single model."""
    name: str
    auc_roc: float
    auc_roc_std: float
    precision: float
    precision_std: float
    recall: float
    recall_std: float
    f1: float
    f1_std: float
    avg_precision: float
    brier_score: float
    fold_aucs: List[float]
    predictions: np.ndarray = None
    probabilities: np.ndarray = None
    actuals: np.ndarray = None


class BenchmarkSuite:
    """Run comprehensive benchmark comparisons."""

    def __init__(
        self,
        spectral_features: pd.DataFrame,
        traditional_features: pd.DataFrame,
        vix_features: pd.DataFrame,
        targets: pd.DataFrame
    ):
        self.spectral = spectral_features
        self.traditional = traditional_features
        self.vix = vix_features
        self.targets = targets
        self.combined = pd.concat([spectral_features, traditional_features], axis=1)

    def run_all_benchmarks(
        self,
        target_col: str,
        n_splits: int = 5,
        train_years: float = 3,
        test_months: int = 6
    ) -> Dict[str, BenchmarkResult]:
        """Run all benchmark models on a target."""

        results = {}

        # 1. Naive baselines
        results['Random'] = self._run_naive_baseline('random', target_col)
        results['Always Positive'] = self._run_naive_baseline('always_positive', target_col)
        results['Historical Rate'] = self._run_naive_baseline('historical_rate', target_col)

        # 2. VIX-only baseline
        results['VIX Proxy Only'] = self._run_model(
            self.vix, target_col,
            LogisticRegression(max_iter=1000, class_weight='balanced'),
            n_splits, train_years, test_months
        )

        # 3. Traditional features with different models
        results['Traditional + LogReg'] = self._run_model(
            self.traditional, target_col,
            LogisticRegression(max_iter=1000, class_weight='balanced'),
            n_splits, train_years, test_months
        )

        results['Traditional + GradBoost'] = self._run_model(
            self.traditional, target_col,
            GradientBoostingClassifier(n_estimators=100, max_depth=4, subsample=0.8, random_state=42),
            n_splits, train_years, test_months
        )

        # 4. Spectral features (Boris's approach)
        results['Spectral + RF (CGECD)'] = self._run_model(
            self.spectral, target_col,
            RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                   class_weight='balanced_subsample', random_state=42, n_jobs=-1),
            n_splits, train_years, test_months
        )

        results['Spectral + GradBoost'] = self._run_model(
            self.spectral, target_col,
            GradientBoostingClassifier(n_estimators=150, max_depth=5, subsample=0.8, random_state=42),
            n_splits, train_years, test_months
        )

        # 5. Combined features
        results['Combined + RF'] = self._run_model(
            self.combined, target_col,
            RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                   class_weight='balanced_subsample', random_state=42, n_jobs=-1),
            n_splits, train_years, test_months
        )

        results['Combined + GradBoost'] = self._run_model(
            self.combined, target_col,
            GradientBoostingClassifier(n_estimators=150, max_depth=5, subsample=0.8, random_state=42),
            n_splits, train_years, test_months
        )

        # 6. Neural network
        results['Combined + MLP'] = self._run_model(
            self.combined, target_col,
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42),
            n_splits, train_years, test_months
        )

        return results

    def _run_naive_baseline(self, baseline_type: str, target_col: str) -> BenchmarkResult:
        """Run naive baseline predictions."""
        y = self.targets[target_col].dropna().values
        n = len(y)

        if baseline_type == 'random':
            probs = np.random.rand(n)
            preds = (probs > 0.5).astype(int)
        elif baseline_type == 'always_positive':
            probs = np.ones(n)
            preds = np.ones(n, dtype=int)
        elif baseline_type == 'historical_rate':
            rate = y.mean()
            probs = np.full(n, rate)
            preds = (probs > 0.5).astype(int)

        try:
            auc = roc_auc_score(y, probs)
            ap = average_precision_score(y, probs)
            brier = brier_score_loss(y, probs)
        except:
            auc, ap, brier = 0.5, y.mean(), 0.25

        return BenchmarkResult(
            name=baseline_type,
            auc_roc=auc, auc_roc_std=0.0,
            precision=precision_score(y, preds, zero_division=0), precision_std=0.0,
            recall=recall_score(y, preds, zero_division=0), recall_std=0.0,
            f1=f1_score(y, preds, zero_division=0), f1_std=0.0,
            avg_precision=ap, brier_score=brier,
            fold_aucs=[auc],
            predictions=preds, probabilities=probs, actuals=y
        )

    def _run_model(
        self,
        features: pd.DataFrame,
        target_col: str,
        model,
        n_splits: int,
        train_years: float,
        test_months: int
    ) -> BenchmarkResult:
        """Run walk-forward validation for a model."""

        # Align data
        common_idx = features.dropna().index.intersection(self.targets[target_col].dropna().index)
        X = features.loc[common_idx].values
        y = self.targets.loc[common_idx, target_col].values

        # Handle inf/nan
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        train_size = int(train_years * 252)
        test_size = int(test_months * 21)
        gap = 10

        if len(X) < train_size + gap + test_size:
            return self._empty_result("Insufficient data")

        available = len(X) - train_size - gap - test_size
        step = max(test_size, available // n_splits)

        fold_aucs, fold_precisions, fold_recalls, fold_f1s = [], [], [], []
        all_preds, all_probs, all_actuals = [], [], []

        for fold in range(n_splits):
            start = fold * step
            train_end = start + train_size
            test_start = train_end + gap
            test_end = min(test_start + test_size, len(X))

            if test_end > len(X):
                break

            X_train, y_train = X[start:train_end], y[start:train_end]
            X_test, y_test = X[test_start:test_end], y[test_start:test_end]

            # Scale
            scaler = RobustScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            X_train_s = np.nan_to_num(X_train_s, nan=0, posinf=0, neginf=0)
            X_test_s = np.nan_to_num(X_test_s, nan=0, posinf=0, neginf=0)

            try:
                from sklearn.base import clone
                model_clone = clone(model)
                model_clone.fit(X_train_s, y_train)
                preds = model_clone.predict(X_test_s)
                probs = model_clone.predict_proba(X_test_s)[:, 1]

                all_preds.extend(preds)
                all_probs.extend(probs)
                all_actuals.extend(y_test)

                if len(np.unique(y_test)) > 1:
                    fold_aucs.append(roc_auc_score(y_test, probs))
                fold_precisions.append(precision_score(y_test, preds, zero_division=0))
                fold_recalls.append(recall_score(y_test, preds, zero_division=0))
                fold_f1s.append(f1_score(y_test, preds, zero_division=0))
            except Exception as e:
                continue

        if not fold_aucs:
            return self._empty_result("All folds failed")

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_actuals = np.array(all_actuals)

        try:
            overall_auc = roc_auc_score(all_actuals, all_probs)
            overall_ap = average_precision_score(all_actuals, all_probs)
            overall_brier = brier_score_loss(all_actuals, all_probs)
        except:
            overall_auc, overall_ap, overall_brier = 0.5, all_actuals.mean(), 0.25

        return BenchmarkResult(
            name=type(model).__name__,
            auc_roc=np.mean(fold_aucs), auc_roc_std=np.std(fold_aucs),
            precision=np.mean(fold_precisions), precision_std=np.std(fold_precisions),
            recall=np.mean(fold_recalls), recall_std=np.std(fold_recalls),
            f1=np.mean(fold_f1s), f1_std=np.std(fold_f1s),
            avg_precision=overall_ap, brier_score=overall_brier,
            fold_aucs=fold_aucs,
            predictions=all_preds, probabilities=all_probs, actuals=all_actuals
        )

    def _empty_result(self, name: str) -> BenchmarkResult:
        return BenchmarkResult(
            name=name, auc_roc=0.5, auc_roc_std=0.0,
            precision=0.0, precision_std=0.0,
            recall=0.0, recall_std=0.0,
            f1=0.0, f1_std=0.0,
            avg_precision=0.0, brier_score=0.25,
            fold_aucs=[0.5]
        )


# =============================================================================
# STATISTICAL SIGNIFICANCE TESTS
# =============================================================================
class StatisticalTests:
    """Perform statistical significance tests between models."""

    @staticmethod
    def paired_t_test(aucs_a: List[float], aucs_b: List[float]) -> Tuple[float, float]:
        """Paired t-test for AUC differences."""
        if len(aucs_a) != len(aucs_b) or len(aucs_a) < 2:
            return 0.0, 1.0
        t_stat, p_value = stats.ttest_rel(aucs_a, aucs_b)
        return t_stat, p_value

    @staticmethod
    def wilcoxon_test(aucs_a: List[float], aucs_b: List[float]) -> Tuple[float, float]:
        """Wilcoxon signed-rank test (non-parametric)."""
        if len(aucs_a) != len(aucs_b) or len(aucs_a) < 2:
            return 0.0, 1.0
        try:
            stat, p_value = stats.wilcoxon(aucs_a, aucs_b)
            return stat, p_value
        except:
            return 0.0, 1.0

    @staticmethod
    def delong_test(y_true: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray) -> float:
        """
        DeLong test for comparing two ROC curves.
        Simplified implementation.
        """
        from scipy.stats import norm

        n1 = np.sum(y_true == 1)
        n0 = np.sum(y_true == 0)

        if n1 == 0 or n0 == 0:
            return 1.0

        auc_a = roc_auc_score(y_true, probs_a)
        auc_b = roc_auc_score(y_true, probs_b)

        # Simplified variance estimate
        var_a = auc_a * (1 - auc_a) / min(n0, n1)
        var_b = auc_b * (1 - auc_b) / min(n0, n1)

        z = (auc_a - auc_b) / np.sqrt(var_a + var_b + 1e-10)
        p_value = 2 * (1 - norm.cdf(abs(z)))

        return p_value


# =============================================================================
# VISUALIZATION
# =============================================================================
class BenchmarkVisualizer:
    """Create publication-quality visualizations."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'cgecd': '#2ecc71',      # Green for CGECD
            'baseline': '#95a5a6',   # Gray for baselines
            'traditional': '#3498db', # Blue for traditional
            'combined': '#9b59b6',    # Purple for combined
        }

    def plot_benchmark_comparison(
        self,
        results: Dict[str, BenchmarkResult],
        target_name: str,
        filename: str
    ):
        """Bar chart comparing all benchmarks."""

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        models = list(results.keys())
        aucs = [results[m].auc_roc for m in models]
        auc_stds = [results[m].auc_roc_std for m in models]
        precisions = [results[m].precision for m in models]
        recalls = [results[m].recall for m in models]

        # Color coding
        colors = []
        for m in models:
            if 'CGECD' in m or 'Spectral' in m:
                colors.append(self.colors['cgecd'])
            elif 'Random' in m or 'Always' in m or 'Historical' in m:
                colors.append(self.colors['baseline'])
            elif 'Combined' in m:
                colors.append(self.colors['combined'])
            else:
                colors.append(self.colors['traditional'])

        # AUC-ROC
        ax = axes[0]
        bars = ax.barh(models, aucs, xerr=auc_stds, color=colors, alpha=0.8, capsize=3)
        ax.axvline(x=0.5, color='red', linestyle='--', label='Random (0.5)')
        ax.set_xlabel('AUC-ROC')
        ax.set_title('Model Comparison: AUC-ROC')
        ax.set_xlim(0.4, 1.0)

        # Add value labels
        for bar, auc in zip(bars, aucs):
            ax.text(auc + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{auc:.3f}', va='center', fontsize=9)

        # Precision
        ax = axes[1]
        ax.barh(models, precisions, color=colors, alpha=0.8)
        ax.set_xlabel('Precision')
        ax.set_title('Model Comparison: Precision')
        ax.set_xlim(0, 1.0)

        # Recall
        ax = axes[2]
        ax.barh(models, recalls, color=colors, alpha=0.8)
        ax.set_xlabel('Recall')
        ax.set_title('Model Comparison: Recall')
        ax.set_xlim(0, 1.0)

        # Legend
        legend_elements = [
            mpatches.Patch(color=self.colors['cgecd'], label='CGECD (Spectral)'),
            mpatches.Patch(color=self.colors['traditional'], label='Traditional'),
            mpatches.Patch(color=self.colors['combined'], label='Combined'),
            mpatches.Patch(color=self.colors['baseline'], label='Baseline'),
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02))

        plt.suptitle(f'Benchmark Comparison: {target_name}', fontsize=14, y=1.08)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_roc_curves(
        self,
        results: Dict[str, BenchmarkResult],
        target_name: str,
        filename: str
    ):
        """Plot ROC curves for all models."""

        fig, ax = plt.subplots(figsize=(10, 8))

        for name, res in results.items():
            if res.actuals is None or res.probabilities is None:
                continue
            if len(np.unique(res.actuals)) < 2:
                continue

            fpr, tpr, _ = roc_curve(res.actuals, res.probabilities)

            if 'CGECD' in name or 'Spectral' in name:
                ax.plot(fpr, tpr, linewidth=2.5, label=f'{name} (AUC={res.auc_roc:.3f})')
            else:
                ax.plot(fpr, tpr, linewidth=1.5, alpha=0.7, label=f'{name} (AUC={res.auc_roc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves: {target_name}')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_precision_recall_curves(
        self,
        results: Dict[str, BenchmarkResult],
        target_name: str,
        positive_rate: float,
        filename: str
    ):
        """Plot Precision-Recall curves."""

        fig, ax = plt.subplots(figsize=(10, 8))

        for name, res in results.items():
            if res.actuals is None or res.probabilities is None:
                continue
            if len(np.unique(res.actuals)) < 2:
                continue

            precision, recall, _ = precision_recall_curve(res.actuals, res.probabilities)

            if 'CGECD' in name or 'Spectral' in name:
                ax.plot(recall, precision, linewidth=2.5, label=f'{name} (AP={res.avg_precision:.3f})')
            else:
                ax.plot(recall, precision, linewidth=1.5, alpha=0.7, label=f'{name} (AP={res.avg_precision:.3f})')

        ax.axhline(y=positive_rate, color='red', linestyle='--', label=f'Random ({positive_rate:.2%})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curves: {target_name}')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_calibration_curves(
        self,
        results: Dict[str, BenchmarkResult],
        target_name: str,
        filename: str
    ):
        """Plot calibration curves."""

        fig, ax = plt.subplots(figsize=(10, 8))

        for name, res in results.items():
            if res.actuals is None or res.probabilities is None:
                continue
            if 'Random' in name or 'Always' in name:
                continue

            try:
                prob_true, prob_pred = calibration_curve(res.actuals, res.probabilities, n_bins=10)
                ax.plot(prob_pred, prob_true, 's-', label=name)
            except:
                continue

        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Curves: {target_name}')
        ax.legend(loc='lower right', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_summary_heatmap(
        self,
        all_results: Dict[str, Dict[str, BenchmarkResult]],
        filename: str
    ):
        """Create summary heatmap across all targets."""

        # Extract AUC-ROC values
        targets = list(all_results.keys())
        models = list(all_results[targets[0]].keys())

        data = []
        for target in targets:
            row = []
            for model in models:
                if model in all_results[target]:
                    row.append(all_results[target][model].auc_roc)
                else:
                    row.append(0.5)
            data.append(row)

        df = pd.DataFrame(data, index=targets, columns=models)

        fig, ax = plt.subplots(figsize=(14, 8))

        # Create heatmap
        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        sns.heatmap(df, annot=True, fmt='.3f', cmap=cmap, center=0.5,
                   vmin=0.4, vmax=0.9, ax=ax, cbar_kws={'label': 'AUC-ROC'})

        ax.set_title('AUC-ROC Comparison Across All Targets', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

        return df


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================
def run_comprehensive_benchmarks():
    """Run all benchmarks and generate reports."""

    print("=" * 80)
    print("CGECD COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading data...")
    prices, _ = load_multi_asset_data(years=15)

    # Build features
    print("\n[2/5] Building features...")

    # Spectral features
    returns = prices.pct_change().dropna()
    n_assets = len(prices.columns)
    extractor = SpectralFeatureExtractor(n_assets)

    spectral_features = []
    for i in range(60, len(returns)):
        date = returns.index[i]
        window_returns = returns.iloc[i-60:i]
        corr = window_returns.corr().values
        corr = np.nan_to_num(corr, nan=0)
        np.fill_diagonal(corr, 1.0)

        feats = extractor.extract_all_features(corr)
        feats['date'] = date
        spectral_features.append(feats)

    spectral_df = pd.DataFrame(spectral_features).set_index('date')
    print(f"  Spectral features: {len(spectral_df.columns)}")

    # Traditional features
    trad_builder = TraditionalFeatureBuilder(prices)
    trad_df = trad_builder.compute_features()
    print(f"  Traditional features: {len(trad_df.columns)}")

    # VIX features
    vix_builder = VIXBasedPredictor(prices)
    vix_df = vix_builder.compute_features()
    print(f"  VIX features: {len(vix_df.columns)}")

    # Targets
    print("\n[3/5] Building targets...")
    target_builder = TargetBuilder(prices)
    targets = target_builder.compute_targets()

    # Align all data
    common_idx = spectral_df.dropna().index
    common_idx = common_idx.intersection(trad_df.dropna().index)
    common_idx = common_idx.intersection(vix_df.dropna().index)
    common_idx = common_idx.intersection(targets.dropna(how='all').index)

    spectral_df = spectral_df.loc[common_idx]
    trad_df = trad_df.loc[common_idx]
    vix_df = vix_df.loc[common_idx]
    targets = targets.loc[common_idx]

    print(f"\nFinal dataset: {len(common_idx)} days")

    # Run benchmarks
    print("\n[4/5] Running benchmarks...")

    suite = BenchmarkSuite(spectral_df, trad_df, vix_df, targets)
    visualizer = BenchmarkVisualizer(OUTPUT_DIR)

    test_targets = [
        ('drawdown_5pct_10d', 'Drawdown >5% in 10 days'),
        ('drawdown_5pct_20d', 'Drawdown >5% in 20 days'),
        ('vol_extreme_10d', 'Extreme volatility in 10 days'),
        ('down_5pct_10d', 'Down >5% in 10 days'),
    ]

    all_results = {}
    summary_rows = []

    for target_col, target_name in test_targets:
        if target_col not in targets.columns:
            continue

        pos_rate = targets[target_col].mean()
        if pos_rate < 0.02 or pos_rate > 0.5:
            continue

        print(f"\n--- {target_name} (positive rate: {pos_rate:.1%}) ---")

        results = suite.run_all_benchmarks(target_col, n_splits=5)
        all_results[target_name] = results

        # Print results
        for name, res in sorted(results.items(), key=lambda x: -x[1].auc_roc):
            print(f"  {name:30s}: AUC={res.auc_roc:.3f}±{res.auc_roc_std:.3f}, "
                  f"Prec={res.precision:.1%}, Recall={res.recall:.1%}")

            summary_rows.append({
                'target': target_name,
                'model': name,
                'auc_roc': res.auc_roc,
                'auc_std': res.auc_roc_std,
                'precision': res.precision,
                'recall': res.recall,
                'f1': res.f1,
                'avg_precision': res.avg_precision
            })

        # Generate visualizations
        safe_name = target_col.replace('_', '-')
        visualizer.plot_benchmark_comparison(results, target_name, f'comparison_{safe_name}.png')
        visualizer.plot_roc_curves(results, target_name, f'roc_{safe_name}.png')
        visualizer.plot_precision_recall_curves(results, target_name, pos_rate, f'pr_{safe_name}.png')
        visualizer.plot_calibration_curves(results, target_name, f'calibration_{safe_name}.png')

        # Statistical significance tests
        print(f"\n  Statistical Tests (CGECD vs others):")
        cgecd_result = results.get('Spectral + RF (CGECD)')
        if cgecd_result and cgecd_result.fold_aucs:
            for name, res in results.items():
                if name == 'Spectral + RF (CGECD)' or not res.fold_aucs:
                    continue
                if len(res.fold_aucs) != len(cgecd_result.fold_aucs):
                    continue

                _, p_value = StatisticalTests.paired_t_test(cgecd_result.fold_aucs, res.fold_aucs)
                diff = cgecd_result.auc_roc - res.auc_roc
                sig = "**" if p_value < 0.05 else ""
                print(f"    vs {name:25s}: Δ={diff:+.3f}, p={p_value:.3f} {sig}")

    # Summary
    print("\n[5/5] Generating summary...")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / 'benchmark_summary.csv', index=False)

    # Summary heatmap
    if all_results:
        heatmap_df = visualizer.plot_summary_heatmap(all_results, 'summary_heatmap.png')

        # Print final summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY: Average AUC-ROC by Model Type")
        print("=" * 80)

        model_avgs = summary_df.groupby('model')['auc_roc'].mean().sort_values(ascending=False)
        for model, auc in model_avgs.items():
            marker = "★" if 'CGECD' in model or 'Spectral' in model else " "
            print(f"  {marker} {model:35s}: {auc:.3f}")

        # Calculate CGECD advantage
        cgecd_avg = model_avgs.get('Spectral + RF (CGECD)', 0.5)
        baseline_avg = model_avgs.get('Historical Rate', 0.5)
        trad_best = model_avgs[[m for m in model_avgs.index if 'Traditional' in m]].max() if any('Traditional' in m for m in model_avgs.index) else 0.5

        print(f"\n  CGECD Advantage over Historical Rate: +{(cgecd_avg - baseline_avg)*100:.1f}%")
        print(f"  CGECD Advantage over Best Traditional: +{(cgecd_avg - trad_best)*100:.1f}%")

    print(f"\nResults saved to {OUTPUT_DIR}/")

    return all_results, summary_df


if __name__ == "__main__":
    results, summary = run_comprehensive_benchmarks()
