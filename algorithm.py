#!/usr/bin/env python3
"""
Correlation Graph Eigenvalue Crisis Detector  (CGECD)  v9b
==========================================================

Surgical fix from v9: ONLY the Vol-LR sub-model changes.
- 3 compact vol features (vol_5d, vol_20d, garch_vol) instead of 16
- L2/C=0.1 matching the HAR-RV benchmark's regularization
- Gentler routing params (config)
- NO force-include, NO MI changes, NO RF changes

Rationale: HAR-RV gets 0.760 on DOWN with effectively 2 vol features
and C=0.1 LR.  Our v9 Vol-LR used 16 features with L1/C=0.5 which
overfits in internal CV → α≈0 → no contribution.  The fix matches
the benchmark's setup so Vol-LR actually activates on DOWN folds.
"""

import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from config import Config
from metrics import compute_metrics

warnings.filterwarnings("ignore")


# =====================================================================
# DATA LOADING
# =====================================================================
class DataLoader:
    BASE_URL = "https://eodhd.com/api"

    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()

    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_file = symbol.replace(".", "_").replace("/", "_")
        cache_path = self.config.cache_dir / f"{cache_file}.pkl"

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    df = pickle.load(f)
                    if len(df) > 100:
                        return df
            except Exception:
                pass

        try:
            params = {
                "api_token": self.config.api_key,
                "fmt": "json",
                "from": start_date,
                "to": end_date,
            }
            resp = self.session.get(
                f"{self.BASE_URL}/eod/{symbol}", params=params, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            with open(cache_path, "wb") as f:
                pickle.dump(df, f)
            return df
        except Exception as e:
            print(f"  Failed to load {symbol}: {e}")
            return pd.DataFrame()


def load_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    loader = DataLoader(config)
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=config.years * 365)).strftime(
        "%Y-%m-%d"
    )

    print(f"Loading {len(config.symbols)} assets...")
    data_dict: Dict[str, pd.Series] = {}
    for symbol, name in config.symbols.items():
        df = loader.get_data(symbol, start, end)
        if not df.empty and "adjusted_close" in df.columns:
            data_dict[name] = df["adjusted_close"]
            print(f"  ✓ {name}: {len(df)} days")
        else:
            print(f"  ✗ {name}: failed")

    prices = pd.DataFrame(data_dict).dropna(how="all").ffill(limit=5).dropna()
    returns = prices.pct_change().dropna()

    print(f"\nDataset: {len(prices)} days, {len(prices.columns)} assets")
    print(f"Range:   {prices.index[0].date()} → {prices.index[-1].date()}")
    return prices, returns


# =====================================================================
# CORRELATION GRAPH
# =====================================================================
class CorrelationGraphBuilder:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.n_assets = len(returns.columns)

    def compute_rolling_correlation(
        self, window: int = 60
    ) -> Dict[pd.Timestamp, np.ndarray]:
        corr_matrices: Dict[pd.Timestamp, np.ndarray] = {}
        for i in range(window, len(self.returns)):
            date = self.returns.index[i]
            w = self.returns.iloc[i - window : i]
            corr = w.corr().values
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            corr_matrices[date] = corr
        return corr_matrices


# =====================================================================
# SPECTRAL FEATURE EXTRACTION
# =====================================================================
class SpectralFeatureExtractor:
    """Curated spectral features with dual-threshold topology."""

    def __init__(self, n_assets: int, corr_window: int = 60):
        self.n_assets = n_assets
        self.corr_window = corr_window

    def extract(
        self, corr_matrix: np.ndarray, threshold: float = 0.3
    ) -> Dict[str, float]:
        C = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(C, 1.0)
        n = self.n_assets

        try:
            eigs = np.linalg.eigvalsh(C)
            eigs = np.sort(eigs)[::-1]
            eigs = np.maximum(eigs, 1e-10)
        except Exception:
            return self._defaults()

        total = np.sum(eigs)
        norm = eigs / total
        entropy = -np.sum(norm * np.log(norm + 1e-10))

        features: Dict[str, float] = {
            "lambda_1": eigs[0],
            "lambda_2": eigs[1] if n > 1 else 0.0,
            "lambda_3": eigs[2] if n > 2 else 0.0,
            "spectral_gap": eigs[0] / (eigs[1] + 1e-10) if n > 1 else float(n),
            "absorption_ratio_1": eigs[0] / total,
            "absorption_ratio_3": np.sum(eigs[: min(3, n)]) / total,
            "absorption_ratio_5": np.sum(eigs[: min(5, n)]) / total,
            "eigenvalue_entropy": entropy,
            "effective_rank": np.exp(entropy),
        }

        q = n / max(self.corr_window, 1)
        mp_upper = (1 + np.sqrt(q)) ** 2
        features["mp_excess"] = float(np.sum(eigs > mp_upper))

        for t_val, t_name in [(threshold, ""), (0.5, "_50")]:
            adj = (np.abs(C) > t_val).astype(float)
            np.fill_diagonal(adj, 0)
            max_edges = n * (n - 1) / 2
            features[f"edge_density{t_name}"] = (
                (np.sum(adj) / 2) / max_edges if max_edges else 0.0
            )

        upper = C[np.triu_indices(n, k=1)]
        features["mean_abs_corr"] = np.mean(np.abs(upper))
        features["median_corr"] = np.median(upper)
        features["frac_corr_above_50"] = np.mean(np.abs(upper) > 0.5)
        features["frac_corr_above_70"] = np.mean(np.abs(upper) > 0.7)
        features["corr_dispersion"] = np.std(upper)

        return features

    def _defaults(self) -> Dict[str, float]:
        n = self.n_assets
        return {
            "lambda_1": 1.0,
            "lambda_2": 0.0,
            "lambda_3": 0.0,
            "spectral_gap": 1.0,
            "absorption_ratio_1": 1.0 / n,
            "absorption_ratio_3": 3.0 / n,
            "absorption_ratio_5": 5.0 / n,
            "eigenvalue_entropy": np.log(n),
            "effective_rank": float(n),
            "mp_excess": 0.0,
            "edge_density": 0.0,
            "edge_density_50": 0.0,
            "mean_abs_corr": 0.0,
            "median_corr": 0.0,
            "frac_corr_above_50": 0.0,
            "frac_corr_above_70": 0.0,
            "corr_dispersion": 0.0,
        }


# =====================================================================
# FEATURE BUILDERS
# =====================================================================
def build_spectral_features(
    returns: pd.DataFrame, config: Config
) -> pd.DataFrame:
    builder = CorrelationGraphBuilder(returns)
    n_assets = len(returns.columns)
    window = config.correlation_window

    print("  Computing correlation matrices …")
    corr_mats = builder.compute_rolling_correlation(window)
    extractor = SpectralFeatureExtractor(n_assets, window)

    print("  Extracting spectral features …")
    rows: List[Dict] = []
    prev_corr: Optional[np.ndarray] = None

    for date in sorted(corr_mats):
        C = corr_mats[date]
        row: Dict[str, object] = {"date": date}
        row.update(extractor.extract(C, config.graph_threshold))

        if prev_corr is not None:
            row["corr_change_norm"] = float(
                np.linalg.norm(C - prev_corr, "fro")
            )
        else:
            row["corr_change_norm"] = 0.0
        prev_corr = C
        rows.append(row)

    base_df = pd.DataFrame(rows).set_index("date")

    print("  Computing dynamics …")
    dynamics = pd.DataFrame(index=base_df.index)

    key_feats = [
        "lambda_1",
        "absorption_ratio_1",
        "mean_abs_corr",
        "eigenvalue_entropy",
        "edge_density",
        "effective_rank",
    ]
    for feat in key_feats:
        if feat not in base_df.columns:
            continue
        s = base_df[feat]
        for lb in config.dynamics_lookbacks:
            rm = s.rolling(lb * 2).mean()
            rs = s.rolling(lb * 2).std()
            dynamics[f"{feat}_zscore_{lb}d"] = (s - rm) / (rs + 1e-10)
            dynamics[f"{feat}_roc_{lb}d"] = s.pct_change(lb)

    for feat in ["lambda_1", "absorption_ratio_1", "mean_abs_corr"]:
        if feat not in base_df.columns:
            continue
        roc = base_df[feat].pct_change(5)
        dynamics[f"{feat}_accel_5d"] = roc.diff(5)

    result = pd.concat([base_df, dynamics], axis=1)
    print(f"  Spectral features: {result.shape[1]}")
    return result


def build_traditional_features(
    prices: pd.DataFrame, returns: pd.DataFrame = None
) -> pd.DataFrame:
    market = prices["SP500"] if "SP500" in prices.columns else prices.iloc[:, 0]
    ret = market.pct_change()

    if returns is None:
        returns = prices.pct_change()

    f = pd.DataFrame(index=prices.index)

    for w in [1, 5, 10, 20, 60]:
        f[f"return_{w}d"] = market.pct_change(w)

    f["volatility_1d"] = ret.abs() * np.sqrt(252)
    for w in [5, 10, 20, 60]:
        f[f"volatility_{w}d"] = ret.rolling(w).std() * np.sqrt(252)
    f["vol_ratio_5_20"] = f["volatility_5d"] / (f["volatility_20d"] + 1e-8)
    f["vol_ratio_5_60"] = f["volatility_5d"] / (f["volatility_60d"] + 1e-8)
    f["vol_ratio_1_20"] = f["volatility_1d"] / (f["volatility_20d"] + 1e-8)
    f["vol_change_5d"] = f["volatility_5d"].pct_change(5)
    f["vol_of_vol"] = f["volatility_5d"].rolling(20).std()

    ret_clean = ret.dropna()
    if len(ret_clean) > 10:
        alpha_g, beta_g = 0.1, 0.85
        omega_g = (1.0 - alpha_g - beta_g) * float(ret_clean.var())
        var_t = float(ret_clean.var())
        garch_vals = np.empty(len(ret_clean))
        garch_vals[0] = np.sqrt(max(var_t, 1e-12)) * np.sqrt(252)
        for i in range(1, len(ret_clean)):
            var_t = omega_g + alpha_g * ret_clean.iloc[i - 1] ** 2 + beta_g * var_t
            garch_vals[i] = np.sqrt(max(var_t, 1e-12)) * np.sqrt(252)
        gseries = pd.Series(garch_vals, index=ret_clean.index)
        f["garch_vol"] = gseries
        f["garch_vol_zscore"] = (
            (gseries - gseries.rolling(60).mean())
            / (gseries.rolling(60).std() + 1e-10)
        )

    f["price_to_sma_20"] = market / market.rolling(20).mean() - 1
    f["price_to_sma_50"] = market / market.rolling(50).mean() - 1

    delta = market.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    f["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    f["momentum_reversal"] = f["return_5d"] - f["return_20d"]

    for w in [20, 60]:
        rm = market.rolling(w).max()
        f[f"drawdown_{w}d"] = (market - rm) / rm
    f["drawdown_speed_5d"] = f["drawdown_20d"].diff(5)

    f["skewness_20d"] = ret.rolling(20).skew()
    f["kurtosis_20d"] = ret.rolling(20).kurt()
    f["downside_vol_20d"] = (
        np.sqrt((ret.clip(upper=0) ** 2).rolling(20).mean()) * np.sqrt(252)
    )
    f["down_up_vol_ratio"] = f["downside_vol_20d"] / (f["volatility_20d"] + 1e-8)
    f["max_loss_5d"] = ret.rolling(5).min()
    f["max_loss_20d"] = ret.rolling(20).min()
    f["neg_days_10d"] = (ret < 0).astype(float).rolling(10).sum()

    if "HighYield" in prices.columns and "InvGradeCorp" in prices.columns:
        hy_ret = prices["HighYield"].pct_change()
        ig_ret = prices["InvGradeCorp"].pct_change()
        spread = hy_ret - ig_ret
        f["credit_spread_5d"] = spread.rolling(5).mean() * 252
        f["credit_spread_zscore"] = (
            (spread.rolling(5).mean() - spread.rolling(60).mean())
            / (spread.rolling(60).std() + 1e-10)
        )

    if "LongTreasury" in prices.columns:
        tlt_ret = prices["LongTreasury"].pct_change()
        fts = tlt_ret.rolling(5).mean() - ret.rolling(5).mean()
        f["flight_to_safety_5d"] = fts * 252
        f["flight_to_safety_zscore"] = (
            (fts - fts.rolling(60).mean()) / (fts.rolling(60).std() + 1e-10)
        )

    sma20 = prices.rolling(20).mean()
    below_sma = (prices < sma20).astype(float)
    f["breadth_below_sma20"] = below_sma.mean(axis=1)
    f["breadth_change_5d"] = f["breadth_below_sma20"].diff(5)

    all_rets = prices.pct_change()
    f["cross_dispersion"] = all_rets.std(axis=1)
    cd_5 = f["cross_dispersion"].rolling(5).mean()
    cd_60 = f["cross_dispersion"].rolling(60).mean()
    cd_std = f["cross_dispersion"].rolling(60).std()
    f["cross_dispersion_zscore"] = (cd_5 - cd_60) / (cd_std + 1e-10)

    if "EmergingMkts" in prices.columns:
        em_ret = prices["EmergingMkts"].pct_change()
        f["em_stress_5d"] = (
            em_ret.rolling(5).mean() - ret.rolling(5).mean()
        ) * 252

    print(f"  Traditional features: {len(f.columns)}")
    return f


# =====================================================================
# FEATURE SELECTION HELPERS
# =====================================================================
def _mi_select_with_filters(
    Xs: np.ndarray,
    y: np.ndarray,
    k: int,
    corr_threshold: float = 0.98,
    random_seed: int = 42,
) -> np.ndarray:
    mi = mutual_info_classif(
        Xs, y, random_state=random_seed, n_neighbors=5
    )
    mi = np.nan_to_num(mi, nan=0.0)

    variances = np.var(Xs, axis=0)
    mi[variances < 1e-6] = 0.0

    sel = np.argsort(mi)[-k:]

    if len(sel) < 3:
        return sel

    Xk = Xs[:, sel]
    corr = np.corrcoef(Xk.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)

    to_drop: set = set()
    for i in range(len(sel)):
        if i in to_drop:
            continue
        for j in range(i + 1, len(sel)):
            if j in to_drop:
                continue
            if abs(corr[i, j]) > corr_threshold:
                if mi[sel[i]] < mi[sel[j]]:
                    to_drop.add(i)
                else:
                    to_drop.add(j)

    if to_drop:
        sel = np.array([sel[i] for i in range(len(sel)) if i not in to_drop])

    return sel


# =====================================================================
# VOLATILITY FEATURES
# =====================================================================

# Full set — kept for backward compat / investigate.py
_VOL_FEATURES = {
    "volatility_1d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volatility_60d",
    "vol_ratio_1_20",
    "vol_ratio_5_20",
    "vol_ratio_5_60",
    "vol_change_5d",
    "vol_of_vol",
    "garch_vol",
    "garch_vol_zscore",
    "max_loss_5d",
    "max_loss_20d",
    "downside_vol_20d",
    "down_up_vol_ratio",
}

# ── NEW: Compact vol set for Vol-LR sub-model ──────────────────
# Matches HAR-RV benchmark's effective features (Corsi 2009):
#   volatility_5d  ≈ rv_w  (weekly realized vol)
#   volatility_20d ≈ rv_m  (monthly realized vol)
# Plus GARCH conditional vol for additional signal.
# With 3 features + L2/C=0.1, this LR should perform at near-
# HAR-RV levels on DOWN, enabling meaningful ensemble blending.
_COMPACT_VOL = {"volatility_5d", "volatility_20d", "garch_vol"}


# =====================================================================
# MODELS
# =====================================================================
class CGECDModel:
    """CGECD v9b — Feature-Augmented RF + Compact Vol-LR Ensemble.

    Changes from v9:
    - Vol-LR uses 3 compact features instead of 16 (_COMPACT_VOL)
    - Vol-LR uses L2/C=0.1 matching HAR-RV benchmark's regularization
    - Config params allow slightly more routing on DOWN folds

    Everything else (augmentation, MI selection, RF) is IDENTICAL to v9.
    """

    name = "CGECD (Ours)"

    def __init__(self, config: Config):
        self.config = config
        self.scaler: Optional[RobustScaler] = None
        self.selected_idx: Optional[np.ndarray] = None
        self.rf: Optional[RandomForestClassifier] = None
        self._feature_names: Optional[List[str]] = None
        self._n_augmented: int = 0
        # Vol-LR ensemble
        self._vol_lr: Optional[LogisticRegression] = None
        self._alpha: float = 0.0
        self._vol_idx: Optional[List[int]] = None

    # -----------------------------------------------------------------
    def _augment_raw(self, X_raw: np.ndarray) -> Tuple[np.ndarray, int]:
        """Add economically-motivated augmented features (on raw data)."""
        names = list(self._feature_names) if self._feature_names else []
        if not names:
            return X_raw, 0

        lookup = {n: i for i, n in enumerate(names)}
        new_cols: List[np.ndarray] = []

        # ── 1. Novel vol ratios ────────────────────────────────
        for v_num, v_den in [
            ("volatility_1d", "volatility_5d"),
            ("volatility_20d", "volatility_60d"),
        ]:
            if v_num in lookup and v_den in lookup:
                a = X_raw[:, lookup[v_num]]
                b = X_raw[:, lookup[v_den]]
                new_cols.append(a / (b + 1e-8))

        # ── 2. Static spectral × volatility interactions ───────
        for feat_a, feat_b in [
            ("lambda_1", "volatility_5d"),
            ("absorption_ratio_1", "volatility_20d"),
            ("lambda_1", "vol_ratio_5_20"),
            ("effective_rank", "drawdown_20d"),
            ("eigenvalue_entropy", "vol_of_vol"),
        ]:
            if feat_a in lookup and feat_b in lookup:
                new_cols.append(
                    X_raw[:, lookup[feat_a]] * X_raw[:, lookup[feat_b]]
                )

        # ── 3. Dynamic spectral × volatility interactions ──────
        for feat_a, feat_b in [
            ("lambda_1_zscore_10d", "vol_ratio_5_20"),
            ("absorption_ratio_1_roc_10d", "vol_change_5d"),
            ("mean_abs_corr_zscore_10d", "downside_vol_20d"),
            ("corr_change_norm", "volatility_5d"),
        ]:
            if feat_a in lookup and feat_b in lookup:
                new_cols.append(
                    X_raw[:, lookup[feat_a]] * X_raw[:, lookup[feat_b]]
                )

        # ── 4. Tail-risk × spectral interactions ───────────────
        for feat_a, feat_b in [
            ("kurtosis_20d", "lambda_1"),
            ("skewness_20d", "absorption_ratio_1"),
            ("max_loss_5d", "spectral_gap"),
        ]:
            if feat_a in lookup and feat_b in lookup:
                new_cols.append(
                    X_raw[:, lookup[feat_a]] * X_raw[:, lookup[feat_b]]
                )

        n_added = len(new_cols)
        if new_cols:
            return np.column_stack([X_raw] + new_cols), n_added
        return X_raw, 0

    # -----------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_clean = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Augment BEFORE scaling
        X_aug, self._n_augmented = self._augment_raw(X_clean)

        # Scale
        self.scaler = RobustScaler()
        Xs = self.scaler.fit_transform(X_aug)

        # Adaptive MI selection
        n_pos = int(np.sum(y))
        k = min(
            self.config.feature_selection_k,
            max(8, n_pos // 3),
        )
        self.selected_idx = _mi_select_with_filters(
            Xs,
            y,
            k,
            corr_threshold=self.config.corr_filter_threshold,
            random_seed=self.config.random_seed,
        )
        Xk = Xs[:, self.selected_idx]

        # Train RF with OOB scoring
        self.rf = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            min_samples_split=self.config.rf_min_samples_split,
            class_weight="balanced_subsample",
            random_state=self.config.random_seed,
            oob_score=True,
            n_jobs=-1,
        )
        self.rf.fit(Xk, y)

        # ── Compact Vol-LR ensemble (CHANGED from v9) ──────────
        self._alpha = 0.0
        self._vol_lr = None

        names = list(self._feature_names) if self._feature_names else []
        vol_idx = [i for i, nm in enumerate(names) if nm in _COMPACT_VOL]
        self._vol_idx = vol_idx

        if len(vol_idx) >= 2:
            Xv = Xs[:, vol_idx]
            if not np.all(np.std(Xv, axis=0) < 1e-10):
                vol_cv_auc = 0.5
                try:
                    tscv = TimeSeriesSplit(n_splits=3)
                    cv_preds = np.full(len(y), np.nan)
                    for tr_i, te_i in tscv.split(Xv):
                        if len(np.unique(y[tr_i])) < 2:
                            continue
                        lr_t = LogisticRegression(
                            penalty="l2",
                            solver="lbfgs",
                            C=0.1,
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=self.config.random_seed,
                        )
                        lr_t.fit(Xv[tr_i], y[tr_i])
                        cv_preds[te_i] = lr_t.predict_proba(Xv[te_i])[:, 1]

                    valid = ~np.isnan(cv_preds)
                    if (
                        np.sum(valid) > 20
                        and len(np.unique(y[valid])) >= 2
                    ):
                        vol_cv_auc = roc_auc_score(y[valid], cv_preds[valid])
                except Exception:
                    vol_cv_auc = 0.5

                rf_oob_auc = 0.5
                try:
                    rf_oob_auc = roc_auc_score(
                        y, self.rf.oob_decision_function_[:, 1]
                    )
                except Exception:
                    pass

                diff = (
                    vol_cv_auc
                    - rf_oob_auc
                    - self.config.vol_correction_threshold
                )
                self._alpha = min(
                    max(0.0, diff * self.config.vol_correction_scale),
                    self.config.vol_correction_max,
                )

                if self._alpha > 0:
                    self._vol_lr = LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        C=0.1,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=self.config.random_seed,
                    )
                    self._vol_lr.fit(Xv, y)

    # -----------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_clean = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_aug, _ = self._augment_raw(X_clean)
        Xs = self.scaler.transform(X_aug)
        Xk = Xs[:, self.selected_idx]
        rf_probs = self.rf.predict_proba(Xk)[:, 1]

        if self._alpha > 0 and self._vol_lr is not None and self._vol_idx:
            Xv = Xs[:, self._vol_idx]
            vol_probs = self._vol_lr.predict_proba(Xv)[:, 1]
            return (1.0 - self._alpha) * rf_probs + self._alpha * vol_probs
        return rf_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


class CGECDRFModel:
    """RF with MI selection — ablation / feature-importance variant."""

    name = "CGECD (RF)"

    def __init__(self, config: Config):
        self.config = config
        self.scaler: Optional[RobustScaler] = None
        self.selected_idx: Optional[np.ndarray] = None
        self.model: Optional[RandomForestClassifier] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_clean = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        self.scaler = RobustScaler()
        Xs = self.scaler.fit_transform(X_clean)

        k = min(self.config.feature_selection_k, Xs.shape[1])
        self.selected_idx = _mi_select_with_filters(
            Xs,
            y,
            k,
            corr_threshold=self.config.corr_filter_threshold,
            random_seed=self.config.random_seed,
        )
        Xk = Xs[:, self.selected_idx]

        self.model = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            min_samples_split=self.config.rf_min_samples_split,
            class_weight="balanced_subsample",
            random_state=self.config.random_seed,
            n_jobs=-1,
        )
        self.model.fit(Xk, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_clean = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        Xs = self.scaler.transform(X_clean)
        return self.model.predict_proba(Xs[:, self.selected_idx])[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


class CGECDNoSelModel:
    """Plain RF without selection — ablation baseline."""

    name = "No Selection RF"

    def __init__(self, config: Config):
        self.config = config
        self.scaler: Optional[RobustScaler] = None
        self.selected_idx = None
        self.model: Optional[RandomForestClassifier] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_clean = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        self.scaler = RobustScaler()
        Xs = self.scaler.fit_transform(X_clean)

        self.model = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            min_samples_split=self.config.rf_min_samples_split,
            class_weight="balanced_subsample",
            random_state=self.config.random_seed,
            n_jobs=-1,
        )
        self.model.fit(Xs, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_clean = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        Xs = self.scaler.transform(X_clean)
        return self.model.predict_proba(Xs)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


# =====================================================================
# WALK-FORWARD EVALUATION  (expanding window)
# =====================================================================
def _compute_fold_schedule(
    n_samples: int,
    train_size: int,
    test_size: int,
    gap: int,
    n_splits: int,
) -> List[Tuple[int, int, int]]:
    available = n_samples - train_size - gap
    n_folds = min(n_splits, max(1, available // test_size))
    schedule: List[Tuple[int, int, int]] = []
    for fold in range(n_folds):
        train_end = train_size + fold * test_size
        test_start = train_end + gap
        test_end = min(test_start + test_size, n_samples)
        if test_start >= n_samples or test_end <= test_start:
            break
        schedule.append((train_end, test_start, test_end))
    return schedule


def walk_forward_evaluate(
    features: pd.DataFrame,
    target: pd.Series,
    model_class,
    config: Config,
) -> Dict:
    valid_dates = target.dropna().index
    X = features.reindex(valid_dates).ffill().fillna(0)
    y = target.loc[valid_dates]

    feature_names = list(X.columns)
    n_features = len(feature_names)

    train_size = int(config.train_years * 252)
    test_size = int(config.test_months * 21)
    gap = config.gap_days

    if len(X) < train_size + gap + test_size:
        return {"error": "Insufficient data"}

    schedule = _compute_fold_schedule(
        len(X), train_size, test_size, gap, config.n_splits
    )
    if not schedule:
        return {"error": "No valid folds"}

    all_probs: List[float] = []
    all_actuals: List[int] = []
    all_dates: List = []
    all_importances: List[np.ndarray] = []

    for fold_idx, (train_end, test_start, test_end) in enumerate(schedule):
        X_train = np.nan_to_num(
            X.iloc[:train_end].values, nan=0, posinf=0, neginf=0
        )
        y_train = y.iloc[:train_end].values
        X_test = np.nan_to_num(
            X.iloc[test_start:test_end].values, nan=0, posinf=0, neginf=0
        )
        y_test = y.iloc[test_start:test_end].values

        if len(np.unique(y_train)) < 2:
            continue

        try:
            model = model_class(config)

            if hasattr(model, "_feature_names"):
                model._feature_names = feature_names

            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)

            all_probs.extend(probs.tolist())
            all_actuals.extend(y_test.tolist())
            all_dates.extend(y.iloc[test_start:test_end].index.tolist())

            # ── Collect feature importances ──
            imp = np.zeros(n_features)
            collected = False

            rf_obj = getattr(model, "rf", None) or getattr(
                model, "model", None
            )
            if rf_obj is not None and hasattr(rf_obj, "feature_importances_"):
                sel = getattr(model, "selected_idx", None)
                if (
                    sel is not None
                    and len(rf_obj.feature_importances_) == len(sel)
                ):
                    for i, s_idx in enumerate(sel):
                        if s_idx < n_features:
                            imp[s_idx] = rf_obj.feature_importances_[i]
                    total_imp = np.sum(imp)
                    if total_imp > 0:
                        imp = imp / total_imp
                    collected = True
                elif len(rf_obj.feature_importances_) == n_features:
                    imp = rf_obj.feature_importances_.copy()
                    collected = True

            if collected:
                all_importances.append(imp)

        except Exception as e:
            print(f"    Fold {fold_idx} failed: {e}")
            continue

    if not all_probs:
        return {"error": "All folds failed"}

    all_probs_arr = np.array(all_probs)
    all_actuals_arr = np.array(all_actuals)

    result: Dict = {
        "metrics": compute_metrics(
            all_actuals_arr,
            (all_probs_arr >= 0.5).astype(int),
            all_probs_arr,
        ),
        "probabilities": all_probs_arr,
        "actuals": all_actuals_arr,
        "predictions": (all_probs_arr >= 0.5).astype(int),
        "dates": all_dates,
        "feature_names": feature_names,
    }
    if all_importances:
        result["feature_importances"] = np.mean(all_importances, axis=0)
    return result