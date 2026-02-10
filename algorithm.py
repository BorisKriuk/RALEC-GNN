#!/usr/bin/env python3
"""
Correlation Graph Eigenvalue Crisis Detector  (CGECD)  v4
=========================================================

Key changes from v3
-------------------
* Reverted to single 60d spectral window — the 30d window added noise
  (spectral-only AUC dropped 0.641→0.637) while bloating the feature
  space from 80→110 features.
* MI selection RESTORED (k=40) — removing it was catastrophic for DOWN
  (0.677→0.636) because tree models with 110 features overfit on ~33
  positive crash events per fold.
* Replaced fixed-weight 3-model ensemble with **adaptive RF + Vol-LR
  correction**:
  - RF on MI-selected features → strong UP detection (matches 0.836
    ablation result).
  - L1 Logistic Regression on 16 volatility features → robust DOWN
    detection (matches HAR-RV's approach: few features, linear model,
    can't overfit).
  - AUC-based correction factor (α) auto-routes predictions:
      • When vol-LR outperforms RF (DOWN regime): α > 0, vol-LR
        dominates.
      • When RF outperforms vol-LR (UP regime): α = 0, pure RF.
* Enhanced traditional features retained (1d vol, GARCH, momentum
  reversal) — these are the features the vol-LR component needs to
  match HAR-RV performance on DOWN.
"""

import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

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

        # Marchenko-Pastur excess eigenvalues
        q = n / max(self.corr_window, 1)
        mp_upper = (1 + np.sqrt(q)) ** 2
        features["mp_excess"] = float(np.sum(eigs > mp_upper))

        # Topology at two thresholds
        for t_val, t_name in [(threshold, ""), (0.5, "_50")]:
            adj = (np.abs(C) > t_val).astype(float)
            np.fill_diagonal(adj, 0)
            max_edges = n * (n - 1) / 2
            features[f"edge_density{t_name}"] = (
                (np.sum(adj) / 2) / max_edges if max_edges else 0.0
            )

        # Correlation statistics
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
    """Spectral features from single 60d rolling correlation graph."""

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

    # ── Dynamics ────────────────────────────────────────────────
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

    # Acceleration
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
    """Enhanced traditional features with 1d realised vol, GARCH
    conditional vol, and momentum reversal."""

    market = prices["SP500"] if "SP500" in prices.columns else prices.iloc[:, 0]
    ret = market.pct_change()

    if returns is None:
        returns = prices.pct_change()

    f = pd.DataFrame(index=prices.index)

    # ── Returns ─────────────────────────────────────────────────
    for w in [1, 5, 10, 20, 60]:
        f[f"return_{w}d"] = market.pct_change(w)

    # ── Volatility (incl. 1-day HAR-RV-style) ──────────────────
    f["volatility_1d"] = ret.abs() * np.sqrt(252)
    for w in [5, 10, 20, 60]:
        f[f"volatility_{w}d"] = ret.rolling(w).std() * np.sqrt(252)
    f["vol_ratio_5_20"] = f["volatility_5d"] / (f["volatility_20d"] + 1e-8)
    f["vol_ratio_5_60"] = f["volatility_5d"] / (f["volatility_60d"] + 1e-8)
    f["vol_ratio_1_20"] = f["volatility_1d"] / (f["volatility_20d"] + 1e-8)
    f["vol_change_5d"] = f["volatility_5d"].pct_change(5)
    f["vol_of_vol"] = f["volatility_5d"].rolling(20).std()

    # ── GARCH(1,1) conditional volatility ───────────────────────
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

    # ── Momentum ────────────────────────────────────────────────
    f["price_to_sma_20"] = market / market.rolling(20).mean() - 1
    f["price_to_sma_50"] = market / market.rolling(50).mean() - 1

    delta = market.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    f["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    f["momentum_reversal"] = f["return_5d"] - f["return_20d"]

    # ── Drawdown ────────────────────────────────────────────────
    for w in [20, 60]:
        rm = market.rolling(w).max()
        f[f"drawdown_{w}d"] = (market - rm) / rm
    f["drawdown_speed_5d"] = f["drawdown_20d"].diff(5)

    # ── Higher moments / tail risk ──────────────────────────────
    f["skewness_20d"] = ret.rolling(20).skew()
    f["kurtosis_20d"] = ret.rolling(20).kurt()
    f["downside_vol_20d"] = (
        np.sqrt((ret.clip(upper=0) ** 2).rolling(20).mean()) * np.sqrt(252)
    )
    f["down_up_vol_ratio"] = f["downside_vol_20d"] / (f["volatility_20d"] + 1e-8)
    f["max_loss_5d"] = ret.rolling(5).min()
    f["max_loss_20d"] = ret.rolling(20).min()
    f["neg_days_10d"] = (ret < 0).astype(float).rolling(10).sum()

    # ── Cross-asset features ────────────────────────────────────
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
# MODELS
# =====================================================================

# Volatility-related feature names used by the Vol-LR component.
# These are identified by exact name matching against the combined
# feature DataFrame columns.
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


class CGECDModel:
    """CGECD v4 — RF + adaptive Vol-LR correction.

    RF on MI-selected features provides strong non-linear detection
    (especially for UP moves).  L1-penalised logistic regression on 16
    volatility features provides robust linear detection (especially for
    DOWN moves — same principle that makes HAR-RV dominant for crash
    prediction).

    An AUC-based correction factor (α) automatically routes predictions:
      • α = 0   when RF has higher OOB-AUC  →  pure RF  (UP regime)
      • α > 0   when Vol-LR has higher CV-AUC →  blended (DOWN regime)

    This avoids the v3 failure mode where fixed-weight ensembles diluted
    the best model for each task.
    """

    name = "CGECD (Ours)"

    def __init__(self, config: Config):
        self.config = config
        self.scaler: Optional[RobustScaler] = None
        self.selected_idx: Optional[np.ndarray] = None
        self.vol_full_idx: Optional[List[int]] = None
        self.rf: Optional[RandomForestClassifier] = None
        self.vol_lr: Optional[LogisticRegression] = None
        self.alpha: float = 0.0
        self._feature_names: Optional[List[str]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_clean = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        self.scaler = RobustScaler()
        Xs = self.scaler.fit_transform(X_clean)

        # ── MI feature selection ────────────────────────────────
        k = min(self.config.feature_selection_k, Xs.shape[1])
        mi = mutual_info_classif(
            Xs, y, random_state=self.config.random_seed, n_neighbors=5
        )
        mi = np.nan_to_num(mi, nan=0.0)
        self.selected_idx = np.argsort(mi)[-k:]
        Xk = Xs[:, self.selected_idx]

        # ── Identify volatility features in FULL feature set ────
        self.vol_full_idx = []
        if self._feature_names is not None:
            self.vol_full_idx = [
                i
                for i, name in enumerate(self._feature_names)
                if name in _VOL_FEATURES
            ]

        has_vol = len(self.vol_full_idx) >= 3
        if has_vol:
            Xv_check = Xs[:, self.vol_full_idx]
            if np.all(np.std(Xv_check, axis=0) < 1e-10):
                has_vol = False

        # ── Train RF with OOB ──────────────────────────────────
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

        try:
            oob_probs = self.rf.oob_decision_function_[:, 1]
            rf_auc = roc_auc_score(y, oob_probs)
        except Exception:
            rf_auc = 0.5

        # ── Train Vol-LR with time-series CV AUC ───────────────
        self.alpha = 0.0
        self.vol_lr = None

        if has_vol:
            Xv = Xs[:, self.vol_full_idx]
            vol_auc = 0.5

            try:
                tscv = TimeSeriesSplit(n_splits=3)
                cv_probs = np.full(len(y), np.nan)

                for tr_idx, te_idx in tscv.split(Xv):
                    if len(np.unique(y[tr_idx])) < 2:
                        continue
                    lr_tmp = LogisticRegression(
                        penalty="l1",
                        solver="saga",
                        C=0.5,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=self.config.random_seed,
                    )
                    lr_tmp.fit(Xv[tr_idx], y[tr_idx])
                    cv_probs[te_idx] = lr_tmp.predict_proba(Xv[te_idx])[:, 1]

                valid = ~np.isnan(cv_probs)
                if np.sum(valid) > 20 and len(np.unique(y[valid])) >= 2:
                    vol_auc = roc_auc_score(y[valid], cv_probs[valid])
            except Exception:
                vol_auc = 0.5

            # Correction factor: positive only when vol-LR is
            # meaningfully better than RF
            diff = (
                vol_auc
                - rf_auc
                - self.config.vol_correction_threshold
            )
            self.alpha = max(0.0, diff * self.config.vol_correction_scale)
            self.alpha = min(self.alpha, self.config.vol_correction_max)

            # Train final vol-LR on ALL training data
            if self.alpha > 0:
                self.vol_lr = LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=0.5,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=self.config.random_seed,
                )
                self.vol_lr.fit(Xv, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_clean = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        Xs = self.scaler.transform(X_clean)
        Xk = Xs[:, self.selected_idx]

        pred_rf = self.rf.predict_proba(Xk)[:, 1]

        if self.vol_lr is not None and self.alpha > 0:
            Xv = Xs[:, self.vol_full_idx]
            pred_vol = self.vol_lr.predict_proba(Xv)[:, 1]
            return (1.0 - self.alpha) * pred_rf + self.alpha * pred_vol

        return pred_rf

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
        mi = mutual_info_classif(
            Xs, y, random_state=self.config.random_seed, n_neighbors=5
        )
        mi = np.nan_to_num(mi, nan=0.0)
        self.selected_idx = np.argsort(mi)[-k:]
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
    """Expanding-window walk-forward CV with aligned folds."""

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

            # Pass feature names so CGECDModel can identify vol features
            if hasattr(model, "_feature_names"):
                model._feature_names = feature_names

            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)

            all_probs.extend(probs.tolist())
            all_actuals.extend(y_test.tolist())
            all_dates.extend(y.iloc[test_start:test_end].index.tolist())

            # ── Feature importance extraction ───────────────────
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
                    imp[sel] = rf_obj.feature_importances_
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