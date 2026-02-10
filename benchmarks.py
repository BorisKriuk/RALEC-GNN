#!/usr/bin/env python3
"""
SOTA benchmark models for crisis prediction.

Signal-based (few features → Logistic Regression):
  • Absorption Ratio   (Kritzman et al., 2011)
  • Turbulence Index    (Kritzman & Li, 2010)
  • GARCH(1,1) Volatility
  • HAR-RV              (Corsi, 2009)

ML baselines (traditional features):
  • Logistic Regression
  • SVM (RBF)
  • Gradient Boosting
  • Random Forest
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

from config import Config


# =====================================================================
# BENCHMARK FEATURE COMPUTATION
# =====================================================================
def compute_absorption_ratio(
    returns: pd.DataFrame, window: int = 60, n_top: int = 5
) -> pd.Series:
    ar = pd.Series(index=returns.index, dtype=float)
    n_assets = len(returns.columns)
    n_top = min(n_top, max(1, n_assets // 5))
    for i in range(window, len(returns)):
        w = returns.iloc[i - window : i]
        corr = w.corr().values
        corr = np.nan_to_num(corr, nan=0)
        np.fill_diagonal(corr, 1.0)
        try:
            eigs = np.sort(np.maximum(np.linalg.eigvalsh(corr), 0))[::-1]
            t = np.sum(eigs)
            if t > 0:
                ar.iloc[i] = np.sum(eigs[:n_top]) / t
        except Exception:
            pass
    return ar


def compute_turbulence(returns: pd.DataFrame, lookback: int = 252) -> pd.Series:
    turb = pd.Series(index=returns.index, dtype=float)
    for i in range(lookback, len(returns)):
        hist = returns.iloc[i - lookback : i]
        cur = returns.iloc[i].values
        mu = hist.mean().values
        cov = hist.cov().values + np.eye(len(hist.columns)) * 1e-6
        try:
            diff = cur - mu
            turb.iloc[i] = diff @ np.linalg.inv(cov) @ diff
        except Exception:
            pass
    return turb


def compute_garch_vol(
    returns: pd.Series, alpha: float = 0.1, beta: float = 0.85
) -> pd.Series:
    """Simple GARCH(1,1) volatility.  Drops leading NaN to avoid propagation."""
    returns = returns.dropna()
    if len(returns) < 10:
        return pd.Series(dtype=float)

    omega = (1 - alpha - beta) * returns.var()
    gv = pd.Series(index=returns.index, dtype=float)
    var_t = returns.var()

    for i in range(1, len(returns)):
        r_prev = returns.iloc[i - 1]
        if np.isfinite(r_prev):
            var_t = omega + alpha * r_prev ** 2 + beta * var_t
        gv.iloc[i] = np.sqrt(max(var_t, 1e-12)) * np.sqrt(252)

    return gv


def compute_har_features(returns: pd.Series) -> pd.DataFrame:
    rv_d = returns.rolling(1).std() * np.sqrt(252)
    rv_m = returns.rolling(22).std() * np.sqrt(252)
    return pd.DataFrame(
        {
            "rv_d": rv_d,
            "rv_w": returns.rolling(5).std() * np.sqrt(252),
            "rv_m": rv_m,
            "rv_ratio": rv_d / (rv_m + 1e-10),
        },
        index=returns.index,
    )


def prepare_benchmark_features(
    prices: pd.DataFrame, returns: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    market = prices["SP500"] if "SP500" in prices.columns else prices.iloc[:, 0]
    mret = market.pct_change()

    print("  Computing benchmark features …")
    features: Dict[str, pd.DataFrame] = {}

    ar = compute_absorption_ratio(returns)
    features["absorption_ratio"] = pd.DataFrame(
        {
            "ar": ar,
            "ar_ma5": ar.rolling(5).mean(),
            "ar_ma20": ar.rolling(20).mean(),
            "ar_zscore": (ar - ar.rolling(60).mean()) / (ar.rolling(60).std() + 1e-10),
            "ar_roc_5d": ar.pct_change(5),
        }
    )

    turb = compute_turbulence(returns)
    features["turbulence"] = pd.DataFrame(
        {
            "turb": turb,
            "turb_ma5": turb.rolling(5).mean(),
            "turb_ma20": turb.rolling(20).mean(),
            "turb_zscore": (turb - turb.rolling(60).mean())
            / (turb.rolling(60).std() + 1e-10),
            "turb_roc_5d": turb.pct_change(5),
        }
    )

    garch = compute_garch_vol(mret)
    features["garch"] = pd.DataFrame(
        {
            "garch": garch,
            "garch_ma5": garch.rolling(5).mean(),
            "garch_zscore": (garch - garch.rolling(60).mean())
            / (garch.rolling(60).std() + 1e-10),
            "garch_roc_5d": garch.pct_change(5),
        }
    )

    features["har_rv"] = compute_har_features(mret)

    for name, df in features.items():
        print(f"    {name}: {len(df.columns)} features")
    return features


# =====================================================================
# BASELINE MODELS
# =====================================================================
class _ScaledLR:
    """Logistic Regression with RobustScaler — reused by signal models."""

    name = "Logistic Regression"

    def __init__(self, config: Config):
        self.config = config
        self.scaler = None
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xc = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        self.scaler = RobustScaler()
        Xs = self.scaler.fit_transform(Xc)
        self.model = LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=1000, random_state=42
        )
        self.model.fit(Xs, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xc = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        return self.model.predict_proba(self.scaler.transform(Xc))[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


class LogisticRegressionModel(_ScaledLR):
    name = "Logistic Regression"


class AbsorptionRatioModel(_ScaledLR):
    name = "Absorption Ratio"


class TurbulenceModel(_ScaledLR):
    name = "Turbulence Index"


class GARCHModel(_ScaledLR):
    name = "GARCH(1,1)"


class HARRVModel(_ScaledLR):
    name = "HAR-RV"


class SVMBaselineModel:
    name = "SVM"

    def __init__(self, config: Config):
        self.config = config
        self.scaler = None
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xc = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        self.scaler = RobustScaler()
        Xs = self.scaler.fit_transform(Xc)
        self.model = SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
        self.model.fit(Xs, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xc = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        return self.model.predict_proba(self.scaler.transform(Xc))[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


class GradientBoostingModel:
    name = "Gradient Boosting"

    def __init__(self, config: Config):
        self.config = config
        self.scaler = None
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xc = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        self.scaler = RobustScaler()
        Xs = self.scaler.fit_transform(Xc)
        sw = compute_sample_weight("balanced", y)
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.model.fit(Xs, y, sample_weight=sw)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xc = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        return self.model.predict_proba(self.scaler.transform(Xc))[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


class RandomForestBaselineModel:
    name = "Random Forest"

    def __init__(self, config: Config):
        self.config = config
        self.scaler = None
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xc = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        self.scaler = RobustScaler()
        Xs = self.scaler.fit_transform(Xc)
        self.model = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            min_samples_split=self.config.rf_min_samples_split,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(Xs, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xc = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        return self.model.predict_proba(self.scaler.transform(Xc))[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)