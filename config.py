#!/usr/bin/env python3
"""Configuration for CGECD Experiment v4"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Main configuration"""

    # API
    api_key: str = field(
        default_factory=lambda: os.getenv("EODHD_API_KEY", "")
        or os.getenv("API_KEY", "")
    )

    # Data
    years: int = 15
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("results")

    # Random Forest
    rf_n_estimators: int = 500
    rf_max_depth: int = 6
    rf_min_samples_leaf: int = 20
    rf_min_samples_split: int = 50

    # XGBoost (kept for benchmarks / future use)
    xgb_n_estimators: int = 400
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.7
    xgb_min_child_weight: int = 10
    xgb_gamma: float = 1.5
    xgb_reg_alpha: float = 0.3
    xgb_reg_lambda: float = 1.0

    # SVM
    svm_C: float = 1.0
    svm_gamma: str = "scale"

    random_seed: int = 42

    # Spectral parameters (single 60d window)
    correlation_window: int = 60
    graph_threshold: float = 0.3
    dynamics_lookbacks: tuple = (10, 20)

    # Feature selection
    feature_selection_k: int = 40

    # Adaptive ensemble — RF + Vol-LR correction
    vol_correction_scale: float = 15.0
    vol_correction_threshold: float = 0.01
    vol_correction_max: float = 0.85

    # Walk-forward validation (expanding window)
    n_splits: int = 20
    train_years: float = 3.0
    test_months: int = 6
    gap_days: int = 5

    # Asset universe
    symbols: Dict[str, str] = field(
        default_factory=lambda: {
            "SPY.US": "SP500",
            "QQQ.US": "Nasdaq100",
            "IWM.US": "Russell2000",
            "XLF.US": "Financials",
            "XLE.US": "Energy",
            "XLK.US": "Technology",
            "XLV.US": "Healthcare",
            "XLU.US": "Utilities",
            "XLP.US": "ConsumerStaples",
            "XLY.US": "ConsumerDisc",
            "XLI.US": "Industrials",
            "XLB.US": "Materials",
            "XLRE.US": "RealEstate",
            "EFA.US": "DevIntl",
            "EEM.US": "EmergingMkts",
            "VGK.US": "Europe",
            "EWJ.US": "Japan",
            "TLT.US": "LongTreasury",
            "IEF.US": "IntermTreasury",
            "LQD.US": "InvGradeCorp",
            "HYG.US": "HighYield",
            "GLD.US": "Gold",
            "USO.US": "Oil",
            "UUP.US": "USDollar",
            "VNQ.US": "REITs",
        }
    )

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.output_dir = Path(self.output_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        if not self.api_key:
            raise ValueError(
                "EODHD_API_KEY not found – set it in .env or as an env-var."
            )