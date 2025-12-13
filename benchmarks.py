#!/usr/bin/env python3
"""
Optimized benchmarks with optional subsampling for faster runs.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from torch_geometric.data import Data

from metrics import MetricsCalculator

logger = logging.getLogger(__name__)

SEED = 42


class PurgedTimeSeriesSplit:
    def __init__(self, n_splits: int = 5, purge_gap: int = 5, min_train_size: int = 100):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.min_train_size = min_train_size
    
    def split(self, X):
        n_samples = len(X) if hasattr(X, '__len__') else X
        fold_size = (n_samples - self.min_train_size) // self.n_splits
        
        for i in range(self.n_splits):
            train_end = self.min_train_size + i * fold_size
            val_start = train_end + self.purge_gap
            val_end = min(val_start + fold_size, n_samples)
            
            if val_start >= n_samples:
                break
                
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            
            yield train_idx, val_idx


class BaselineModels:
    def __init__(self, config):
        self.config = config
        
    def prepare_features(
        self,
        graphs: List[Data],
        regime_df: pd.DataFrame,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features from graph sequence.
        
        Args:
            graphs: List of graph snapshots
            regime_df: DataFrame with regime labels
            stride: Skip every N samples (1 = no skip, 2 = use every other, etc.)
                   Useful for faster baseline training without losing much accuracy.
        """
        regime_df = regime_df.copy()
        regime_df['date'] = pd.to_datetime(regime_df['date'])
        
        features_list = []
        labels = []
        
        indices = range(0, len(graphs) - self.config.seq_len, stride)
        
        for i in indices:
            seq = graphs[i:i + self.config.seq_len]
            ts = seq[-1].timestamp
            
            match = regime_df[regime_df['date'] <= ts]
            if len(match) > 0:
                seq_features = []
                for g in seq:
                    node_mean = g.x.mean(dim=0).numpy()
                    node_std = g.x.std(dim=0).numpy()
                    seq_features.extend(node_mean)
                    seq_features.extend(node_std)
                
                features_list.append(seq_features)
                labels.append(match.iloc[-1]['regime'])
        
        return np.array(features_list), np.array(labels)
    
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        fast_mode: bool = False
    ) -> Dict[str, Dict]:
        """
        Train and evaluate baseline models.
        
        Args:
            X: Feature matrix
            y: Labels
            n_splits: Number of CV folds
            fast_mode: If True, use fewer trees and shallower models for speed
        """
        results = {}
        
        if fast_mode:
            # Faster but slightly less accurate
            models = {
                'logistic_regression': LogisticRegression(
                    max_iter=500, 
                    multi_class='multinomial',
                    class_weight='balanced',
                    random_state=SEED,
                    solver='lbfgs'
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=50,  # Reduced from 100
                    max_depth=8,      # Reduced from 10
                    class_weight='balanced',
                    random_state=SEED,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=50,  # Reduced from 100
                    max_depth=4,      # Reduced from 5
                    random_state=SEED,
                    subsample=0.8     # Stochastic gradient boosting
                )
            }
        else:
            models = {
                'logistic_regression': LogisticRegression(
                    max_iter=1000, 
                    multi_class='multinomial',
                    class_weight='balanced',
                    random_state=SEED
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight='balanced',
                    random_state=SEED,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=SEED
                )
            }
        
        tscv = PurgedTimeSeriesSplit(
            n_splits=n_splits, 
            purge_gap=self.config.purge_gap, 
            min_train_size=50
        )
        
        for name, model in models.items():
            logger.info(f"Training baseline: {name}" + (" [FAST MODE]" if fast_mode else ""))
            
            fold_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(len(X))):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Handle NaN/Inf
                X_train = np.nan_to_num(X_train, nan=0, posinf=1e10, neginf=-1e10)
                X_val = np.nan_to_num(X_val, nan=0, posinf=1e10, neginf=-1e10)
                
                # Clip extreme values
                X_train = np.clip(X_train, -1e10, 1e10)
                X_val = np.clip(X_val, -1e10, 1e10)
                
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                
                preds = model_clone.predict(X_val)
                probs = model_clone.predict_proba(X_val)
                
                fold_m = MetricsCalculator.calculate_all_metrics(
                    y_val, preds, probs, self.config.num_regimes
                )
                fold_metrics.append(fold_m)
            
            results[name] = {
                'accuracy_mean': np.mean([m.accuracy for m in fold_metrics]),
                'accuracy_std': np.std([m.accuracy for m in fold_metrics]),
                'balanced_accuracy_mean': np.mean([m.balanced_accuracy for m in fold_metrics]),
                'balanced_accuracy_std': np.std([m.balanced_accuracy for m in fold_metrics]),
                'macro_f1_mean': np.mean([m.macro_f1 for m in fold_metrics]),
                'macro_f1_std': np.std([m.macro_f1 for m in fold_metrics]),
                'crisis_recall_mean': np.mean([m.crisis_recall for m in fold_metrics]),
                'crisis_recall_std': np.std([m.crisis_recall for m in fold_metrics]),
                'cohen_kappa_mean': np.mean([m.cohen_kappa for m in fold_metrics]),
                'cohen_kappa_std': np.std([m.cohen_kappa for m in fold_metrics]),
                'fold_metrics': [m.to_dict() for m in fold_metrics]
            }
            
            logger.info(
                f"  {name}: Acc={results[name]['accuracy_mean']:.2%}±{results[name]['accuracy_std']:.2%}, "
                f"Crisis Recall={results[name]['crisis_recall_mean']:.2%}±{results[name]['crisis_recall_std']:.2%}, "
                f"Macro-F1={results[name]['macro_f1_mean']:.3f}±{results[name]['macro_f1_std']:.3f}"
            )
        
        return results