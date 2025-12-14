#!/usr/bin/env python3
"""
benchmarks.py - Fixed with proper crisis-aware cross-validation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from torch_geometric.data import Data

from metrics import MetricsCalculator

logger = logging.getLogger(__name__)

SEED = 42


class CrisisAwareTimeSeriesSplit:
    """
    Time series split that ensures minimum crisis samples in both train and validation.
    Uses later portions of data where crisis events are more likely to be in history.
    """
    
    def __init__(
        self, 
        n_splits: int = 5, 
        purge_gap: int = 5,
        min_crisis_train: int = 20,
        min_crisis_val: int = 5,
        val_ratio: float = 0.15
    ):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.min_crisis_train = min_crisis_train
        self.min_crisis_val = min_crisis_val
        self.val_ratio = val_ratio
    
    def split(self, n_samples, labels=None):
        if labels is None:
            raise ValueError("Labels required for CrisisAwareTimeSeriesSplit")
        
        labels = np.array(labels)
        
        # Find where we have enough crisis samples in history
        crisis_cumsum = np.cumsum(labels == 2)
        
        # Find earliest point where we have min_crisis_train in history
        valid_train_starts = np.where(crisis_cumsum >= self.min_crisis_train)[0]
        if len(valid_train_starts) == 0:
            logger.warning(f"Not enough crisis samples. Total crisis: {(labels == 2).sum()}")
            # Fall back to simple split
            min_train = int(n_samples * 0.5)
            val_size = int(n_samples * self.val_ratio)
            for i in range(self.n_splits):
                train_end = min_train + i * val_size
                if train_end >= n_samples - val_size:
                    break
                val_start = train_end + self.purge_gap
                val_end = min(val_start + val_size, n_samples)
                yield np.arange(0, train_end), np.arange(val_start, val_end)
            return
        
        min_train_idx = valid_train_starts[0]
        
        # Calculate fold parameters
        remaining_samples = n_samples - min_train_idx
        val_size = max(int(remaining_samples * self.val_ratio), 30)
        
        # Space out folds
        step_size = (n_samples - min_train_idx - val_size) // self.n_splits
        
        folds_generated = 0
        for i in range(self.n_splits):
            train_end = min_train_idx + i * step_size
            val_start = train_end + self.purge_gap
            val_end = min(val_start + val_size, n_samples)
            
            if val_start >= n_samples or val_end <= val_start:
                continue
            
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            
            # Check crisis samples
            train_crisis = (labels[train_idx] == 2).sum()
            val_crisis = (labels[val_idx] == 2).sum()
            
            # If validation has no crisis, try to expand it
            if val_crisis < self.min_crisis_val:
                # Try expanding validation window
                for expand in range(5, 50, 5):
                    new_val_end = min(val_end + expand, n_samples)
                    expanded_val_idx = np.arange(val_start, new_val_end)
                    new_val_crisis = (labels[expanded_val_idx] == 2).sum()
                    if new_val_crisis >= self.min_crisis_val:
                        val_idx = expanded_val_idx
                        val_crisis = new_val_crisis
                        break
            
            logger.info(f"Fold {folds_generated+1}: train={len(train_idx)} (crisis={train_crisis}), "
                       f"val={len(val_idx)} (crisis={val_crisis})")
            
            yield train_idx, val_idx
            folds_generated += 1


class BaselineModels:
    def __init__(self, config):
        self.config = config
        
    def prepare_features(
        self,
        graphs: List[Data],
        regime_df: pd.DataFrame,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features from graph sequence."""
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
                    node_max = g.x.max(dim=0).values.numpy()
                    node_min = g.x.min(dim=0).values.numpy()
                    seq_features.extend(node_mean)
                    seq_features.extend(node_std)
                    seq_features.extend(node_max)
                    seq_features.extend(node_min)
                
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
        """Train and evaluate baseline models."""
        results = {}
        
        if fast_mode:
            models = {
                'logistic_regression': LogisticRegression(
                    max_iter=500, 
                    multi_class='multinomial',
                    class_weight='balanced',
                    random_state=SEED,
                    solver='lbfgs'
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=50,
                    max_depth=8,
                    class_weight='balanced',
                    random_state=SEED,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=4,
                    random_state=SEED,
                    subsample=0.8
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
                    n_estimators=200,
                    max_depth=12,
                    class_weight='balanced_subsample',
                    random_state=SEED,
                    n_jobs=-1,
                    min_samples_leaf=2
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=6,
                    random_state=SEED,
                    learning_rate=0.05,
                    subsample=0.8
                )
            }
        
        # Use crisis-aware split
        tscv = CrisisAwareTimeSeriesSplit(
            n_splits=n_splits, 
            purge_gap=self.config.purge_gap,
            min_crisis_train=20,
            min_crisis_val=5
        )
        
        for name, model in models.items():
            logger.info(f"Training baseline: {name}" + (" [FAST MODE]" if fast_mode else ""))
            
            fold_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(len(X), y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Handle NaN/Inf
                X_train = np.nan_to_num(X_train, nan=0, posinf=1e10, neginf=-1e10)
                X_val = np.nan_to_num(X_val, nan=0, posinf=1e10, neginf=-1e10)
                X_train = np.clip(X_train, -1e10, 1e10)
                X_val = np.clip(X_val, -1e10, 1e10)
                
                # Check if we have enough classes
                train_classes = np.unique(y_train)
                val_classes = np.unique(y_val)
                
                if len(train_classes) < 2:
                    logger.warning(f"Fold {fold+1} skipped: only {len(train_classes)} class(es) in train")
                    continue
                
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                
                preds = model_clone.predict(X_val)
                probs = model_clone.predict_proba(X_val)
                
                # Ensure probs has correct shape for all 3 classes
                if probs.shape[1] < 3:
                    full_probs = np.zeros((len(probs), 3))
                    for i, c in enumerate(model_clone.classes_):
                        full_probs[:, int(c)] = probs[:, i]
                    probs = full_probs
                
                fold_m = MetricsCalculator.calculate_all_metrics(
                    y_val, preds, probs, self.config.num_regimes
                )
                fold_metrics.append(fold_m)
            
            if not fold_metrics:
                logger.warning(f"No valid folds for {name}")
                continue
            
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