#!/usr/bin/env python3

import numpy as np
from typing import Dict, Any
from dataclasses import dataclass, field
from scipy import stats
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, log_loss,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
)


@dataclass
class QuantMetrics:
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    cohen_kappa: float = 0.0
    mcc: float = 0.0
    precision_per_class: Dict[int, float] = field(default_factory=dict)
    recall_per_class: Dict[int, float] = field(default_factory=dict)
    f1_per_class: Dict[int, float] = field(default_factory=dict)
    log_loss_value: float = 0.0
    roc_auc_ovr: float = 0.0
    brier_score: float = 0.0
    crisis_recall: float = 0.0
    crisis_precision: float = 0.0
    regime_transition_accuracy: float = 0.0
    expected_calibration_error: float = 0.0
    early_warning_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                result[k] = {str(kk): float(vv) for kk, vv in v.items()}
            elif isinstance(v, (np.floating, float)):
                result[k] = float(v)
            else:
                result[k] = v
        return result
    
    def summary(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.2%} | Balanced: {self.balanced_accuracy:.2%} | "
            f"Macro-F1: {self.macro_f1:.3f} | Kappa: {self.cohen_kappa:.3f} | "
            f"Crisis Recall: {self.crisis_recall:.2%}"
        )


class MetricsCalculator:
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None,
        num_classes: int = 3
    ) -> QuantMetrics:
        metrics = QuantMetrics()
        
        # Ensure arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        metrics.accuracy = (y_true == y_pred).mean()
        metrics.balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(num_classes)), zero_division=0
        )
        
        metrics.macro_f1 = f1.mean()
        metrics.weighted_f1 = np.average(f1, weights=support) if support.sum() > 0 else 0.0
        
        for i in range(num_classes):
            metrics.precision_per_class[i] = float(precision[i])
            metrics.recall_per_class[i] = float(recall[i])
            metrics.f1_per_class[i] = float(f1[i])
        
        try:
            metrics.cohen_kappa = cohen_kappa_score(y_true, y_pred)
        except:
            metrics.cohen_kappa = 0.0
            
        try:
            metrics.mcc = matthews_corrcoef(y_true, y_pred)
        except:
            metrics.mcc = 0.0
        
        crisis_mask = y_true == 2
        if crisis_mask.sum() > 0:
            metrics.crisis_recall = float(recall[2])
            metrics.crisis_precision = float(precision[2])
        
        if y_prob is not None:
            y_prob = np.asarray(y_prob)
            
            # Ensure y_prob has correct shape
            if len(y_prob.shape) == 1:
                y_prob = y_prob.reshape(-1, 1)
            if y_prob.shape[1] < num_classes:
                full_prob = np.zeros((len(y_prob), num_classes))
                full_prob[:, :y_prob.shape[1]] = y_prob
                y_prob = full_prob
            
            try:
                metrics.log_loss_value = log_loss(y_true, y_prob, labels=list(range(num_classes)))
            except:
                metrics.log_loss_value = float('inf')
            
            try:
                y_true_onehot = np.zeros((len(y_true), num_classes))
                for i, label in enumerate(y_true):
                    y_true_onehot[i, int(label)] = 1
                metrics.roc_auc_ovr = roc_auc_score(
                    y_true_onehot, y_prob, 
                    multi_class='ovr', 
                    average='macro'
                )
            except:
                metrics.roc_auc_ovr = 0.5
            
            metrics.brier_score = MetricsCalculator._brier_multi(y_true, y_prob, num_classes)
            metrics.expected_calibration_error = MetricsCalculator._ece(y_true, y_prob)
        
        metrics.regime_transition_accuracy = MetricsCalculator._transition_accuracy(y_true, y_pred)
        metrics.early_warning_score = MetricsCalculator._early_warning_score(y_true, y_pred)
        
        return metrics
    
    @staticmethod
    def _brier_multi(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
        y_true_onehot = np.zeros((len(y_true), num_classes))
        for i, label in enumerate(y_true):
            y_true_onehot[i, int(label)] = 1
        return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))
    
    @staticmethod
    def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
        
        return ece
    
    @staticmethod
    def _transition_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        transitions = np.where(np.diff(y_true) != 0)[0] + 1
        if len(transitions) == 0:
            return 0.0
        return (y_true[transitions] == y_pred[transitions]).mean()
    
    @staticmethod
    def _early_warning_score(y_true: np.ndarray, y_pred: np.ndarray, lookahead: int = 5) -> float:
        score = 0.0
        count = 0
        
        for i in range(1, len(y_true)):
            if y_true[i] == 2 and y_true[i-1] != 2:
                start = max(0, i - lookahead)
                pre_crisis_preds = y_pred[start:i]
                if len(pre_crisis_preds) > 0:
                    elevated_risk = (pre_crisis_preds >= 1).mean()
                    score += elevated_risk
                    count += 1
        
        return score / max(count, 1)