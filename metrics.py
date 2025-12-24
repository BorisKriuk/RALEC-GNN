#!/usr/bin/env python3
"""
metrics.py - Enhanced with Quantitative Finance Metrics for Regime Detection Research
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats
from scipy.special import rel_entr
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, log_loss,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix, average_precision_score
)
import warnings

warnings.filterwarnings('ignore')


@dataclass
class QuantMetrics:
    """Comprehensive quantitative metrics for regime detection models."""
    
    # Classification Metrics
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    cohen_kappa: float = 0.0
    mcc: float = 0.0
    
    # Per-Class Metrics
    precision_per_class: Dict[int, float] = field(default_factory=dict)
    recall_per_class: Dict[int, float] = field(default_factory=dict)
    f1_per_class: Dict[int, float] = field(default_factory=dict)
    
    # Probabilistic Metrics
    log_loss_value: float = 0.0
    roc_auc_ovr: float = 0.0
    brier_score: float = 0.0
    expected_calibration_error: float = 0.0
    
    # Crisis-Specific Metrics
    crisis_recall: float = 0.0
    crisis_precision: float = 0.0
    crisis_f1: float = 0.0
    crisis_auc: float = 0.0
    crisis_average_precision: float = 0.0
    
    # Regime Transition Metrics
    regime_transition_accuracy: float = 0.0
    early_warning_score: float = 0.0
    transition_lead_time: float = 0.0
    false_alarm_rate: float = 0.0
    
    # Quantitative Finance Metrics
    regime_persistence_accuracy: float = 0.0
    regime_duration_mae: float = 0.0
    information_coefficient: float = 0.0
    hit_rate: float = 0.0
    
    # Tail Risk Metrics
    tail_risk_detection_rate: float = 0.0
    var_breach_detection: float = 0.0
    expected_shortfall_accuracy: float = 0.0
    
    # Economic Value Metrics
    economic_value_added: float = 0.0
    sharpe_improvement: float = 0.0
    max_drawdown_reduction: float = 0.0
    calmar_improvement: float = 0.0
    
    # Stability Metrics
    prediction_stability: float = 0.0
    regime_entropy: float = 0.0
    confidence_when_correct: float = 0.0
    confidence_when_wrong: float = 0.0
    
    # Information Theoretic Metrics
    mutual_information: float = 0.0
    normalized_mutual_information: float = 0.0
    kl_divergence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                result[k] = {str(kk): float(vv) if isinstance(vv, (np.floating, float)) else vv 
                           for kk, vv in v.items()}
            elif isinstance(v, (np.floating, float)):
                result[k] = float(v)
            elif isinstance(v, np.ndarray):
                result[k] = v.tolist()
            else:
                result[k] = v
        return result
    
    def summary(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.2%} | Balanced: {self.balanced_accuracy:.2%} | "
            f"Macro-F1: {self.macro_f1:.3f} | Kappa: {self.cohen_kappa:.3f} | "
            f"Crisis Recall: {self.crisis_recall:.2%} | IC: {self.information_coefficient:.3f}"
        )
    
    def quant_summary(self) -> str:
        """Summary focused on quant-relevant metrics."""
        return (
            f"Hit Rate: {self.hit_rate:.2%} | IC: {self.information_coefficient:.3f} | "
            f"Tail Detection: {self.tail_risk_detection_rate:.2%} | "
            f"Early Warning: {self.early_warning_score:.2%} | "
            f"Sharpe Improvement: {self.sharpe_improvement:.3f}"
        )


class MetricsCalculator:
    """Calculator for comprehensive quantitative finance metrics."""
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None,
        num_classes: int = 3,
        returns: np.ndarray = None,
        volatilities: np.ndarray = None
    ) -> QuantMetrics:
        """
        Calculate all metrics including quant-specific ones.
        
        Args:
            y_true: True regime labels
            y_pred: Predicted regime labels
            y_prob: Predicted probabilities (n_samples, n_classes)
            num_classes: Number of regime classes
            returns: Optional market returns for economic metrics
            volatilities: Optional volatility series for tail risk metrics
        """
        metrics = QuantMetrics()
        
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Basic Classification Metrics
        metrics.accuracy = float((y_true == y_pred).mean())
        metrics.balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(num_classes)), zero_division=0
        )
        
        metrics.macro_f1 = float(f1.mean())
        metrics.weighted_f1 = float(np.average(f1, weights=support) if support.sum() > 0 else 0.0)
        
        for i in range(num_classes):
            metrics.precision_per_class[i] = float(precision[i])
            metrics.recall_per_class[i] = float(recall[i])
            metrics.f1_per_class[i] = float(f1[i])
        
        # Agreement Metrics
        try:
            metrics.cohen_kappa = float(cohen_kappa_score(y_true, y_pred))
        except:
            metrics.cohen_kappa = 0.0
            
        try:
            metrics.mcc = float(matthews_corrcoef(y_true, y_pred))
        except:
            metrics.mcc = 0.0
        
        # Crisis-Specific Metrics (Regime 2)
        crisis_mask = y_true == 2
        if crisis_mask.sum() > 0:
            metrics.crisis_recall = float(recall[2])
            metrics.crisis_precision = float(precision[2])
            metrics.crisis_f1 = float(f1[2])
        
        # Probabilistic Metrics
        if y_prob is not None:
            y_prob = np.asarray(y_prob)
            if len(y_prob.shape) == 1:
                y_prob = y_prob.reshape(-1, 1)
            if y_prob.shape[1] < num_classes:
                full_prob = np.zeros((len(y_prob), num_classes))
                full_prob[:, :y_prob.shape[1]] = y_prob
                y_prob = full_prob
            
            # Normalize probabilities
            y_prob = np.clip(y_prob, 1e-10, 1.0)
            y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
            
            try:
                metrics.log_loss_value = float(log_loss(y_true, y_prob, labels=list(range(num_classes))))
            except:
                metrics.log_loss_value = float('inf')
            
            try:
                y_true_onehot = np.zeros((len(y_true), num_classes))
                for i, label in enumerate(y_true):
                    y_true_onehot[i, int(label)] = 1
                metrics.roc_auc_ovr = float(roc_auc_score(
                    y_true_onehot, y_prob, 
                    multi_class='ovr', 
                    average='macro'
                ))
            except:
                metrics.roc_auc_ovr = 0.5
            
            # Crisis-specific probabilistic metrics
            try:
                crisis_binary = (y_true == 2).astype(int)
                crisis_probs = y_prob[:, 2]
                metrics.crisis_auc = float(roc_auc_score(crisis_binary, crisis_probs))
                metrics.crisis_average_precision = float(average_precision_score(crisis_binary, crisis_probs))
            except:
                metrics.crisis_auc = 0.5
                metrics.crisis_average_precision = 0.0
            
            metrics.brier_score = MetricsCalculator._brier_multi(y_true, y_prob, num_classes)
            metrics.expected_calibration_error = MetricsCalculator._ece(y_true, y_prob)
            
            # Confidence analysis
            confidences = np.max(y_prob, axis=1)
            correct_mask = y_true == y_pred
            metrics.confidence_when_correct = float(confidences[correct_mask].mean()) if correct_mask.sum() > 0 else 0.0
            metrics.confidence_when_wrong = float(confidences[~correct_mask].mean()) if (~correct_mask).sum() > 0 else 0.0
            
            # Information theoretic metrics
            metrics.mutual_information = MetricsCalculator._mutual_information(y_true, y_pred, num_classes)
            metrics.normalized_mutual_information = MetricsCalculator._normalized_mutual_info(y_true, y_pred, num_classes)
            metrics.kl_divergence = MetricsCalculator._kl_divergence(y_true, y_prob, num_classes)
        
        # Regime Transition Metrics
        metrics.regime_transition_accuracy = MetricsCalculator._transition_accuracy(y_true, y_pred)
        metrics.early_warning_score = MetricsCalculator._early_warning_score(y_true, y_pred)
        metrics.transition_lead_time = MetricsCalculator._transition_lead_time(y_true, y_pred, y_prob)
        metrics.false_alarm_rate = MetricsCalculator._false_alarm_rate(y_true, y_pred)
        
        # Regime Persistence Metrics
        metrics.regime_persistence_accuracy = MetricsCalculator._regime_persistence_accuracy(y_true, y_pred)
        metrics.regime_duration_mae = MetricsCalculator._regime_duration_mae(y_true, y_pred)
        
        # Prediction Stability
        metrics.prediction_stability = MetricsCalculator._prediction_stability(y_pred)
        metrics.regime_entropy = MetricsCalculator._regime_entropy(y_pred)
        
        # Quantitative Finance Metrics
        metrics.information_coefficient = MetricsCalculator._information_coefficient(y_true, y_pred, num_classes)
        metrics.hit_rate = MetricsCalculator._hit_rate(y_true, y_pred)
        
        # Tail Risk Metrics
        if volatilities is not None:
            metrics.tail_risk_detection_rate = MetricsCalculator._tail_risk_detection(
                y_true, y_pred, volatilities
            )
            metrics.var_breach_detection = MetricsCalculator._var_breach_detection(
                y_true, y_pred, volatilities
            )
        
        # Economic Value Metrics
        if returns is not None:
            econ_metrics = MetricsCalculator._economic_value_metrics(
                y_true, y_pred, y_prob, returns
            )
            metrics.economic_value_added = econ_metrics['eva']
            metrics.sharpe_improvement = econ_metrics['sharpe_improvement']
            metrics.max_drawdown_reduction = econ_metrics['mdd_reduction']
            metrics.calmar_improvement = econ_metrics['calmar_improvement']
        
        return metrics
    
    @staticmethod
    def _brier_multi(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
        y_true_onehot = np.zeros((len(y_true), num_classes))
        for i, label in enumerate(y_true):
            y_true_onehot[i, int(label)] = 1
        return float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))
    
    @staticmethod
    def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
        """Expected Calibration Error with adaptive binning."""
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
        
        return float(ece)
    
    @staticmethod
    def _transition_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Accuracy specifically at regime transition points."""
        transitions = np.where(np.diff(y_true) != 0)[0] + 1
        if len(transitions) == 0:
            return 0.0
        return float((y_true[transitions] == y_pred[transitions]).mean())
    
    @staticmethod
    def _early_warning_score(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        lookahead: int = 5
    ) -> float:
        """Score for detecting elevated risk before crisis onset."""
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
        
        return float(score / max(count, 1))
    
    @staticmethod
    def _transition_lead_time(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: np.ndarray = None,
        prob_threshold: float = 0.3
    ) -> float:
        """Average lead time (in periods) before crisis detection."""
        if y_prob is None:
            return 0.0
        
        lead_times = []
        
        for i in range(1, len(y_true)):
            if y_true[i] == 2 and y_true[i-1] != 2:
                # Look back to find when crisis probability first exceeded threshold
                for j in range(i-1, max(0, i-20), -1):
                    if y_prob[j, 2] >= prob_threshold:
                        lead_times.append(i - j)
                        break
        
        return float(np.mean(lead_times)) if lead_times else 0.0
    
    @staticmethod
    def _false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Rate of false crisis predictions."""
        crisis_preds = y_pred == 2
        actual_crisis = y_true == 2
        
        false_alarms = crisis_preds & ~actual_crisis
        total_non_crisis = (~actual_crisis).sum()
        
        return float(false_alarms.sum() / max(total_non_crisis, 1))
    
    @staticmethod
    def _regime_persistence_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Accuracy during regime persistence (non-transition) periods."""
        non_transitions = np.ones(len(y_true), dtype=bool)
        transitions = np.where(np.diff(y_true) != 0)[0] + 1
        
        # Mark transition points and surrounding periods
        for t in transitions:
            start = max(0, t - 2)
            end = min(len(y_true), t + 3)
            non_transitions[start:end] = False
        
        if non_transitions.sum() == 0:
            return 0.0
        
        return float((y_true[non_transitions] == y_pred[non_transitions]).mean())
    
    @staticmethod
    def _regime_duration_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean absolute error of predicted regime durations."""
        def get_durations(labels):
            durations = []
            current_regime = labels[0]
            current_duration = 1
            
            for i in range(1, len(labels)):
                if labels[i] == current_regime:
                    current_duration += 1
                else:
                    durations.append((current_regime, current_duration))
                    current_regime = labels[i]
                    current_duration = 1
            durations.append((current_regime, current_duration))
            return durations
        
        true_durations = get_durations(y_true)
        pred_durations = get_durations(y_pred)
        
        # Compare average durations per regime
        true_avg = {r: np.mean([d for reg, d in true_durations if reg == r]) 
                   for r in range(3) if any(reg == r for reg, d in true_durations)}
        pred_avg = {r: np.mean([d for reg, d in pred_durations if reg == r]) 
                   for r in range(3) if any(reg == r for reg, d in pred_durations)}
        
        errors = []
        for r in true_avg:
            if r in pred_avg:
                errors.append(abs(true_avg[r] - pred_avg[r]))
        
        return float(np.mean(errors)) if errors else 0.0
    
    @staticmethod
    def _prediction_stability(y_pred: np.ndarray) -> float:
        """Measure of prediction stability (fewer flips = more stable)."""
        if len(y_pred) < 2:
            return 1.0
        
        flips = np.sum(np.diff(y_pred) != 0)
        max_flips = len(y_pred) - 1
        
        return float(1.0 - flips / max_flips)
    
    @staticmethod
    def _regime_entropy(y_pred: np.ndarray) -> float:
        """Entropy of predicted regime distribution."""
        unique, counts = np.unique(y_pred, return_counts=True)
        probs = counts / len(y_pred)
        return float(-np.sum(probs * np.log(probs + 1e-10)))
    
    @staticmethod
    def _information_coefficient(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        num_classes: int
    ) -> float:
        """
        Information Coefficient - correlation between predictions and outcomes.
        Adapted for classification as rank correlation.
        """
        try:
            # Spearman correlation between true and predicted regimes
            ic, _ = stats.spearmanr(y_true, y_pred)
            return float(ic) if not np.isnan(ic) else 0.0
        except:
            return 0.0
    
    @staticmethod
    def _hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Hit rate - percentage of correct directional predictions.
        For regimes: correctly predicting risk level direction.
        """
        if len(y_true) < 2:
            return 0.0
        
        # Direction of change in risk level
        true_direction = np.sign(np.diff(y_true.astype(float)))
        pred_direction = np.sign(np.diff(y_pred.astype(float)))
        
        # Only count where there was actual movement
        movement_mask = true_direction != 0
        if movement_mask.sum() == 0:
            return 0.0
        
        hits = (true_direction[movement_mask] == pred_direction[movement_mask]).sum()
        return float(hits / movement_mask.sum())
    
    @staticmethod
    def _tail_risk_detection(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        volatilities: np.ndarray,
        tail_percentile: float = 95
    ) -> float:
        """Detection rate of tail risk events (high volatility periods)."""
        vol_threshold = np.percentile(volatilities, tail_percentile)
        tail_events = volatilities >= vol_threshold
        
        # Check if model predicted elevated risk (regime 1 or 2) during tail events
        elevated_pred = y_pred >= 1
        
        if tail_events.sum() == 0:
            return 0.0
        
        return float((elevated_pred & tail_events).sum() / tail_events.sum())
    
    @staticmethod
    def _var_breach_detection(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        volatilities: np.ndarray,
        var_level: float = 0.99
    ) -> float:
        """Detection rate of VaR breach events."""
        # Approximate VaR breaches as extreme volatility events
        var_threshold = np.percentile(volatilities, var_level * 100)
        var_breaches = volatilities >= var_threshold
        
        # Crisis prediction should precede or coincide with VaR breaches
        crisis_pred = y_pred == 2
        
        if var_breaches.sum() == 0:
            return 0.0
        
        # Check if crisis was predicted at breach or one period before
        detected = 0
        for i in np.where(var_breaches)[0]:
            if crisis_pred[i] or (i > 0 and crisis_pred[i-1]):
                detected += 1
        
        return float(detected / var_breaches.sum())
    
    @staticmethod
    def _economic_value_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """Calculate economic value of predictions."""
        returns = np.asarray(returns).flatten()
        
        # Ensure alignment
        min_len = min(len(returns), len(y_pred))
        returns = returns[:min_len]
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]
        
        # Strategy: reduce exposure in predicted high-risk regimes
        # Regime 0: 100% exposure, Regime 1: 50% exposure, Regime 2: 0% exposure
        exposure = np.where(y_pred == 0, 1.0, np.where(y_pred == 1, 0.5, 0.0))
        
        # Perfect foresight strategy
        perfect_exposure = np.where(y_true == 0, 1.0, np.where(y_true == 1, 0.5, 0.0))
        
        # Calculate strategy returns
        strategy_returns = returns * exposure
        perfect_returns = returns * perfect_exposure
        buy_hold_returns = returns
        
        # Sharpe ratios (annualized)
        def sharpe(rets):
            if len(rets) < 2 or np.std(rets) == 0:
                return 0.0
            return float(np.mean(rets) / np.std(rets) * np.sqrt(252))
        
        strategy_sharpe = sharpe(strategy_returns)
        buyhold_sharpe = sharpe(buy_hold_returns)
        perfect_sharpe = sharpe(perfect_returns)
        
        # Max drawdown
        def max_drawdown(rets):
            cumulative = np.cumprod(1 + rets)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            return float(np.min(drawdowns))
        
        strategy_mdd = max_drawdown(strategy_returns)
        buyhold_mdd = max_drawdown(buy_hold_returns)
        
        # Economic Value Added (strategy return - buy&hold return)
        eva = float(np.sum(strategy_returns) - np.sum(buy_hold_returns))
        
        # Calmar ratio improvement
        def calmar(rets):
            mdd = abs(max_drawdown(rets))
            if mdd == 0:
                return 0.0
            return float(np.mean(rets) * 252 / mdd)
        
        strategy_calmar = calmar(strategy_returns)
        buyhold_calmar = calmar(buy_hold_returns)
        
        return {
            'eva': eva,
            'sharpe_improvement': strategy_sharpe - buyhold_sharpe,
            'mdd_reduction': buyhold_mdd - strategy_mdd,  # Positive is better
            'calmar_improvement': strategy_calmar - buyhold_calmar
        }
    
    @staticmethod
    def _mutual_information(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int
    ) -> float:
        """Mutual information between true and predicted labels."""
        try:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            cm = cm.astype(float)
            cm /= cm.sum()
            
            # Marginals
            p_true = cm.sum(axis=1)
            p_pred = cm.sum(axis=0)
            
            mi = 0.0
            for i in range(num_classes):
                for j in range(num_classes):
                    if cm[i, j] > 0 and p_true[i] > 0 and p_pred[j] > 0:
                        mi += cm[i, j] * np.log(cm[i, j] / (p_true[i] * p_pred[j]))
            
            return float(mi)
        except:
            return 0.0
    
    @staticmethod
    def _normalized_mutual_info(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int
    ) -> float:
        """Normalized mutual information."""
        mi = MetricsCalculator._mutual_information(y_true, y_pred, num_classes)
        
        # Entropies
        def entropy(labels):
            _, counts = np.unique(labels, return_counts=True)
            probs = counts / len(labels)
            return -np.sum(probs * np.log(probs + 1e-10))
        
        h_true = entropy(y_true)
        h_pred = entropy(y_pred)
        
        if h_true == 0 or h_pred == 0:
            return 0.0
        
        return float(2 * mi / (h_true + h_pred))
    
    @staticmethod
    def _kl_divergence(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        num_classes: int
    ) -> float:
        """KL divergence from true distribution to predicted."""
        try:
            # True distribution
            true_dist = np.zeros(num_classes)
            for label in y_true:
                true_dist[int(label)] += 1
            true_dist /= len(y_true)
            
            # Average predicted distribution
            pred_dist = y_prob.mean(axis=0)
            pred_dist = np.clip(pred_dist, 1e-10, 1.0)
            pred_dist /= pred_dist.sum()
            
            kl = np.sum(rel_entr(true_dist + 1e-10, pred_dist))
            return float(kl)
        except:
            return 0.0


class RegimeMetricsReport:
    """Generate comprehensive metrics reports for regime detection."""
    
    @staticmethod
    def generate_report(
        metrics: QuantMetrics,
        model_name: str = "Model"
    ) -> str:
        """Generate a formatted report of all metrics."""
        
        report = f"""
{'='*80}
REGIME DETECTION METRICS REPORT: {model_name}
{'='*80}

CLASSIFICATION PERFORMANCE
--------------------------
  Accuracy:              {metrics.accuracy:.2%}
  Balanced Accuracy:     {metrics.balanced_accuracy:.2%}
  Macro F1:              {metrics.macro_f1:.4f}
  Weighted F1:           {metrics.weighted_f1:.4f}
  Cohen's Kappa:         {metrics.cohen_kappa:.4f}
  MCC:                   {metrics.mcc:.4f}

PER-CLASS METRICS
-----------------
  Bull/Low Vol (0):   Precision={metrics.precision_per_class.get(0, 0):.2%}  Recall={metrics.recall_per_class.get(0, 0):.2%}  F1={metrics.f1_per_class.get(0, 0):.4f}
  Normal (1):         Precision={metrics.precision_per_class.get(1, 0):.2%}  Recall={metrics.recall_per_class.get(1, 0):.2%}  F1={metrics.f1_per_class.get(1, 0):.4f}
  Crisis (2):         Precision={metrics.precision_per_class.get(2, 0):.2%}  Recall={metrics.recall_per_class.get(2, 0):.2%}  F1={metrics.f1_per_class.get(2, 0):.4f}

CRISIS DETECTION (Critical for Risk Management)
-----------------------------------------------
  Crisis Recall:               {metrics.crisis_recall:.2%}
  Crisis Precision:            {metrics.crisis_precision:.2%}
  Crisis F1:                   {metrics.crisis_f1:.4f}
  Crisis AUC:                  {metrics.crisis_auc:.4f}
  Crisis Average Precision:    {metrics.crisis_average_precision:.4f}
  False Alarm Rate:            {metrics.false_alarm_rate:.2%}

PROBABILISTIC CALIBRATION
-------------------------
  Log Loss:                    {metrics.log_loss_value:.4f}
  ROC-AUC (OvR):              {metrics.roc_auc_ovr:.4f}
  Brier Score:                 {metrics.brier_score:.4f}
  ECE:                         {metrics.expected_calibration_error:.4f}
  Confidence (Correct):        {metrics.confidence_when_correct:.2%}
  Confidence (Wrong):          {metrics.confidence_when_wrong:.2%}

REGIME DYNAMICS
---------------
  Transition Accuracy:         {metrics.regime_transition_accuracy:.2%}
  Persistence Accuracy:        {metrics.regime_persistence_accuracy:.2%}
  Regime Duration MAE:         {metrics.regime_duration_mae:.2f} periods
  Prediction Stability:        {metrics.prediction_stability:.2%}
  Regime Entropy:              {metrics.regime_entropy:.4f}

EARLY WARNING & TAIL RISK
-------------------------
  Early Warning Score:         {metrics.early_warning_score:.2%}
  Transition Lead Time:        {metrics.transition_lead_time:.2f} periods
  Tail Risk Detection:         {metrics.tail_risk_detection_rate:.2%}
  VaR Breach Detection:        {metrics.var_breach_detection:.2%}

QUANTITATIVE FINANCE METRICS
----------------------------
  Information Coefficient:     {metrics.information_coefficient:.4f}
  Hit Rate:                    {metrics.hit_rate:.2%}
  Normalized MI:               {metrics.normalized_mutual_information:.4f}
  KL Divergence:               {metrics.kl_divergence:.4f}

ECONOMIC VALUE (if available)
-----------------------------
  Economic Value Added:        {metrics.economic_value_added:.4f}
  Sharpe Improvement:          {metrics.sharpe_improvement:.4f}
  Max DD Reduction:            {metrics.max_drawdown_reduction:.4f}
  Calmar Improvement:          {metrics.calmar_improvement:.4f}

{'='*80}
"""
        return report
    
    @staticmethod
    def compare_models(
        metrics_dict: Dict[str, QuantMetrics],
        key_metrics: List[str] = None
    ) -> str:
        """Generate comparison table for multiple models."""
        
        if key_metrics is None:
            key_metrics = [
                'accuracy', 'balanced_accuracy', 'macro_f1', 'crisis_recall',
                'crisis_precision', 'early_warning_score', 'information_coefficient',
                'roc_auc_ovr', 'false_alarm_rate'
            ]
        
        models = list(metrics_dict.keys())
        
        # Header
        header = f"{'Metric':<30}" + "".join([f"{m:<20}" for m in models])
        separator = "=" * (30 + 20 * len(models))
        
        report = f"\n{separator}\nMODEL COMPARISON\n{separator}\n{header}\n{'-' * len(header)}\n"
        
        for metric in key_metrics:
            row = f"{metric:<30}"
            for model in models:
                value = getattr(metrics_dict[model], metric, 0)
                if isinstance(value, float):
                    if 'rate' in metric or 'accuracy' in metric or 'recall' in metric or 'precision' in metric:
                        row += f"{value:<20.2%}"
                    else:
                        row += f"{value:<20.4f}"
                else:
                    row += f"{str(value):<20}"
            report += row + "\n"
        
        report += separator + "\n"
        
        return report