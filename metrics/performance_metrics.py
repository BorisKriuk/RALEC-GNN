"""Performance evaluation metrics for RALEC-GNN."""

import torch
import numpy as np
from typing import Dict, Union, Tuple, Optional, List, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class PerformanceEvaluator:
    """Calculate comprehensive performance metrics."""
    
    def __init__(self):
        self.regime_names = ['Normal', 'Bull', 'Bear', 'Crisis', 'Recovery']
        
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        return_detailed: bool = True
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            predictions: Model predictions (logits or probabilities)
            labels: True labels
            return_detailed: Whether to return detailed metrics
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy
        if predictions.dim() > 1:
            preds = predictions.argmax(dim=1).cpu().numpy()
        else:
            preds = predictions.cpu().numpy()
            
        labels = labels.cpu().numpy()
        
        # Basic metrics
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        
        # Crisis-specific metrics
        crisis_metrics = self._calculate_crisis_metrics(preds, labels)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            **crisis_metrics
        }
        
        if return_detailed:
            # Per-class metrics
            per_class_metrics = self._calculate_per_class_metrics(preds, labels)
            metrics['per_class'] = per_class_metrics
            
            # Confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(labels, preds).tolist()
            
            # Transition accuracy
            metrics['transition_accuracy'] = self._calculate_transition_accuracy(
                preds, labels
            )
            
        return metrics
    
    def _calculate_crisis_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate crisis-specific metrics."""
        # Define crisis regimes (2=Bear, 3=Crisis)
        crisis_regimes = [2, 3]
        
        # Binary classification: crisis vs non-crisis
        pred_crisis = np.isin(predictions, crisis_regimes)
        true_crisis = np.isin(labels, crisis_regimes)
        
        # Crisis detection metrics
        true_positives = np.sum(pred_crisis & true_crisis)
        false_positives = np.sum(pred_crisis & ~true_crisis)
        false_negatives = np.sum(~pred_crisis & true_crisis)
        true_negatives = np.sum(~pred_crisis & ~true_crisis)
        
        # Calculate rates
        if (true_positives + false_negatives) > 0:
            crisis_recall = true_positives / (true_positives + false_negatives)
        else:
            crisis_recall = 0.0
            
        if (true_positives + false_positives) > 0:
            crisis_precision = true_positives / (true_positives + false_positives)
        else:
            crisis_precision = 0.0
            
        if (crisis_precision + crisis_recall) > 0:
            crisis_f1 = 2 * crisis_precision * crisis_recall / (crisis_precision + crisis_recall)
        else:
            crisis_f1 = 0.0
            
        return {
            'crisis_recall': crisis_recall,
            'crisis_precision': crisis_precision,
            'crisis_f1': crisis_f1,
            'crisis_true_positives': int(true_positives),
            'crisis_false_positives': int(false_positives),
            'crisis_false_negatives': int(false_negatives)
        }
    
    def _calculate_per_class_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Calculate per-regime metrics."""
        per_class_metrics = {}
        
        for i, regime_name in enumerate(self.regime_names):
            # Binary classification for this regime
            pred_regime = predictions == i
            true_regime = labels == i
            
            tp = np.sum(pred_regime & true_regime)
            fp = np.sum(pred_regime & ~true_regime)
            fn = np.sum(~pred_regime & true_regime)
            tn = np.sum(~pred_regime & ~true_regime)
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_metrics[regime_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': int(np.sum(true_regime))
            }
            
        return per_class_metrics
    
    def _calculate_transition_accuracy(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Calculate accuracy of regime transition predictions."""
        if len(predictions) < 2:
            return 0.0
            
        # Identify transitions
        pred_transitions = np.diff(predictions) != 0
        true_transitions = np.diff(labels) != 0
        
        # Transition detection accuracy
        correct_transitions = pred_transitions == true_transitions
        transition_accuracy = np.mean(correct_transitions)
        
        return float(transition_accuracy)
    
    def calculate_lead_time(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate average lead time for crisis prediction.
        
        Args:
            predictions: Model predictions
            labels: True labels
            timestamps: Optional timestamps for each prediction
            
        Returns:
            Average lead time in time units
        """
        crisis_regimes = [2, 3]  # Bear and Crisis
        lead_times = []
        
        for i in range(len(predictions)):
            # Check if we predict a crisis
            if predictions[i] in crisis_regimes:
                # Look ahead for actual crisis
                for j in range(i, min(i + 30, len(labels))):  # Look up to 30 steps ahead
                    if labels[j] in crisis_regimes:
                        lead_time = j - i
                        if timestamps is not None:
                            lead_time = timestamps[j] - timestamps[i]
                        lead_times.append(lead_time)
                        break
                        
        if lead_times:
            return float(np.mean(lead_times))
        return 0.0
        
    def format_metrics_report(self, metrics: Dict[str, Any]) -> str:
        """Format metrics into readable report."""
        report = "\nPerformance Metrics Report\n"
        report += "=" * 50 + "\n\n"
        
        # Overall metrics
        report += "Overall Performance:\n"
        report += f"  Accuracy:          {metrics['accuracy']:.4f}\n"
        report += f"  Precision:         {metrics['precision']:.4f}\n"
        report += f"  Recall:            {metrics['recall']:.4f}\n"
        report += f"  F1-Score:          {metrics['f1']:.4f}\n\n"
        
        # Crisis-specific metrics
        report += "Crisis Detection Performance:\n"
        report += f"  Crisis Recall:     {metrics['crisis_recall']:.4f}\n"
        report += f"  Crisis Precision:  {metrics['crisis_precision']:.4f}\n"
        report += f"  Crisis F1-Score:   {metrics['crisis_f1']:.4f}\n"
        report += f"  True Positives:    {metrics['crisis_true_positives']}\n"
        report += f"  False Positives:   {metrics['crisis_false_positives']}\n"
        report += f"  False Negatives:   {metrics['crisis_false_negatives']}\n\n"
        
        # Per-regime metrics if available
        if 'per_class' in metrics:
            report += "Per-Regime Performance:\n"
            report += "-" * 50 + "\n"
            report += f"{'Regime':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n"
            report += "-" * 50 + "\n"
            
            for regime, regime_metrics in metrics['per_class'].items():
                report += f"{regime:<12} "
                report += f"{regime_metrics['precision']:<10.3f} "
                report += f"{regime_metrics['recall']:<10.3f} "
                report += f"{regime_metrics['f1']:<10.3f} "
                report += f"{regime_metrics['support']:<10}\n"
                
        return report