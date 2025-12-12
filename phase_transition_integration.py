#!/usr/bin/env python3
"""
Integration of Phase Transition Detection with RALEC-GNN
Enables regime-aware predictions and early crisis warnings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.data import Data
from collections import deque
import logging

from phase_transition_detection import (
    PhaseTransitionDetector,
    EarlyWarningSystem,
    PhaseTransitionIndicators,
    RegimeMemoryNetwork
)
from theoretical_framework import MarketPhaseSpace
from causal_discovery import CausalDiscoveryModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhaseAwareRALECGNN(nn.Module):
    """
    RALEC-GNN enhanced with phase transition detection.
    
    Key features:
    1. Early warning system for regime changes
    2. Regime-adaptive message passing
    3. Phase-aware edge construction
    4. Crisis anticipation and preparation
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_features: int,
        num_assets: int,
        hidden_dim: int = 128,
        use_phase_detection: bool = True
    ):
        super().__init__()
        
        self.base_model = base_model
        self.num_assets = num_assets
        self.use_phase_detection = use_phase_detection
        
        if use_phase_detection:
            # Phase transition detector
            self.phase_detector = PhaseTransitionDetector(
                input_dim=num_features,
                hidden_dim=hidden_dim,
                num_regimes=3
            )
            
            # Regime-adaptive components
            self.regime_adapter = RegimeAdaptiveModule(
                input_dim=num_features,
                hidden_dim=hidden_dim
            )
            
            # Crisis preparation module
            self.crisis_prep = CrisisPreparationModule(
                hidden_dim=hidden_dim
            )
            
            # Phase-aware attention
            self.phase_attention = PhaseAwareAttention(
                hidden_dim=hidden_dim,
                num_heads=4
            )
        
        # Enhanced output layer
        self.output_enhancement = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim),  # Base output + phase features
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 4)  # Regime + crisis probability
        )
        
        # Memory for tracking regime history
        self.regime_history = deque(maxlen=100)
        self.transition_history = deque(maxlen=20)
        
    def forward(
        self,
        graph_sequence: List[Data],
        return_phase_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Process graph sequence with phase transition awareness.
        """
        if self.use_phase_detection:
            # Detect phase transitions
            phase_results = self.phase_detector(graph_sequence)
            
            # Update regime history
            self._update_regime_history(phase_results)
            
            # Adapt model based on phase
            adapted_sequence = []
            for graph in graph_sequence:
                adapted_graph = self._adapt_to_phase(graph, phase_results)
                adapted_sequence.append(adapted_graph)
            
            # Check for crisis preparation
            if self._should_prepare_for_crisis(phase_results):
                adapted_sequence = self.crisis_prep.prepare_graphs(
                    adapted_sequence, phase_results
                )
            
            # Forward through base model with adaptations
            base_output = self.base_model(adapted_sequence)
            
            # Enhance output with phase information
            enhanced_output = self._enhance_output(base_output, phase_results)
            
            output = {
                **enhanced_output,
                'phase_analysis': phase_results if return_phase_analysis else None,
                'regime_trajectory': list(self.regime_history)[-10:],
                'transition_alerts': self._generate_alerts(phase_results)
            }
        else:
            # Standard forward pass
            output = self.base_model(graph_sequence)
        
        return output
    
    def _adapt_to_phase(
        self,
        graph: Data,
        phase_results: Dict[str, Any]
    ) -> Data:
        """
        Adapt graph processing based on detected phase.
        """
        # Get current regime probabilities
        regime_probs = phase_results['next_regime_probs']
        
        # Apply regime-specific adaptations
        adapted_features = self.regime_adapter(
            graph.x,
            regime_probs,
            phase_results['warning_level']
        )
        
        # Apply phase-aware attention to edges
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            edge_attr_adapted = self.phase_attention(
                graph.edge_attr,
                adapted_features,
                graph.edge_index,
                regime_probs
            )
        else:
            edge_attr_adapted = None
        
        # Create adapted graph
        adapted_graph = Data(
            x=adapted_features,
            edge_index=graph.edge_index,
            edge_attr=edge_attr_adapted
        )
        
        # Add phase information
        adapted_graph.phase_info = {
            'regime_probs': regime_probs,
            'warning_level': phase_results['warning_level'],
            'transition_prob': phase_results['transition_probability']
        }
        
        return adapted_graph
    
    def _should_prepare_for_crisis(self, phase_results: Dict[str, Any]) -> bool:
        """
        Determine if crisis preparation should be activated.
        """
        # Check multiple conditions
        conditions = [
            phase_results['transition_probability'] > 0.7,
            phase_results['warning_level'] > 0.8,
            phase_results['next_regime_probs'][2] > 0.5,  # Crisis probability
            phase_results.get('percolation_risk', 0) > 0.6
        ]
        
        # Activate if multiple conditions met
        return sum(conditions) >= 2
    
    def _enhance_output(
        self,
        base_output: Dict[str, Any],
        phase_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance predictions with phase information.
        """
        enhanced = base_output.copy()
        
        # Extract phase features
        phase_features = torch.cat([
            phase_results['transition_probability'].unsqueeze(0),
            phase_results['warning_level'].unsqueeze(0),
            phase_results['next_regime_probs'],
            torch.tensor([
                phase_results['landscape_roughness'],
                phase_results.get('percolation_risk', 0),
                phase_results['regime_stability'][0].item()
                if torch.is_tensor(phase_results['regime_stability'])
                else phase_results['regime_stability']
            ])
        ])
        
        # Combine with base predictions
        if 'regime_logits' in base_output:
            combined_features = torch.cat([
                base_output['regime_logits'].mean(dim=0),
                phase_features
            ])
        else:
            combined_features = phase_features
        
        # Enhanced predictions
        enhanced_pred = self.output_enhancement(combined_features.unsqueeze(0))
        
        enhanced['regime_logits_enhanced'] = enhanced_pred[:, :3]
        enhanced['crisis_probability_enhanced'] = torch.sigmoid(enhanced_pred[:, 3])
        
        # Add phase-specific predictions
        enhanced['transition_imminent'] = phase_results['transition_probability'] > 0.8
        enhanced['current_phase_stability'] = phase_results['regime_stability']
        enhanced['critical_indicators'] = phase_results['critical_indicators']
        
        return enhanced
    
    def _update_regime_history(self, phase_results: Dict[str, Any]):
        """Update tracking of regime evolution"""
        current_regime = torch.argmax(phase_results['next_regime_probs']).item()
        self.regime_history.append(current_regime)
        
        # Track transitions
        if len(self.regime_history) > 1:
            if self.regime_history[-1] != self.regime_history[-2]:
                self.transition_history.append({
                    'from': self.regime_history[-2],
                    'to': self.regime_history[-1],
                    'time': len(self.regime_history),
                    'probability': phase_results['transition_probability'].item()
                })
    
    def _generate_alerts(self, phase_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable alerts based on phase detection"""
        alerts = []
        
        # Critical warning
        if phase_results['warning_level'] > 0.9:
            alerts.append({
                'type': 'CRITICAL',
                'message': 'Imminent regime transition detected',
                'probability': phase_results['transition_probability'].item(),
                'recommended_action': 'Reduce risk exposure immediately'
            })
        
        # High warning
        elif phase_results['warning_level'] > 0.7:
            alerts.append({
                'type': 'HIGH',
                'message': 'Elevated transition risk',
                'indicators': phase_results['critical_indicators'],
                'recommended_action': 'Monitor closely and prepare hedges'
            })
        
        # Specific indicator alerts
        indicators = phase_results['indicators']
        if indicators.autocorrelation > 0.85:
            alerts.append({
                'type': 'INDICATOR',
                'message': 'Critical slowing down detected',
                'value': indicators.autocorrelation,
                'interpretation': 'System losing resilience'
            })
        
        if indicators.flickering > 0.5:
            alerts.append({
                'type': 'INDICATOR',
                'message': 'High regime flickering',
                'value': indicators.flickering,
                'interpretation': 'Bistable dynamics - transition imminent'
            })
        
        return alerts


class RegimeAdaptiveModule(nn.Module):
    """
    Adapts node features based on current regime and transition risk.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Regime-specific transformations
        self.regime_transforms = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(3)
        ])
        
        # Transition adaptation
        self.transition_adapter = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # Features + warning level
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, input_dim)
        
    def forward(
        self,
        features: torch.Tensor,
        regime_probs: torch.Tensor,
        warning_level: torch.Tensor
    ) -> torch.Tensor:
        """
        Adapt features based on regime and transition risk.
        """
        # Regime-weighted transformation
        regime_outputs = []
        for i, transform in enumerate(self.regime_transforms):
            regime_out = transform(features)
            weighted = regime_out * regime_probs[i]
            regime_outputs.append(weighted)
        
        regime_adapted = torch.stack(regime_outputs).sum(dim=0)
        
        # Transition adaptation
        warning_expanded = warning_level.unsqueeze(0).expand(features.shape[0], 1)
        transition_input = torch.cat([features, warning_expanded], dim=1)
        transition_adapted = self.transition_adapter(transition_input)
        
        # Combine
        combined = torch.cat([regime_adapted, transition_adapted], dim=1)
        output = self.output_proj(combined)
        
        # Residual connection
        return features + 0.5 * output


class CrisisPreparationModule(nn.Module):
    """
    Prepares the model for impending crisis by adjusting processing.
    """
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        # Crisis feature enhancer
        self.feature_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.3),  # More dropout for robustness
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Edge weight adjuster
        self.edge_adjuster = nn.Sequential(
            nn.Linear(8 + 1, 16),  # Edge features + crisis score
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Sigmoid()
        )
        
        # Robustness enhancer
        self.robustness_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def prepare_graphs(
        self,
        graph_sequence: List[Data],
        phase_results: Dict[str, Any]
    ) -> List[Data]:
        """
        Prepare graphs for crisis conditions.
        """
        crisis_score = phase_results['next_regime_probs'][2].item()  # Crisis probability
        
        prepared_graphs = []
        for graph in graph_sequence:
            # Enhance features for robustness
            enhanced_features = self.feature_enhancer(graph.x)
            robust_features = self.robustness_net(enhanced_features)
            
            # Adjust edge weights based on crisis proximity
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                crisis_tensor = torch.full((graph.edge_attr.shape[0], 1), crisis_score)
                edge_input = torch.cat([graph.edge_attr, crisis_tensor], dim=1)
                adjusted_edges = self.edge_adjuster(edge_input)
                
                # Increase edge weights to capture contagion
                edge_attr = graph.edge_attr * (1 + adjusted_edges)
            else:
                edge_attr = None
            
            # Create prepared graph
            prepared = Data(
                x=robust_features,
                edge_index=graph.edge_index,
                edge_attr=edge_attr
            )
            
            # Add crisis preparation flag
            prepared.crisis_prepared = True
            prepared.crisis_score = crisis_score
            
            prepared_graphs.append(prepared)
        
        return prepared_graphs


class PhaseAwareAttention(nn.Module):
    """
    Attention mechanism that considers market phase.
    """
    
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Phase-conditioned attention
        self.phase_query = nn.Linear(hidden_dim + 3, hidden_dim)  # +3 for regime probs
        self.phase_key = nn.Linear(hidden_dim + 3, hidden_dim)
        self.phase_value = nn.Linear(hidden_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        edge_attr: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        regime_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply phase-aware attention to edges.
        """
        if edge_attr.shape[0] == 0:
            return edge_attr
        
        # Get source and target features
        source_features = node_features[edge_index[0]]
        target_features = node_features[edge_index[1]]
        
        # Expand regime probs for each edge
        regime_expanded = regime_probs.unsqueeze(0).expand(edge_attr.shape[0], -1)
        
        # Create phase-conditioned queries and keys
        query_input = torch.cat([source_features, regime_expanded], dim=1)
        key_input = torch.cat([target_features, regime_expanded], dim=1)
        
        queries = self.phase_query(query_input).unsqueeze(1)
        keys = self.phase_key(key_input).unsqueeze(1)
        values = self.phase_value(edge_attr).unsqueeze(1)
        
        # Apply attention
        attended, _ = self.attention(queries, keys, values)
        attended = attended.squeeze(1)
        
        # Output projection
        output = self.output_proj(attended)
        
        # Residual connection
        if output.shape[1] == edge_attr.shape[1]:
            return edge_attr + 0.5 * output
        else:
            # Adjust dimensions if needed
            return edge_attr + 0.5 * output[:, :edge_attr.shape[1]]


class PhaseTransitionLoss(nn.Module):
    """
    Loss function that emphasizes phase transition detection.
    """
    
    def __init__(self, alpha: float = 0.2, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha  # Weight for transition prediction
        self.beta = beta   # Weight for early warning
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        phase_results: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute phase-aware loss.
        """
        # Standard regime prediction loss
        if 'regime_logits' in predictions and 'regime_labels' in targets:
            regime_loss = F.cross_entropy(
                predictions['regime_logits'],
                targets['regime_labels']
            )
        else:
            regime_loss = torch.tensor(0.0)
        
        # Transition prediction loss
        if 'transition_labels' in targets:
            transition_pred = phase_results['transition_probability']
            transition_loss = F.binary_cross_entropy(
                transition_pred,
                targets['transition_labels'].float()
            )
        else:
            transition_loss = torch.tensor(0.0)
        
        # Early warning loss (encourage early detection)
        if 'time_to_transition' in targets:
            # Penalize based on how close we are to transition
            time_factor = 1 / (targets['time_to_transition'] + 1)
            warning_loss = -torch.log(phase_results['warning_level'] + 1e-8) * time_factor
        else:
            warning_loss = torch.tensor(0.0)
        
        # Crisis preparation loss
        if 'crisis_occurred' in targets and targets['crisis_occurred']:
            # Penalize if not prepared for crisis
            crisis_prep_score = phase_results['next_regime_probs'][2]  # Crisis probability
            crisis_loss = -torch.log(crisis_prep_score + 1e-8)
        else:
            crisis_loss = torch.tensor(0.0)
        
        # Combine losses
        total_loss = (
            regime_loss +
            self.alpha * transition_loss +
            self.beta * warning_loss +
            0.1 * crisis_loss
        )
        
        return total_loss


def integrate_phase_transitions(
    base_model: nn.Module,
    config: Dict[str, Any]
) -> PhaseAwareRALECGNN:
    """
    Factory function to create phase-aware RALEC-GNN.
    """
    phase_model = PhaseAwareRALECGNN(
        base_model=base_model,
        num_features=config['num_features'],
        num_assets=config['num_assets'],
        hidden_dim=config.get('hidden_dim', 128),
        use_phase_detection=True
    )
    
    logger.info("Phase transition integration complete")
    logger.info("Features:")
    logger.info("  - Early warning system with 8 indicators")
    logger.info("  - Potential landscape reconstruction")
    logger.info("  - Regime memory and adaptation")
    logger.info("  - Crisis preparation mode")
    logger.info("  - Phase-aware attention")
    
    return phase_model


if __name__ == "__main__":
    # Test integration
    logger.info("Testing phase transition integration...")
    
    # Dummy base model
    class DummyModel(nn.Module):
        def forward(self, x):
            return {
                'regime_logits': torch.randn(1, 3),
                'predictions': torch.randn(1, 10)
            }
    
    # Configuration
    config = {
        'num_features': 16,
        'num_assets': 50,
        'hidden_dim': 128
    }
    
    # Create phase-aware model
    base_model = DummyModel()
    phase_model = integrate_phase_transitions(base_model, config)
    
    # Test with dummy data
    graph_sequence = []
    for t in range(20):
        graph = Data(
            x=torch.randn(config['num_assets'], config['num_features']),
            edge_index=torch.randint(0, config['num_assets'], (2, 100)),
            edge_attr=torch.randn(100, 8)
        )
        graph_sequence.append(graph)
    
    # Forward pass
    output = phase_model(graph_sequence, return_phase_analysis=True)
    
    logger.info(f"Output keys: {list(output.keys())}")
    if 'transition_alerts' in output:
        logger.info(f"Alerts: {len(output['transition_alerts'])}")
    if 'crisis_probability_enhanced' in output:
        logger.info(f"Enhanced crisis probability: {output['crisis_probability_enhanced'].item():.3f}")