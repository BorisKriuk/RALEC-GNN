#!/usr/bin/env python3
"""
Integration of Emergent Risk Metrics with RALEC-GNN
Enhances the model with systemic risk awareness and proactive risk management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.data import Data
import logging

from emergent_risk_metrics import (
    EmergentRiskMetrics, 
    SystemicRiskIndicators,
    create_risk_alert_system
)
from meta_learning_crisis_memory import MetaLearningCrisisMemory
from phase_transition_detection import PhaseTransitionDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAwareRALECGNN(nn.Module):
    """
    RALEC-GNN enhanced with emergent risk metrics for comprehensive
    systemic risk monitoring and management.
    
    Key features:
    1. Real-time systemic risk assessment
    2. Risk-aware predictions
    3. Proactive alert generation
    4. Defensive mode activation
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_features: int,
        num_assets: int,
        hidden_dim: int = 256,
        use_risk_metrics: bool = True,
        risk_threshold: float = 0.7,
        memory_enabled: bool = True
    ):
        super().__init__()
        
        self.base_model = base_model
        self.num_assets = num_assets
        self.use_risk_metrics = use_risk_metrics
        self.risk_threshold = risk_threshold
        
        if use_risk_metrics:
            # Emergent risk analyzer
            self.risk_analyzer = EmergentRiskMetrics(
                num_assets=num_assets,
                hidden_dim=hidden_dim,
                memory_enabled=memory_enabled
            )
            
            # Risk-aware prediction head
            self.risk_aware_head = RiskAwarePredictionHead(
                input_dim=hidden_dim,
                risk_dim=15  # Number of risk indicators
            )
            
            # Defensive mode controller
            self.defensive_controller = DefensiveModeController(
                hidden_dim=hidden_dim
            )
            
            # Risk-conditioned attention
            self.risk_attention = RiskConditionedAttention(
                hidden_dim=hidden_dim,
                num_heads=8
            )
        
        # Risk history tracking
        self.risk_history = []
        self.alert_history = []
        self.defensive_mode = False
        
    def forward(
        self,
        graph_sequence: List[Data],
        returns_sequence: Optional[torch.Tensor] = None,
        crisis_memory: Optional[Any] = None,
        phase_indicators: Optional[Dict[str, float]] = None,
        return_risk_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Process graph sequence with emergent risk awareness.
        """
        # Compute systemic risk if enabled
        if self.use_risk_metrics:
            risk_indicators = self.risk_analyzer.compute_systemic_risk(
                graph_sequence,
                returns_sequence=returns_sequence,
                crisis_memory=crisis_memory,
                phase_indicators=phase_indicators
            )
            
            # Store risk history
            self.risk_history.append(risk_indicators)
            
            # Generate alerts
            alerts = create_risk_alert_system(risk_indicators)
            self.alert_history.append(alerts)
            
            # Check defensive mode activation
            if risk_indicators.overall_systemic_risk > self.risk_threshold:
                self.defensive_mode = True
                defensive_params = self.defensive_controller.get_defensive_parameters(
                    risk_indicators
                )
            else:
                self.defensive_mode = False
                defensive_params = None
            
            # Enhance graphs with risk information
            enhanced_sequence = []
            for graph in graph_sequence:
                enhanced_graph = self._enhance_with_risk(
                    graph, risk_indicators, defensive_params
                )
                enhanced_sequence.append(enhanced_graph)
        else:
            enhanced_sequence = graph_sequence
            risk_indicators = None
            alerts = []
            defensive_params = None
        
        # Forward through base model
        base_output = self.base_model(enhanced_sequence)
        
        # Risk-aware predictions if enabled
        if self.use_risk_metrics and risk_indicators is not None:
            # Extract risk vector
            risk_vector = self._extract_risk_vector(risk_indicators)
            
            # Risk-aware prediction
            risk_predictions = self.risk_aware_head(
                base_output.get('hidden_states', torch.zeros(1, self.risk_attention.hidden_dim)),
                risk_vector
            )
            
            # Apply risk-conditioned attention
            if 'attention_weights' in base_output:
                risk_attention = self.risk_attention(
                    base_output['hidden_states'],
                    risk_vector
                )
                base_output['attention_weights'] = risk_attention
            
            # Combine outputs
            output = {
                **base_output,
                **risk_predictions,
                'defensive_mode': self.defensive_mode,
                'defensive_params': defensive_params,
                'alerts': alerts,
                'risk_level': risk_indicators.overall_systemic_risk
            }
            
            if return_risk_analysis:
                output['risk_analysis'] = {
                    'indicators': risk_indicators,
                    'decomposition': risk_indicators.risk_decomposition,
                    'critical_nodes': torch.where(
                        risk_indicators.systemic_importance > 0.8
                    )[0].tolist() if risk_indicators.systemic_importance is not None else []
                }
        else:
            output = base_output
        
        return output
    
    def _enhance_with_risk(
        self,
        graph: Data,
        risk_indicators: SystemicRiskIndicators,
        defensive_params: Optional[Dict[str, Any]]
    ) -> Data:
        """Enhance graph with risk information."""
        # Add risk features to nodes
        num_nodes = graph.num_nodes
        risk_features = torch.zeros(num_nodes, 6)
        
        # Overall risk broadcast
        risk_features[:, 0] = risk_indicators.overall_systemic_risk
        
        # Network fragility
        risk_features[:, 1] = risk_indicators.network_fragility
        
        # Cascade probability
        risk_features[:, 2] = risk_indicators.cascade_probability
        
        # Node-specific systemic importance
        if risk_indicators.systemic_importance is not None and \
           len(risk_indicators.systemic_importance) == num_nodes:
            risk_features[:, 3] = risk_indicators.systemic_importance
        
        # Herding indicator
        risk_features[:, 4] = risk_indicators.herding_index
        
        # Defensive mode indicator
        risk_features[:, 5] = float(self.defensive_mode)
        
        # Concatenate with original features
        if hasattr(graph, 'x') and graph.x is not None:
            enhanced_x = torch.cat([graph.x, risk_features], dim=1)
        else:
            enhanced_x = risk_features
        
        # Apply defensive transformations if needed
        if defensive_params:
            enhanced_x, edge_index, edge_attr = self._apply_defensive_measures(
                enhanced_x, graph.edge_index, graph.edge_attr, defensive_params
            )
        else:
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr
        
        # Create enhanced graph
        enhanced_graph = Data(
            x=enhanced_x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        # Add risk metadata
        enhanced_graph.risk_metadata = {
            'overall_risk': risk_indicators.overall_systemic_risk,
            'defensive_mode': self.defensive_mode,
            'num_alerts': len(self.alert_history[-1]) if self.alert_history else 0
        }
        
        return enhanced_graph
    
    def _extract_risk_vector(self, risk_indicators: SystemicRiskIndicators) -> torch.Tensor:
        """Extract risk indicators as vector."""
        risk_vector = torch.tensor([
            risk_indicators.network_fragility,
            risk_indicators.contagion_potential,
            risk_indicators.clustering_risk,
            risk_indicators.herding_index,
            risk_indicators.synchronization_risk,
            risk_indicators.diversity_loss,
            risk_indicators.information_contagion,
            risk_indicators.uncertainty_propagation,
            risk_indicators.emergence_indicator,
            risk_indicators.self_organization,
            risk_indicators.cascade_probability,
            risk_indicators.memory_fragility,
            risk_indicators.adaptation_capacity,
            risk_indicators.overall_systemic_risk,
            float(self.defensive_mode)
        ])
        
        return risk_vector
    
    def _apply_defensive_measures(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        defensive_params: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply defensive transformations when in high-risk mode."""
        # 1. Feature dampening to reduce volatility
        dampening_factor = defensive_params.get('feature_dampening', 0.8)
        node_features = node_features * dampening_factor
        
        # 2. Edge pruning to reduce contagion
        if defensive_params.get('edge_pruning', False):
            edge_threshold = defensive_params.get('edge_threshold', 0.5)
            
            if edge_attr is not None and edge_attr.shape[0] > 0:
                # Keep only strong edges
                edge_mask = edge_attr[:, 0] > edge_threshold
                edge_index = edge_index[:, edge_mask]
                edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
        
        # 3. Add noise for robustness
        if defensive_params.get('add_noise', False):
            noise_level = defensive_params.get('noise_level', 0.01)
            node_features = node_features + torch.randn_like(node_features) * noise_level
        
        return node_features, edge_index, edge_attr
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of current risk status."""
        if not self.risk_history:
            return {
                'status': 'No risk data available',
                'risk_level': 0.0,
                'defensive_mode': False
            }
        
        current_risk = self.risk_history[-1]
        recent_alerts = self.alert_history[-1] if self.alert_history else []
        
        summary = {
            'status': self._get_risk_status(current_risk.overall_systemic_risk),
            'risk_level': current_risk.overall_systemic_risk,
            'defensive_mode': self.defensive_mode,
            'key_risks': {
                'network_fragility': current_risk.network_fragility,
                'cascade_probability': current_risk.cascade_probability,
                'herding_index': current_risk.herding_index,
                'emergence_indicator': current_risk.emergence_indicator
            },
            'active_alerts': len(recent_alerts),
            'critical_alerts': sum(1 for a in recent_alerts if a['level'] == 'CRITICAL'),
            'risk_trend': self._compute_risk_trend()
        }
        
        return summary
    
    def _get_risk_status(self, risk_level: float) -> str:
        """Convert risk level to status string."""
        if risk_level < 0.3:
            return "LOW - System operating normally"
        elif risk_level < 0.7:
            return "MEDIUM - Elevated risk, monitoring required"
        elif risk_level < 0.9:
            return "HIGH - Significant risk, defensive measures active"
        else:
            return "CRITICAL - Extreme risk, maximum defensive posture"
    
    def _compute_risk_trend(self) -> str:
        """Compute recent risk trend."""
        if len(self.risk_history) < 2:
            return "STABLE"
        
        recent_risks = [r.overall_systemic_risk for r in self.risk_history[-5:]]
        
        if len(recent_risks) >= 2:
            trend = recent_risks[-1] - recent_risks[0]
            
            if trend > 0.1:
                return "INCREASING"
            elif trend < -0.1:
                return "DECREASING"
        
        return "STABLE"


class RiskAwarePredictionHead(nn.Module):
    """
    Prediction head that incorporates risk indicators.
    """
    
    def __init__(self, input_dim: int, risk_dim: int = 15):
        super().__init__()
        
        # Risk encoder
        self.risk_encoder = nn.Sequential(
            nn.Linear(risk_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32)
        )
        
        # Combined predictor
        self.predictor = nn.Sequential(
            nn.Linear(input_dim + 32, input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 3)  # Regime predictions
        )
        
        # Risk-adjusted confidence
        self.confidence_adjuster = nn.Sequential(
            nn.Linear(risk_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        risk_vector: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generate risk-aware predictions."""
        # Encode risk
        risk_encoding = self.risk_encoder(risk_vector.unsqueeze(0))
        
        # Combine with hidden states
        combined = torch.cat([hidden_states, risk_encoding], dim=-1)
        
        # Make predictions
        regime_logits = self.predictor(combined)
        
        # Adjust confidence based on risk
        base_confidence = F.softmax(regime_logits, dim=-1).max()
        risk_adjustment = self.confidence_adjuster(risk_vector.unsqueeze(0))
        
        # Lower confidence in high-risk situations
        adjusted_confidence = base_confidence * risk_adjustment
        
        return {
            'regime_logits_risk_aware': regime_logits,
            'prediction_confidence': adjusted_confidence,
            'risk_adjustment_factor': risk_adjustment
        }


class DefensiveModeController(nn.Module):
    """
    Controls defensive measures based on risk levels.
    """
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        # Parameter generator
        self.param_generator = nn.Sequential(
            nn.Linear(15, hidden_dim),  # Risk indicators
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6)  # Defensive parameters
        )
        
    def get_defensive_parameters(
        self,
        risk_indicators: SystemicRiskIndicators
    ) -> Dict[str, Any]:
        """Generate defensive parameters based on risk."""
        # Extract risk features
        risk_vector = torch.tensor([
            risk_indicators.network_fragility,
            risk_indicators.cascade_probability,
            risk_indicators.herding_index,
            risk_indicators.synchronization_risk,
            risk_indicators.information_contagion,
            risk_indicators.emergence_indicator,
            risk_indicators.overall_systemic_risk,
            # Pad to 15 dimensions
            0, 0, 0, 0, 0, 0, 0, 0
        ]).unsqueeze(0)
        
        # Generate parameters
        params = self.param_generator(risk_vector).squeeze()
        
        # Parse parameters
        defensive_params = {
            'feature_dampening': torch.sigmoid(params[0]).item() * 0.5 + 0.5,  # 0.5-1.0
            'edge_pruning': torch.sigmoid(params[1]).item() > 0.5,
            'edge_threshold': torch.sigmoid(params[2]).item() * 0.8 + 0.2,  # 0.2-1.0
            'add_noise': torch.sigmoid(params[3]).item() > 0.5,
            'noise_level': torch.sigmoid(params[4]).item() * 0.05,  # 0-0.05
            'attention_scaling': torch.sigmoid(params[5]).item() * 0.5 + 0.5  # 0.5-1.0
        }
        
        # Risk-based adjustments
        if risk_indicators.cascade_probability > 0.8:
            defensive_params['edge_pruning'] = True
            defensive_params['edge_threshold'] = max(0.7, defensive_params['edge_threshold'])
        
        if risk_indicators.herding_index > 0.8:
            defensive_params['add_noise'] = True
            defensive_params['noise_level'] = max(0.02, defensive_params['noise_level'])
        
        return defensive_params


class RiskConditionedAttention(nn.Module):
    """
    Attention mechanism conditioned on risk levels.
    """
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Risk-to-attention mapping
        self.risk_to_attention = nn.Sequential(
            nn.Linear(15, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
            nn.Softmax(dim=-1)
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        risk_vector: torch.Tensor
    ) -> torch.Tensor:
        """Apply risk-conditioned attention."""
        # Get risk-based attention weights
        risk_attention_weights = self.risk_to_attention(risk_vector.unsqueeze(0))
        
        # Apply attention
        attended, _ = self.attention(
            hidden_states.unsqueeze(0),
            hidden_states.unsqueeze(0),
            hidden_states.unsqueeze(0)
        )
        
        # Weight by risk
        # In high-risk situations, focus on critical components
        if risk_vector[0] > 0.7:  # High overall risk
            # Apply risk-weighted attention scaling
            for i in range(self.num_heads):
                attended[:, :, i * self.hidden_dim // self.num_heads:(i + 1) * self.hidden_dim // self.num_heads] *= \
                    risk_attention_weights[0, i]
        
        return attended.squeeze(0)


def integrate_emergent_risk(
    base_model: nn.Module,
    config: Dict[str, Any]
) -> RiskAwareRALECGNN:
    """
    Factory function to create risk-aware RALEC-GNN.
    """
    risk_model = RiskAwareRALECGNN(
        base_model=base_model,
        num_features=config['num_features'],
        num_assets=config['num_assets'],
        hidden_dim=config.get('hidden_dim', 256),
        use_risk_metrics=True,
        risk_threshold=config.get('risk_threshold', 0.7),
        memory_enabled=config.get('memory_enabled', True)
    )
    
    logger.info("Emergent risk integration complete")
    logger.info("Features:")
    logger.info("  - Real-time systemic risk monitoring")
    logger.info("  - Proactive alert generation")
    logger.info("  - Defensive mode activation")
    logger.info("  - Risk-aware predictions")
    logger.info("  - Network fragility analysis")
    logger.info("  - Cascade risk assessment")
    logger.info("  - Collective behavior monitoring")
    
    return risk_model


if __name__ == "__main__":
    # Test integration
    logger.info("Testing emergent risk integration...")
    
    # Dummy base model
    class DummyModel(nn.Module):
        def forward(self, x):
            return {
                'regime_logits': torch.randn(1, 3),
                'hidden_states': torch.randn(1, 256)
            }
    
    # Configuration
    config = {
        'num_features': 16,
        'num_assets': 50,
        'hidden_dim': 256,
        'risk_threshold': 0.7
    }
    
    # Create risk-aware model
    base_model = DummyModel()
    risk_model = integrate_emergent_risk(base_model, config)
    
    # Test with dummy data
    graph_sequence = []
    returns_sequence = []
    
    for t in range(30):
        # Simulate increasing risk
        if t < 20:
            x = torch.randn(config['num_assets'], config['num_features']) * 0.1
            returns = torch.randn(config['num_assets']) * 0.02
        else:
            # Crisis period
            x = torch.randn(config['num_assets'], config['num_features']) * 0.5
            returns = torch.randn(config['num_assets']) * 0.05
            # Add correlation
            common_factor = torch.randn(1, config['num_features']) * 0.3
            x += common_factor
        
        # Create graph with increasing connectivity
        num_edges = 100 + t * 10
        edge_index = torch.randint(0, config['num_assets'], (2, num_edges))
        
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=torch.randn(num_edges, 8)
        )
        
        graph_sequence.append(graph)
        returns_sequence.append(returns)
    
    returns_sequence = torch.stack(returns_sequence)
    
    # Forward pass with risk analysis
    output = risk_model(
        graph_sequence,
        returns_sequence=returns_sequence,
        return_risk_analysis=True
    )
    
    # Display results
    logger.info("\nRisk Analysis Results:")
    logger.info("-" * 50)
    
    # Get risk summary
    summary = risk_model.get_risk_summary()
    
    logger.info(f"Status: {summary['status']}")
    logger.info(f"Risk Level: {summary['risk_level']:.3f}")
    logger.info(f"Defensive Mode: {summary['defensive_mode']}")
    logger.info(f"Active Alerts: {summary['active_alerts']}")
    logger.info(f"Critical Alerts: {summary['critical_alerts']}")
    logger.info(f"Risk Trend: {summary['risk_trend']}")
    
    if 'risk_analysis' in output:
        logger.info("\nKey Risk Indicators:")
        for key, value in summary['key_risks'].items():
            logger.info(f"  {key}: {value:.3f}")
        
        logger.info("\nRisk Decomposition:")
        risk_decomp = output['risk_analysis']['indicators'].risk_decomposition
        for component, weight in risk_decomp.items():
            logger.info(f"  {component}: {weight:.3f}")
    
    if output.get('alerts'):
        logger.info(f"\nAlerts ({len(output['alerts'])}):")
        for alert in output['alerts'][:3]:  # Show first 3
            logger.info(f"  [{alert['level']}] {alert['type']}: {alert['message']}")
    
    logger.info("\nEmergent risk integration test complete!")