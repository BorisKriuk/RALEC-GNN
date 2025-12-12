"""
Enhanced RALEC-GNN Model
Integrates all 6 phases into a unified architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from torch_geometric.data import Data

# Import all phase implementations
from .theoretical_framework import FinancialNetworkMorphology
from .causal_discovery import CausalDiscoveryModule
from .phase_transition_detection import PhaseTransitionDetector
from .meta_learning_crisis_memory import MetaLearningCrisisMemory
from .emergent_risk_metrics import EmergentRiskMetrics
from .optimized_train import OptimizedTrainingMixin


class EnhancedRALECGNN(nn.Module, OptimizedTrainingMixin):
    """
    Enhanced RALEC-GNN with all 6 phases integrated:
    1. Optimized training pipeline
    2. Financial network morphology theory
    3. Causal discovery
    4. Phase transition detection
    5. Meta-learning crisis memory
    6. Emergent risk metrics
    """
    
    def __init__(
        self,
        num_features: int,
        num_assets: int,
        hidden_dim: int = 256,
        num_regimes: int = 5,
        dropout: float = 0.2,
        # Phase toggles
        use_optimization: bool = True,
        use_theory: bool = True,
        use_causal: bool = True,
        use_phase_detection: bool = True,
        use_meta_learning: bool = True,
        use_risk_metrics: bool = True
    ):
        super().__init__()
        
        self.num_features = num_features
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes
        
        # Phase 1: Optimization is handled by mixin
        self.use_optimization = use_optimization
        
        # Phase 2: Theoretical Framework
        if use_theory:
            self.morphology = FinancialNetworkMorphology(
                num_assets=num_assets,
                hidden_dim=hidden_dim
            )
        
        # Phase 3: Causal Discovery
        if use_causal:
            self.causal_module = CausalDiscoveryModule(
                num_features=num_features,
                hidden_dim=hidden_dim,
                max_lag=5
            )
        
        # Phase 4: Phase Transition Detection
        if use_phase_detection:
            self.phase_detector = PhaseTransitionDetector(
                input_dim=num_features,
                hidden_dim=hidden_dim
            )
        
        # Phase 5: Meta-Learning
        if use_meta_learning:
            self.crisis_memory = MetaLearningCrisisMemory(
                input_dim=num_features,
                hidden_dim=hidden_dim,
                memory_size=1000,
                num_prototypes=10
            )
        
        # Phase 6: Emergent Risk Metrics
        if use_risk_metrics:
            self.risk_metrics = EmergentRiskMetrics(
                num_assets=num_assets,
                hidden_dim=hidden_dim,
                memory_enabled=use_meta_learning
            )
        
        # Core GNN layers (simplified for integration)
        self.gnn_layers = nn.ModuleList([
            nn.Linear(num_features + 20, hidden_dim),  # +20 for enhancements
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        
        # Output heads
        self.regime_head = nn.Linear(hidden_dim, num_regimes)
        self.risk_head = nn.Linear(hidden_dim, 1)
        
        # Activation and regularization
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # State
        self.defensive_mode = False
        
    def forward(
        self,
        graph_sequence: List[Data],
        return_risk_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass through enhanced RALEC-GNN.
        
        Args:
            graph_sequence: List of graph snapshots
            return_risk_analysis: Whether to return detailed risk metrics
            
        Returns:
            Dictionary containing predictions and optional risk analysis
        """
        # Extract current graph
        current_graph = graph_sequence[-1]
        x = current_graph.x
        edge_index = current_graph.edge_index
        
        # Phase 2: Apply theoretical framework
        if hasattr(self, 'morphology'):
            market_state = self.morphology.compute_market_state(current_graph)
            regime_probs = self.morphology.predict_regime(market_state)
            x = self._enhance_features_with_theory(x, market_state)
        else:
            regime_probs = None
        
        # Phase 3: Causal discovery
        if hasattr(self, 'causal_module'):
            causal_adj, causal_strength = self.causal_module(
                graph_sequence, 
                edge_index
            )
            edge_index = self._update_edges_with_causality(
                edge_index, causal_adj, causal_strength
            )
        
        # Phase 4: Phase transition detection
        if hasattr(self, 'phase_detector'):
            phase_analysis = self.phase_detector.detect_phase_transition(
                graph_sequence
            )
            x = self._enhance_features_with_phase(x, phase_analysis)
        else:
            phase_analysis = None
        
        # Phase 5: Meta-learning
        if hasattr(self, 'crisis_memory'):
            current_obs = x.mean(dim=0)  # Aggregate observation
            memory_output = self.crisis_memory.process_crisis_observation(
                current_obs,
                is_crisis=self._check_crisis_state(phase_analysis),
                episode_info=self._get_episode_info(graph_sequence)
            )
            x = self._enhance_features_with_memory(x, memory_output)
        else:
            memory_output = None
        
        # Phase 6: Risk metrics
        if hasattr(self, 'risk_metrics'):
            risk_indicators = self.risk_metrics.compute_systemic_risk(
                graph_sequence,
                crisis_memory=self.crisis_memory if hasattr(self, 'crisis_memory') else None,
                phase_indicators=phase_analysis
            )
            
            # Apply defensive measures if needed
            if risk_indicators.overall_systemic_risk > 0.7:
                self.defensive_mode = True
                x, edge_index = self._apply_defensive_measures(x, edge_index)
        else:
            risk_indicators = None
        
        # Core GNN processing
        h = x
        for i, layer in enumerate(self.gnn_layers):
            h = layer(h)
            h = self.activation(h)
            h = self.batch_norm(h)
            h = self.dropout(h)
            
            # Graph aggregation (simplified)
            if i < len(self.gnn_layers) - 1:
                h = self._graph_aggregate(h, edge_index)
        
        # Final predictions
        regime_logits = self.regime_head(h.mean(dim=0, keepdim=True))
        risk_score = torch.sigmoid(self.risk_head(h.mean(dim=0, keepdim=True)))
        
        # Prepare output
        output = {
            'predictions': regime_logits,
            'risk_score': risk_score,
            'hidden_states': h
        }
        
        # Add phase-specific outputs
        if regime_probs is not None:
            output['theory_regime_probs'] = regime_probs
            
        if memory_output is not None:
            output['memory_insights'] = memory_output
            
        if risk_indicators is not None and return_risk_analysis:
            output['risk_analysis'] = {
                'overall_risk': risk_indicators.overall_systemic_risk,
                'network_fragility': risk_indicators.network_fragility,
                'cascade_probability': risk_indicators.cascade_probability,
                'herding_index': risk_indicators.herding_index,
                'risk_decomposition': risk_indicators.risk_decomposition
            }
            
        if phase_analysis is not None:
            output['phase_analysis'] = phase_analysis
            
        return output
    
    def _enhance_features_with_theory(self, x: torch.Tensor, market_state: Any) -> torch.Tensor:
        """Enhance features with theoretical framework insights."""
        theory_features = torch.zeros(x.shape[0], 4)
        theory_features[:, 0] = market_state.volatility
        theory_features[:, 1] = market_state.correlation
        theory_features[:, 2] = market_state.liquidity
        theory_features[:, 3] = market_state.phase_distance
        
        return torch.cat([x, theory_features], dim=1)
    
    def _update_edges_with_causality(
        self, 
        edge_index: torch.Tensor,
        causal_adj: torch.Tensor,
        causal_strength: torch.Tensor
    ) -> torch.Tensor:
        """Update edge structure based on causal discovery."""
        # Keep only strong causal edges
        threshold = 0.3
        strong_edges = causal_strength > threshold
        
        if strong_edges.any():
            causal_edges = torch.nonzero(strong_edges).t()
            # Combine with original edges (union)
            combined = torch.cat([edge_index, causal_edges], dim=1)
            # Remove duplicates
            combined = torch.unique(combined, dim=1)
            return combined
        
        return edge_index
    
    def _enhance_features_with_phase(
        self,
        x: torch.Tensor,
        phase_analysis: Dict[str, Any]
    ) -> torch.Tensor:
        """Enhance features with phase transition indicators."""
        phase_features = torch.zeros(x.shape[0], 8)
        
        indicators = phase_analysis.get('indicators', {})
        phase_features[:, 0] = indicators.get('autocorrelation', 0)
        phase_features[:, 1] = indicators.get('variance', 1)
        phase_features[:, 2] = indicators.get('critical_slowing_down', 0)
        phase_features[:, 3] = phase_analysis.get('transition_probability', 0)
        phase_features[:, 4] = phase_analysis.get('warning_level', 0)
        
        return torch.cat([x, phase_features], dim=1)
    
    def _enhance_features_with_memory(
        self,
        x: torch.Tensor,
        memory_output: Dict[str, Any]
    ) -> torch.Tensor:
        """Enhance features with crisis memory insights."""
        memory_features = torch.zeros(x.shape[0], 4)
        
        memory_features[:, 0] = memory_output['crisis_probability']
        memory_features[:, 1] = memory_output['prototype_confidence']
        memory_features[:, 2] = float(memory_output['prototype_match'])
        memory_features[:, 3] = len(memory_output['similar_episodes']) / 10
        
        return torch.cat([x, memory_features], dim=1)
    
    def _apply_defensive_measures(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> tuple:
        """Apply defensive measures in high-risk situations."""
        # Feature dampening
        x = x * 0.8
        
        # Edge pruning (keep only top 70% strongest connections)
        if edge_index.shape[1] > 100:
            # Simplified: randomly keep 70% of edges
            keep_prob = 0.7
            mask = torch.rand(edge_index.shape[1]) < keep_prob
            edge_index = edge_index[:, mask]
            
        return x, edge_index
    
    def _graph_aggregate(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Simple graph aggregation (mean of neighbors)."""
        # This is simplified - in practice would use proper GNN message passing
        return h
    
    def _check_crisis_state(self, phase_analysis: Optional[Dict]) -> bool:
        """Check if system is in crisis state."""
        if phase_analysis:
            return phase_analysis.get('warning_level', 0) > 0.7
        return False
    
    def _get_episode_info(self, graph_sequence: List[Data]) -> Dict[str, Any]:
        """Extract episode information for crisis memory."""
        # Simplified episode info extraction
        return {
            'trigger': 'market_volatility',
            'severity': 0.5,
            'affected_assets': list(range(min(10, self.num_assets))),
            'market_state': {
                'volatility': 0.3,
                'correlation': 0.5,
                'liquidity': 0.7
            }
        }
    
    def activate_defensive_mode(self):
        """Manually activate defensive mode."""
        self.defensive_mode = True
        print("⚠️  Defensive mode ACTIVATED")
        
    def deactivate_defensive_mode(self):
        """Manually deactivate defensive mode."""
        self.defensive_mode = False
        print("✓ Defensive mode deactivated")