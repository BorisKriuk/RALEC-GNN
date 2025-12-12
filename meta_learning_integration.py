#!/usr/bin/env python3
"""
Integration of Meta-Learning Crisis Memory with RALEC-GNN
Enables the model to learn from past crises and rapidly adapt to new ones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.data import Data
from collections import deque
import logging
from datetime import datetime

from meta_learning_crisis_memory import (
    MetaLearningCrisisMemory,
    CrisisEpisode,
    CrisisAnticipationModule
)
from phase_transition_detection import PhaseTransitionDetector
from causal_discovery import CausalDiscoveryModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaLearningRALECGNN(nn.Module):
    """
    RALEC-GNN enhanced with meta-learning crisis memory.
    
    Key capabilities:
    1. Remembers past crisis patterns
    2. Rapidly adapts to new crisis types
    3. Anticipates future crises based on learned patterns
    4. Provides crisis-specific predictions
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_features: int,
        num_assets: int,
        hidden_dim: int = 256,
        memory_size: int = 1000,
        use_meta_learning: bool = True
    ):
        super().__init__()
        
        self.base_model = base_model
        self.num_assets = num_assets
        self.use_meta_learning = use_meta_learning
        
        if use_meta_learning:
            # Meta-learning components
            self.crisis_memory = MetaLearningCrisisMemory(
                input_dim=num_features,
                hidden_dim=hidden_dim,
                memory_size=memory_size,
                num_prototypes=10
            )
            
            self.crisis_anticipation = CrisisAnticipationModule(
                self.crisis_memory
            )
            
            # Crisis-aware prediction head
            self.crisis_aware_head = CrisisAwarePredictionHead(
                input_dim=hidden_dim,
                num_prototypes=10
            )
            
            # Memory-guided edge constructor
            self.memory_edge_constructor = MemoryGuidedEdgeConstructor(
                node_dim=num_features,
                memory_dim=hidden_dim,
                edge_dim=8
            )
        
        # Crisis detection state
        self.crisis_state = CrisisState()
        self.historical_context = deque(maxlen=100)
        
    def forward(
        self,
        graph_sequence: List[Data],
        return_meta_insights: bool = False
    ) -> Dict[str, Any]:
        """
        Process graph sequence with meta-learning enhancements.
        """
        if self.use_meta_learning:
            # Extract current observation
            current_features = self._extract_observation_features(graph_sequence[-1])
            
            # Check if in crisis and prepare episode info
            is_crisis, episode_info = self._detect_crisis_state(graph_sequence)
            
            # Process through crisis memory
            memory_output = self.crisis_memory.process_crisis_observation(
                current_features,
                is_crisis=is_crisis,
                episode_info=episode_info
            )
            
            # Anticipate future crisis
            anticipation = self.crisis_anticipation.anticipate_crisis(
                current_features,
                list(self.historical_context)
            )
            
            # Enhance graphs with memory insights
            enhanced_sequence = []
            for graph in graph_sequence:
                enhanced_graph = self._enhance_with_memory(
                    graph, memory_output, anticipation
                )
                enhanced_sequence.append(enhanced_graph)
            
            # Check for rapid adaptation need
            if self._should_adapt(memory_output, anticipation):
                adaptation_result = self._perform_rapid_adaptation(graph_sequence)
            else:
                adaptation_result = None
            
            # Forward through base model
            base_output = self.base_model(enhanced_sequence)
            
            # Crisis-aware predictions
            crisis_predictions = self.crisis_aware_head(
                base_output.get('hidden_states', torch.zeros(1, self.crisis_memory.crisis_encoder[0].out_features)),
                memory_output,
                anticipation
            )
            
            # Update historical context
            self.historical_context.append(current_features)
            
            # Combine outputs
            output = {
                **base_output,
                **crisis_predictions,
                'memory_insights': memory_output if return_meta_insights else None,
                'crisis_anticipation': anticipation,
                'adaptation_result': adaptation_result,
                'crisis_state': self.crisis_state.get_state()
            }
        else:
            # Standard forward pass
            output = self.base_model(graph_sequence)
        
        return output
    
    def _extract_observation_features(self, graph: Data) -> torch.Tensor:
        """Extract features for crisis memory from graph"""
        # Aggregate node features
        node_features = graph.x.mean(dim=0)
        
        # Add graph-level statistics
        graph_stats = torch.tensor([
            graph.x.std().item(),
            graph.num_edges / (graph.num_nodes * (graph.num_nodes - 1)),  # Density
            graph.x.max().item() - graph.x.min().item()  # Range
        ])
        
        # Combine
        observation = torch.cat([node_features, graph_stats])
        
        # Pad or truncate to expected dimension
        target_dim = self.crisis_memory.crisis_encoder[0].in_features
        if observation.shape[0] < target_dim:
            observation = F.pad(observation, (0, target_dim - observation.shape[0]))
        else:
            observation = observation[:target_dim]
        
        return observation
    
    def _detect_crisis_state(
        self,
        graph_sequence: List[Data]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Detect if currently in crisis and gather episode information"""
        # Use last few graphs to detect crisis
        recent_graphs = graph_sequence[-5:]
        
        # Simple crisis detection (can be replaced with more sophisticated method)
        volatilities = []
        correlations = []
        
        for graph in recent_graphs:
            vol = graph.x.std().item()
            volatilities.append(vol)
            
            if graph.num_nodes > 1:
                corr = torch.corrcoef(graph.x.T).mean().item()
                correlations.append(corr)
        
        avg_vol = np.mean(volatilities)
        avg_corr = np.mean(correlations) if correlations else 0
        
        # Crisis thresholds
        is_crisis = avg_vol > 0.3 or avg_corr > 0.6
        
        # Update crisis state
        self.crisis_state.update(is_crisis, avg_vol, avg_corr)
        
        if is_crisis:
            # Gather episode information
            episode_info = {
                'trigger': self._infer_trigger(avg_vol, avg_corr),
                'affected_assets': self._identify_affected_assets(graph_sequence[-1]),
                'severity': min(1.0, (avg_vol + avg_corr) / 2),
                'early_warnings': {
                    'volatility': avg_vol,
                    'correlation': avg_corr,
                    'density': graph_sequence[-1].num_edges / (graph_sequence[-1].num_nodes ** 2)
                },
                'market_state': {
                    'volatility': avg_vol,
                    'correlation': avg_corr,
                    'liquidity': 1 / (1 + avg_vol)  # Simplified
                },
                'pattern': torch.stack([self._extract_observation_features(g) 
                                      for g in recent_graphs])
            }
            
            # Add causal structure if available
            if hasattr(graph_sequence[-1], 'causal_adjacency'):
                episode_info['causal_structure'] = graph_sequence[-1].causal_adjacency
            
            return True, episode_info
        
        return False, None
    
    def _infer_trigger(self, volatility: float, correlation: float) -> str:
        """Infer crisis trigger type from indicators"""
        if volatility > 0.5:
            return 'volatility_spike'
        elif correlation > 0.7:
            return 'contagion'
        elif volatility > 0.3 and correlation > 0.5:
            return 'systemic'
        else:
            return 'liquidity_shock'
    
    def _identify_affected_assets(self, graph: Data) -> List[int]:
        """Identify most affected assets in crisis"""
        # Find assets with highest volatility/stress
        node_volatilities = graph.x.std(dim=1)
        
        # Get top 20% most volatile
        k = max(1, int(0.2 * graph.num_nodes))
        _, top_indices = torch.topk(node_volatilities, k)
        
        return top_indices.tolist()
    
    def _enhance_with_memory(
        self,
        graph: Data,
        memory_output: Dict[str, Any],
        anticipation: Dict[str, Any]
    ) -> Data:
        """Enhance graph with memory insights"""
        # Add memory-based features to nodes
        memory_features = torch.zeros(graph.num_nodes, 4)
        
        # Crisis probability from memory
        memory_features[:, 0] = memory_output['crisis_probability']
        
        # Prototype confidence
        memory_features[:, 1] = memory_output['prototype_confidence']
        
        # Time to crisis from anticipation
        memory_features[:, 2] = anticipation['time_to_crisis'] / 30  # Normalize
        
        # Preparation urgency
        memory_features[:, 3] = anticipation['preparation_urgency']
        
        # Concatenate with original features
        enhanced_x = torch.cat([graph.x, memory_features], dim=1)
        
        # Memory-guided edge construction
        if hasattr(self, 'memory_edge_constructor'):
            memory_edges, memory_edge_attr = self.memory_edge_constructor(
                graph, memory_output, anticipation
            )
            
            # Combine with existing edges
            if graph.edge_index.shape[1] > 0:
                combined_edges = torch.cat([graph.edge_index, memory_edges], dim=1)
                
                if graph.edge_attr is not None and memory_edge_attr is not None:
                    combined_edge_attr = torch.cat([graph.edge_attr, memory_edge_attr], dim=0)
                else:
                    combined_edge_attr = None
            else:
                combined_edges = memory_edges
                combined_edge_attr = memory_edge_attr
        else:
            combined_edges = graph.edge_index
            combined_edge_attr = graph.edge_attr
        
        # Create enhanced graph
        enhanced_graph = Data(
            x=enhanced_x,
            edge_index=combined_edges,
            edge_attr=combined_edge_attr
        )
        
        # Add memory metadata
        enhanced_graph.memory_metadata = {
            'prototype_match': memory_output['prototype_match'].item() 
                if torch.is_tensor(memory_output['prototype_match']) 
                else memory_output['prototype_match'],
            'similar_episodes': len(memory_output['similar_episodes']),
            'anticipation_confidence': anticipation['confidence']
        }
        
        return enhanced_graph
    
    def _should_adapt(
        self,
        memory_output: Dict[str, Any],
        anticipation: Dict[str, Any]
    ) -> bool:
        """Determine if rapid adaptation is needed"""
        conditions = [
            memory_output['prototype_confidence'] < 0.5,  # Low confidence
            anticipation['time_to_crisis'] < 5,  # Imminent crisis
            anticipation['preparation_urgency'] > 0.8,  # High urgency
            len(memory_output['similar_episodes']) < 2  # Few similar examples
        ]
        
        return sum(conditions) >= 2
    
    def _perform_rapid_adaptation(
        self,
        graph_sequence: List[Data]
    ) -> Dict[str, Any]:
        """Perform rapid adaptation using meta-learning"""
        # Extract recent observations for adaptation
        recent_observations = [
            self._extract_observation_features(g) for g in graph_sequence[-10:]
        ]
        
        # Perform adaptation
        adaptation_result = self.crisis_memory.adapt_to_new_crisis(
            recent_observations
        )
        
        logger.info(f"Rapid adaptation performed: {adaptation_result['adapted']}")
        
        return adaptation_result
    
    def consolidate_memory(self):
        """Consolidate crisis memory after training or periodically"""
        if self.use_meta_learning:
            self.crisis_memory.consolidate_learning()
            logger.info("Crisis memory consolidated")


class CrisisState:
    """Track current crisis state and history"""
    
    def __init__(self):
        self.in_crisis = False
        self.crisis_start = None
        self.crisis_duration = 0
        self.max_severity = 0
        self.crisis_history = []
        
    def update(self, is_crisis: bool, volatility: float, correlation: float):
        """Update crisis state"""
        severity = (volatility + correlation) / 2
        
        if is_crisis and not self.in_crisis:
            # Crisis started
            self.in_crisis = True
            self.crisis_start = datetime.now()
            self.crisis_duration = 0
            self.max_severity = severity
            
        elif is_crisis and self.in_crisis:
            # Crisis continuing
            self.crisis_duration += 1
            self.max_severity = max(self.max_severity, severity)
            
        elif not is_crisis and self.in_crisis:
            # Crisis ended
            self.in_crisis = False
            crisis_record = {
                'start': self.crisis_start,
                'duration': self.crisis_duration,
                'max_severity': self.max_severity
            }
            self.crisis_history.append(crisis_record)
            
    def get_state(self) -> Dict[str, Any]:
        """Get current state summary"""
        return {
            'in_crisis': self.in_crisis,
            'crisis_duration': self.crisis_duration,
            'max_severity': self.max_severity,
            'total_crises': len(self.crisis_history),
            'avg_crisis_duration': np.mean([c['duration'] for c in self.crisis_history])
                if self.crisis_history else 0
        }


class CrisisAwarePredictionHead(nn.Module):
    """
    Prediction head that uses crisis memory insights.
    """
    
    def __init__(self, input_dim: int, num_prototypes: int):
        super().__init__()
        
        # Prototype-specific predictors
        self.prototype_heads = nn.ModuleList([
            nn.Linear(input_dim, 3) for _ in range(num_prototypes)
        ])
        
        # General predictor
        self.general_head = nn.Linear(input_dim, 3)
        
        # Severity predictor
        self.severity_head = nn.Linear(input_dim + 8, 1)  # +8 for additional features
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_output: Dict[str, Any],
        anticipation: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Generate crisis-aware predictions"""
        # Get prototype match
        prototype_idx = memory_output['prototype_match']
        if torch.is_tensor(prototype_idx):
            prototype_idx = prototype_idx.item()
        
        # Prototype-specific prediction
        if prototype_idx < len(self.prototype_heads):
            prototype_pred = self.prototype_heads[prototype_idx](hidden_states)
        else:
            prototype_pred = self.general_head(hidden_states)
        
        # General prediction
        general_pred = self.general_head(hidden_states)
        
        # Weighted combination based on prototype confidence
        confidence = memory_output['prototype_confidence']
        if torch.is_tensor(confidence):
            confidence = confidence.unsqueeze(-1)
        
        regime_logits = confidence * prototype_pred + (1 - confidence) * general_pred
        
        # Severity prediction with additional features
        severity_features = torch.cat([
            hidden_states,
            torch.tensor([
                memory_output['crisis_probability'].item() 
                    if torch.is_tensor(memory_output['crisis_probability'])
                    else memory_output['crisis_probability'],
                anticipation['time_to_crisis'] / 30,
                anticipation['preparation_urgency'],
                anticipation['confidence'],
                float(len(memory_output['similar_episodes'])) / 10,
                float(prototype_idx) / 10,
                confidence.item() if torch.is_tensor(confidence) else confidence,
                0.0  # Placeholder
            ], device=hidden_states.device).unsqueeze(0)
        ], dim=1)
        
        severity = torch.sigmoid(self.severity_head(severity_features))
        
        return {
            'regime_logits_memory': regime_logits,
            'crisis_severity': severity,
            'prototype_used': prototype_idx,
            'memory_confidence': confidence
        }


class MemoryGuidedEdgeConstructor(nn.Module):
    """
    Construct edges based on crisis memory patterns.
    """
    
    def __init__(self, node_dim: int, memory_dim: int, edge_dim: int):
        super().__init__()
        
        # Pattern-based edge scorer
        self.edge_scorer = nn.Sequential(
            nn.Linear(node_dim * 2 + 4, memory_dim),  # Node pairs + memory features
            nn.ReLU(),
            nn.BatchNorm1d(memory_dim),
            nn.Linear(memory_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        graph: Data,
        memory_output: Dict[str, Any],
        anticipation: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create memory-guided edges"""
        num_nodes = graph.num_nodes
        
        # Get similar episodes for edge patterns
        similar_episodes = memory_output.get('similar_episodes', [])
        
        if not similar_episodes:
            # Return empty edges
            return torch.zeros(2, 0, dtype=torch.long), None
        
        # Extract edge patterns from similar episodes
        # Simplified: create edges between stressed nodes
        node_stress = graph.x.std(dim=1)
        stressed_nodes = torch.where(node_stress > node_stress.mean())[0]
        
        # Create edges between stressed nodes
        edge_list = []
        edge_features = []
        
        for i in range(len(stressed_nodes)):
            for j in range(i+1, len(stressed_nodes)):
                node_i = stressed_nodes[i]
                node_j = stressed_nodes[j]
                
                # Compute edge score
                pair_features = torch.cat([
                    graph.x[node_i],
                    graph.x[node_j],
                    torch.tensor([
                        memory_output['crisis_probability'].item()
                            if torch.is_tensor(memory_output['crisis_probability'])
                            else memory_output['crisis_probability'],
                        anticipation['time_to_crisis'] / 30,
                        anticipation['preparation_urgency'],
                        anticipation['confidence']
                    ])
                ])
                
                edge_score = self.edge_scorer(pair_features.unsqueeze(0))
                
                if edge_score > 0.5:  # Threshold
                    edge_list.append([node_i.item(), node_j.item()])
                    edge_list.append([node_j.item(), node_i.item()])  # Bidirectional
                    
                    # Create edge features
                    edge_feat = torch.zeros(8)
                    edge_feat[0] = edge_score
                    edge_feat[1] = memory_output['prototype_confidence'].item() \
                        if torch.is_tensor(memory_output['prototype_confidence']) \
                        else memory_output['prototype_confidence']
                    edge_feat[2] = anticipation['time_to_crisis'] / 30
                    
                    edge_features.append(edge_feat)
                    edge_features.append(edge_feat)  # Same for both directions
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_attr = torch.stack(edge_features)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = None
        
        return edge_index, edge_attr


def integrate_meta_learning(
    base_model: nn.Module,
    config: Dict[str, Any]
) -> MetaLearningRALECGNN:
    """
    Factory function to create meta-learning enhanced RALEC-GNN.
    """
    meta_model = MetaLearningRALECGNN(
        base_model=base_model,
        num_features=config['num_features'],
        num_assets=config['num_assets'],
        hidden_dim=config.get('hidden_dim', 256),
        memory_size=config.get('memory_size', 1000),
        use_meta_learning=True
    )
    
    logger.info("Meta-learning integration complete")
    logger.info("Features:")
    logger.info("  - Crisis episode memory bank")
    logger.info("  - Prototype-based learning")
    logger.info("  - Rapid crisis adaptation")
    logger.info("  - Pattern matching and retrieval")
    logger.info("  - Crisis anticipation module")
    
    return meta_model


if __name__ == "__main__":
    # Test integration
    logger.info("Testing meta-learning integration...")
    
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
        'memory_size': 500
    }
    
    # Create meta-learning model
    base_model = DummyModel()
    meta_model = integrate_meta_learning(base_model, config)
    
    # Test with dummy data
    graph_sequence = []
    for t in range(20):
        # Simulate crisis development
        if t < 10:
            # Normal period
            x = torch.randn(config['num_assets'], config['num_features']) * 0.1
        else:
            # Crisis period
            x = torch.randn(config['num_assets'], config['num_features']) * 0.5
            x += torch.randn(1, config['num_features']) * 0.3  # Common factor
        
        graph = Data(
            x=x,
            edge_index=torch.randint(0, config['num_assets'], (2, 100)),
            edge_attr=torch.randn(100, 8)
        )
        graph_sequence.append(graph)
    
    # Forward pass
    output = meta_model(graph_sequence, return_meta_insights=True)
    
    logger.info(f"Output keys: {list(output.keys())}")
    logger.info(f"Crisis anticipation: {output['crisis_anticipation']['time_to_crisis']:.1f} days")
    logger.info(f"Crisis state: {output['crisis_state']}")
    
    if output['memory_insights']:
        logger.info(f"Retrieved {len(output['memory_insights']['similar_episodes'])} similar episodes")
        logger.info(f"Prototype match confidence: {output['memory_insights']['prototype_confidence']:.3f}")