#!/usr/bin/env python3
"""
Integration of Causal Discovery with RALEC-GNN
Replaces correlation-based edges with discovered causal relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.data import Data
import logging

from causal_discovery import (
    CausalDiscoveryModule,
    CausalContagionPredictor,
    CausalEdge
)
from theoretical_framework import MarketPhaseSpace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalRALECGNN(nn.Module):
    """
    RALEC-GNN enhanced with causal discovery.
    
    Key improvements:
    1. Edges based on causal relationships, not correlations
    2. Directional information flow following causality
    3. Regime-aware causal strength modulation
    4. Contagion prediction using causal paths
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_features: int,
        num_assets: int,
        max_lag: int = 5,
        use_causal_discovery: bool = True
    ):
        super().__init__()
        
        self.base_model = base_model
        self.num_assets = num_assets
        self.use_causal_discovery = use_causal_discovery
        
        if use_causal_discovery:
            # Initialize causal discovery module
            self.causal_discovery = CausalDiscoveryModule(
                num_features=num_assets,
                max_lag=max_lag,
                use_theory_constraints=True
            )
            
            # Contagion predictor
            self.contagion_predictor = CausalContagionPredictor()
            
            # Causal edge attention
            self.causal_attention = CausalEdgeAttention(
                edge_dim=8,
                hidden_dim=64
            )
        
        # Crisis detection enhancement
        self.crisis_detector = nn.Sequential(
            nn.Linear(num_features + 8, 64),  # Node features + causal metrics
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        graph_sequence: List[Data],
        return_causal_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Process graph sequence with causal discovery.
        """
        if self.use_causal_discovery:
            # Discover causal relationships
            causal_results = self.causal_discovery(graph_sequence)
            
            # Replace correlation edges with causal edges
            enhanced_sequence = []
            for i, (graph, causal_result) in enumerate(zip(graph_sequence, causal_results)):
                enhanced_graph = self._enhance_with_causality(graph, causal_result)
                enhanced_sequence.append(enhanced_graph)
            
            # Use enhanced sequence for prediction
            base_output = self.base_model(enhanced_sequence)
            
            # Add causal-specific predictions
            causal_predictions = self._compute_causal_predictions(
                enhanced_sequence, causal_results
            )
            
            output = {
                **base_output,
                **causal_predictions
            }
            
            if return_causal_analysis:
                output['causal_analysis'] = causal_results
        else:
            # Standard forward pass without causal discovery
            output = self.base_model(graph_sequence)
        
        return output
    
    def _enhance_with_causality(
        self,
        graph: Data,
        causal_result: Dict[str, Any]
    ) -> Data:
        """
        Enhance graph with discovered causal relationships.
        """
        # Create new graph with causal edges
        enhanced_graph = Data(
            x=graph.x.clone(),
            edge_index=causal_result['edge_index'],
            edge_attr=causal_result['edge_features']
        )
        
        # Add causal metrics to node features
        causal_adjacency = causal_result['causal_adjacency']
        
        # Compute node-level causal metrics
        in_strength = causal_adjacency.sum(dim=0)  # How much node is influenced
        out_strength = causal_adjacency.sum(dim=1)  # How much node influences others
        
        # Betweenness centrality proxy
        total_paths = causal_adjacency @ causal_adjacency
        betweenness_proxy = total_paths.sum(dim=0) + total_paths.sum(dim=1)
        
        # Stack causal metrics
        causal_node_features = torch.stack([
            in_strength,
            out_strength,
            betweenness_proxy,
            in_strength / (out_strength + 1e-6)  # Vulnerability ratio
        ], dim=1)
        
        # Concatenate with original features
        enhanced_graph.x = torch.cat([graph.x, causal_node_features], dim=1)
        
        # Apply causal attention to edges
        if hasattr(self, 'causal_attention'):
            enhanced_graph.edge_attr = self.causal_attention(
                enhanced_graph.edge_attr,
                enhanced_graph.x,
                enhanced_graph.edge_index
            )
        
        return enhanced_graph
    
    def _compute_causal_predictions(
        self,
        graph_sequence: List[Data],
        causal_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute predictions specific to causal structure.
        """
        predictions = {}
        
        # Latest graph and causal structure
        latest_graph = graph_sequence[-1]
        latest_causal = causal_results[-1]
        
        # 1. Crisis prediction using causal features
        causal_features = self._extract_causal_features(latest_causal)
        crisis_input = torch.cat([
            latest_graph.x.mean(dim=0),  # Average node features
            causal_features
        ])
        
        crisis_prob = self.crisis_detector(crisis_input.unsqueeze(0))
        predictions['causal_crisis_probability'] = crisis_prob.squeeze()
        
        # 2. Contagion simulation
        # Identify potential shock nodes (high volatility or negative returns)
        returns = latest_graph.x[:, 0]  # Assuming first feature is returns
        volatility = latest_graph.x[:, 1]  # Assuming second feature is volatility
        
        shock_candidates = torch.where(
            (returns < returns.mean() - 2 * returns.std()) |
            (volatility > volatility.mean() + 2 * volatility.std())
        )[0].tolist()
        
        if shock_candidates:
            contagion_result = self.contagion_predictor.predict_contagion_path(
                latest_causal['causal_adjacency'],
                shock_candidates[:3],  # Top 3 shock nodes
                latest_graph.x
            )
            
            predictions['contagion_risk'] = contagion_result['contagion_probabilities'].mean()
            predictions['systemic_nodes'] = contagion_result['critical_nodes']
            predictions['expected_affected'] = contagion_result['num_affected']
        
        # 3. Causal regime indicator
        # High causal density + high strength = crisis regime
        causal_density = (latest_causal['causal_adjacency'] > 0).float().mean()
        avg_strength = latest_causal['causal_adjacency'][
            latest_causal['causal_adjacency'] > 0
        ].mean() if (latest_causal['causal_adjacency'] > 0).any() else torch.tensor(0.0)
        
        predictions['causal_regime_score'] = causal_density * avg_strength
        
        return predictions
    
    def _extract_causal_features(self, causal_result: Dict[str, Any]) -> torch.Tensor:
        """
        Extract summary features from causal discovery results.
        """
        causal_adj = causal_result['causal_adjacency']
        
        features = []
        
        # Network density
        density = (causal_adj > 0).float().mean()
        features.append(density)
        
        # Average causal strength
        if (causal_adj > 0).any():
            avg_strength = causal_adj[causal_adj > 0].mean()
        else:
            avg_strength = torch.tensor(0.0)
        features.append(avg_strength)
        
        # Clustering coefficient proxy
        A_squared = causal_adj @ causal_adj
        A_cubed = A_squared @ causal_adj
        clustering = torch.trace(A_cubed) / (causal_adj.sum() + 1e-6)
        features.append(clustering)
        
        # Spectral radius (largest eigenvalue)
        if causal_adj.shape[0] > 1:
            eigenvalues, _ = torch.linalg.eig(causal_adj)
            spectral_radius = torch.max(torch.abs(eigenvalues.real))
        else:
            spectral_radius = torch.tensor(0.0)
        features.append(spectral_radius)
        
        # Number of strongly connected components
        # (simplified: ratio of bidirectional edges)
        bidirectional = ((causal_adj > 0) & (causal_adj.t() > 0)).float().sum()
        total_edges = (causal_adj > 0).float().sum()
        scc_ratio = bidirectional / (total_edges + 1e-6)
        features.append(scc_ratio)
        
        # Average shortest path length proxy
        # Using powers of adjacency matrix
        reachability = torch.eye(causal_adj.shape[0])
        for i in range(1, 4):
            reachability += torch.matrix_power((causal_adj > 0).float(), i)
        avg_distance = 1 / (reachability.mean() + 1e-6)
        features.append(avg_distance)
        
        # Causal heterogeneity (std of causal strengths)
        if (causal_adj > 0).any():
            heterogeneity = causal_adj[causal_adj > 0].std()
        else:
            heterogeneity = torch.tensor(0.0)
        features.append(heterogeneity)
        
        # Hub concentration (max out-degree / avg out-degree)
        out_degrees = (causal_adj > 0).float().sum(dim=1)
        if out_degrees.mean() > 0:
            hub_concentration = out_degrees.max() / out_degrees.mean()
        else:
            hub_concentration = torch.tensor(1.0)
        features.append(hub_concentration)
        
        return torch.stack(features)


class CausalEdgeAttention(nn.Module):
    """
    Attention mechanism for causal edges that considers:
    1. Causal strength and confidence
    2. Temporal lag
    3. Mechanism type
    4. Current market regime
    """
    
    def __init__(self, edge_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.node_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_dim, edge_dim)
        
    def forward(
        self,
        edge_attr: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention to enhance causal edge features.
        """
        if edge_attr.shape[0] == 0:
            return edge_attr
        
        # Encode edges
        edge_encoded = self.edge_encoder(edge_attr)
        
        # Get source and target node features for each edge
        source_features = node_features[edge_index[0]]
        target_features = node_features[edge_index[1]]
        
        # Combine node features
        node_context = (source_features + target_features) / 2
        
        # Take first few dimensions if needed
        if node_context.shape[1] > edge_attr.shape[1]:
            node_context = node_context[:, :edge_attr.shape[1]]
        
        node_encoded = self.node_encoder(node_context)
        
        # Apply attention
        # Query: edge features, Key/Value: node context
        edge_enhanced, _ = self.attention(
            edge_encoded.unsqueeze(1),
            node_encoded.unsqueeze(1),
            node_encoded.unsqueeze(1)
        )
        
        edge_enhanced = edge_enhanced.squeeze(1)
        
        # Project back to edge dimension
        edge_attr_enhanced = self.output_proj(edge_enhanced)
        
        # Residual connection
        return edge_attr + 0.5 * edge_attr_enhanced


class CausalLoss(nn.Module):
    """
    Loss function that incorporates causal structure.
    
    Encourages:
    1. Predictions consistent with causal flow
    2. Stronger penalties for missing causal propagation
    3. Regime-appropriate causal density
    """
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.05):
        super().__init__()
        self.alpha = alpha  # Weight for causal consistency
        self.beta = beta   # Weight for density regularization
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        causal_adjacency: torch.Tensor,
        regime_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute causal-aware loss.
        """
        # Standard prediction loss
        if 'regime_logits' in predictions and 'regime_labels' in targets:
            prediction_loss = F.cross_entropy(
                predictions['regime_logits'],
                targets['regime_labels']
            )
        else:
            prediction_loss = torch.tensor(0.0)
        
        # Causal consistency loss
        if 'node_predictions' in predictions and 'node_targets' in targets:
            node_preds = predictions['node_predictions']
            node_targets = targets['node_targets']
            
            # Compute prediction errors
            errors = (node_preds - node_targets).abs()
            
            # Weight errors by causal influence
            # Errors in causally downstream nodes should be penalized less
            # if upstream predictions are also wrong
            causal_weights = 1 + self.alpha * causal_adjacency.sum(dim=0)
            weighted_errors = errors * causal_weights
            
            causal_loss = weighted_errors.mean()
        else:
            causal_loss = torch.tensor(0.0)
        
        # Causal density regularization
        # Different regimes should have different edge densities
        current_density = (causal_adjacency > 0).float().mean()
        
        # Expected densities: [Bull, Normal, Crisis]
        expected_densities = torch.tensor([0.2, 0.3, 0.5])
        expected_density = torch.sum(regime_probs * expected_densities)
        
        density_loss = self.beta * (current_density - expected_density) ** 2
        
        # Total loss
        total_loss = prediction_loss + causal_loss + density_loss
        
        return total_loss


def integrate_causal_discovery(
    base_model: nn.Module,
    config: Dict[str, Any]
) -> CausalRALECGNN:
    """
    Factory function to create causal-enhanced RALEC-GNN.
    """
    causal_model = CausalRALECGNN(
        base_model=base_model,
        num_features=config['num_features'],
        num_assets=config['num_assets'],
        max_lag=config.get('max_lag', 5),
        use_causal_discovery=True
    )
    
    logger.info("Causal discovery integration complete")
    logger.info("Features:")
    logger.info("  - Neural Granger causality")
    logger.info("  - PCMCI for contemporaneous relationships")
    logger.info("  - Threshold effect detection")
    logger.info("  - Contagion path prediction")
    logger.info("  - Causal attention mechanism")
    
    return causal_model


if __name__ == "__main__":
    # Example integration
    logger.info("Testing causal integration...")
    
    # Create dummy base model
    class DummyModel(nn.Module):
        def forward(self, x):
            return {'regime_logits': torch.randn(1, 3)}
    
    # Configuration
    config = {
        'num_features': 16,
        'num_assets': 50,
        'max_lag': 5
    }
    
    # Create causal model
    base_model = DummyModel()
    causal_model = integrate_causal_discovery(base_model, config)
    
    # Test with dummy data
    graph_sequence = []
    for _ in range(10):
        graph = Data(
            x=torch.randn(config['num_assets'], config['num_features']),
            edge_index=torch.randint(0, config['num_assets'], (2, 100)),
            edge_attr=torch.randn(100, 8)
        )
        graph_sequence.append(graph)
    
    # Forward pass
    output = causal_model(graph_sequence, return_causal_analysis=True)
    
    logger.info(f"Output keys: {list(output.keys())}")
    if 'causal_crisis_probability' in output:
        logger.info(f"Causal crisis probability: {output['causal_crisis_probability']:.3f}")
    if 'causal_regime_score' in output:
        logger.info(f"Causal regime score: {output['causal_regime_score']:.3f}")