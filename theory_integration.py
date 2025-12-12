#!/usr/bin/env python3
"""
Integration of Theoretical Framework with RALEC-GNN
Connects Financial Network Morphology Theory to practical implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.data import Data
import logging

from theoretical_framework import (
    FinancialNetworkMorphology,
    MarketPhaseSpace,
    IsingModelRegimeDetector,
    PercolationTheoryAnalyzer,
    TheoreticalMetrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TheoryGuidedEdgeConstructor(nn.Module):
    """
    Edge constructor that incorporates theoretical principles from FNMT.
    
    Combines learned parameters with theoretical constraints:
    1. Regime-dependent edge formation probabilities
    2. Stress-induced preferential attachment
    3. Information asymmetry penalties
    """
    
    def __init__(self, node_features: int, edge_features: int, hidden_dim: int = 128):
        super().__init__()
        
        # Initialize theoretical framework
        self.theory = FinancialNetworkMorphology()
        
        # Learned components
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Regime-specific parameters (learned)
        self.regime_params = nn.Parameter(torch.randn(3, 3))  # α, β, γ for each regime
        
        # Market state encoder
        self.market_encoder = nn.Sequential(
            nn.Linear(3, 32),  # volatility, correlation, liquidity
            nn.ReLU(),
            nn.Linear(32, 3)   # regime logits
        )
        
    def compute_market_state(self, graph: Data) -> MarketPhaseSpace:
        """Extract market state from graph features"""
        # Aggregate node features
        node_features = graph.x
        
        # Compute market indicators
        volatility = torch.std(node_features).item()
        
        # Compute pairwise correlations efficiently
        normalized = (node_features - node_features.mean(0)) / (node_features.std(0) + 1e-8)
        correlation_matrix = torch.mm(normalized, normalized.t()) / node_features.shape[0]
        avg_correlation = correlation_matrix.mean().item()
        
        # Liquidity proxy (inverse of volatility spread)
        vol_spread = torch.std(torch.std(node_features, dim=0)).item()
        liquidity = 1 / (1 + vol_spread)
        
        return MarketPhaseSpace(
            volatility=min(volatility, 1.0),
            correlation=np.clip(avg_correlation, -1, 1),
            liquidity=min(liquidity, 1.0)
        )
    
    def forward(self, graph: Data) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Construct edges using theory-guided approach.
        
        Returns:
            edge_index: Learned edges
            info: Dictionary with theoretical metrics
        """
        # Encode nodes
        node_embeddings = self.node_encoder(graph.x)
        
        # Compute market state
        market_state = self.compute_market_state(graph)
        market_tensor = torch.tensor(
            [market_state.volatility, market_state.correlation, market_state.liquidity],
            device=graph.x.device
        )
        
        # Get regime probabilities
        regime_logits = self.market_encoder(market_tensor)
        regime_probs = F.softmax(regime_logits, dim=0)
        
        # Compute all pairwise edge probabilities
        num_nodes = graph.x.shape[0]
        edge_probs = torch.zeros(num_nodes, num_nodes, device=graph.x.device)
        
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                # Node similarity
                similarity = F.cosine_similarity(
                    node_embeddings[i].unsqueeze(0),
                    node_embeddings[j].unsqueeze(0)
                ).item()
                
                # Market stress
                stress = (market_state.volatility * market_state.correlation) / (market_state.liquidity + 0.1)
                
                # Information asymmetry
                asymmetry = torch.abs(graph.x[i].std() - graph.x[j].std()).item()
                
                # Regime-weighted parameters
                params = torch.matmul(regime_probs, self.regime_params)
                alpha, beta, gamma = params[0], params[1], params[2]
                
                # Theory-based edge probability
                logit = alpha * similarity + beta * stress - gamma * asymmetry
                prob = torch.sigmoid(logit)
                
                edge_probs[i, j] = prob
                edge_probs[j, i] = prob
        
        # Sample edges based on probabilities (or take top-k)
        k = min(int(num_nodes * num_nodes * 0.1), 100)  # Top 10% or 100 edges
        top_k = torch.topk(edge_probs.flatten(), k)
        
        # Convert to edge_index format
        edge_index = []
        for idx in top_k.indices:
            i = idx // num_nodes
            j = idx % num_nodes
            if i < j:  # Avoid duplicates
                edge_index.append([i.item(), j.item()])
                edge_index.append([j.item(), i.item()])
        
        edge_index = torch.tensor(edge_index, device=graph.x.device).t()
        
        # Theoretical metrics for monitoring
        info = {
            'market_state': market_state,
            'regime_probs': regime_probs.detach().cpu().numpy(),
            'avg_edge_prob': edge_probs.mean().item(),
            'network_density': len(edge_index[0]) / (num_nodes * (num_nodes - 1)),
            'stress_level': stress
        }
        
        return edge_index, info


class PhaseTransitionDetector(nn.Module):
    """
    Detects and predicts phase transitions using theoretical indicators.
    
    Monitors:
    1. Order parameter fluctuations (susceptibility)
    2. Critical slowing down
    3. Network topology changes
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        self.ising_model = IsingModelRegimeDetector(num_assets=100)  # Will be resized
        self.percolation = PercolationTheoryAnalyzer()
        
        # Neural transition predictor
        self.transition_net = nn.Sequential(
            nn.Linear(10, hidden_dim),  # Multiple indicators
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # No transition, Bull→Normal, Normal→Crisis, Crisis→Bull
        )
        
        # History tracking
        self.history_window = 20
        self.spin_history = []
        self.metric_history = []
        
    def update_history(self, graph: Data, adjacency_matrix: np.ndarray):
        """Update historical tracking"""
        # Convert features to spins (risk-on/off)
        returns = graph.x[:, 0].cpu().numpy()  # Assuming first feature is returns
        spins = np.sign(returns)
        
        self.spin_history.append(spins)
        if len(self.spin_history) > self.history_window:
            self.spin_history.pop(0)
        
        # Track metrics
        metrics = {
            'volatility': np.std(returns),
            'correlation': np.corrcoef(graph.x.cpu().numpy().T)[0, 1:].mean(),
            'giant_component': self.percolation.find_giant_component(adjacency_matrix)[0] / len(returns)
        }
        self.metric_history.append(metrics)
        if len(self.metric_history) > self.history_window:
            self.metric_history.pop(0)
    
    def compute_transition_indicators(self) -> torch.Tensor:
        """Compute theoretical indicators of phase transition"""
        if len(self.spin_history) < 5:
            return torch.zeros(10)
        
        indicators = []
        
        # 1. Magnetization (market sentiment)
        magnetizations = [self.ising_model.compute_magnetization(s) for s in self.spin_history[-5:]]
        indicators.append(np.mean(magnetizations))
        indicators.append(np.std(magnetizations))  # Susceptibility proxy
        
        # 2. Autocorrelation (critical slowing down)
        if len(self.metric_history) >= 10:
            vol_series = [m['volatility'] for m in self.metric_history[-10:]]
            autocorr = np.corrcoef(vol_series[:-1], vol_series[1:])[0, 1]
            indicators.append(autocorr)
        else:
            indicators.append(0.0)
        
        # 3. Network indicators
        if self.metric_history:
            latest = self.metric_history[-1]
            indicators.append(latest['volatility'])
            indicators.append(latest['correlation'])
            indicators.append(latest['giant_component'])
        else:
            indicators.extend([0.0, 0.0, 0.0])
        
        # 4. Rate of change indicators
        if len(self.metric_history) >= 2:
            vol_change = self.metric_history[-1]['volatility'] - self.metric_history[-2]['volatility']
            corr_change = self.metric_history[-1]['correlation'] - self.metric_history[-2]['correlation']
            giant_change = self.metric_history[-1]['giant_component'] - self.metric_history[-2]['giant_component']
            indicators.extend([vol_change, corr_change, giant_change])
        else:
            indicators.extend([0.0, 0.0, 0.0])
        
        # 5. Energy (from Ising model)
        if self.spin_history:
            energy = self.ising_model.compute_energy(self.spin_history[-1])
            indicators.append(energy / len(self.spin_history[-1]))  # Normalized
        else:
            indicators.append(0.0)
        
        return torch.tensor(indicators, dtype=torch.float32)
    
    def forward(self, graph: Data, adjacency_matrix: torch.Tensor) -> Dict[str, Any]:
        """
        Detect phase transition probability.
        
        Returns:
            Dictionary with transition predictions and indicators
        """
        # Update history
        adj_np = adjacency_matrix.detach().cpu().numpy()
        self.update_history(graph, adj_np)
        
        # Compute indicators
        indicators = self.compute_transition_indicators().to(graph.x.device)
        
        # Neural prediction
        transition_logits = self.transition_net(indicators)
        transition_probs = F.softmax(transition_logits, dim=0)
        
        # Theory-based detection
        if len(self.spin_history) >= self.history_window:
            ising_detection = self.ising_model.detect_phase_transition(
                np.array(self.spin_history)
            )
        else:
            ising_detection = {'transition_probability': 0.0, 'susceptibility': 0.0}
        
        return {
            'transition_probs': transition_probs.detach().cpu().numpy(),
            'transition_types': ['No transition', 'Bull→Normal', 'Normal→Crisis', 'Crisis→Bull'],
            'ising_transition_prob': ising_detection['transition_probability'],
            'susceptibility': ising_detection['susceptibility'],
            'indicators': indicators.detach().cpu().numpy()
        }


class TheoryInformedLoss(nn.Module):
    """
    Loss function incorporating theoretical constraints.
    
    Encourages:
    1. Regime-appropriate network density
    2. Realistic clustering patterns
    3. Power-law degree distributions during crisis
    """
    
    def __init__(self, morphology: FinancialNetworkMorphology):
        super().__init__()
        self.morphology = morphology
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        market_state: MarketPhaseSpace,
        adjacency_matrix: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute theory-informed loss components.
        """
        losses = {}
        
        # 1. Standard prediction loss
        if 'regime_logits' in predictions and 'regime_labels' in targets:
            losses['prediction'] = F.cross_entropy(
                predictions['regime_logits'],
                targets['regime_labels']
            )
        
        # 2. Network density regularization
        regime_probs = self.morphology.compute_regime_probability(market_state)
        expected_densities = torch.tensor([0.2, 0.3, 0.5])  # Bull, Normal, Crisis
        expected_density = torch.dot(
            torch.tensor(regime_probs, dtype=torch.float32),
            expected_densities
        ).to(adjacency_matrix.device)
        
        actual_density = adjacency_matrix.mean()
        losses['density'] = F.mse_loss(actual_density, expected_density)
        
        # 3. Clustering coefficient regularization
        # During crisis, clustering should increase
        if regime_probs[2] > 0.5:  # Crisis regime
            # Encourage triangle formation
            # Count triangles: Tr(A³) / 6
            A_squared = torch.matmul(adjacency_matrix, adjacency_matrix)
            A_cubed = torch.matmul(A_squared, adjacency_matrix)
            num_triangles = torch.trace(A_cubed) / 6
            
            # Normalize by number of possible triangles
            n = adjacency_matrix.shape[0]
            max_triangles = n * (n - 1) * (n - 2) / 6
            clustering = num_triangles / (max_triangles + 1e-8)
            
            # Higher clustering in crisis
            target_clustering = torch.tensor(0.6, device=adjacency_matrix.device)
            losses['clustering'] = F.mse_loss(clustering, target_clustering)
        else:
            losses['clustering'] = torch.tensor(0.0, device=adjacency_matrix.device)
        
        # 4. Spectral radius regularization (contagion potential)
        # Limit spectral radius to prevent explosive contagion
        eigenvalues, _ = torch.linalg.eig(adjacency_matrix)
        spectral_radius = torch.max(torch.abs(eigenvalues.real))
        max_radius = torch.tensor(2.0, device=adjacency_matrix.device)
        losses['spectral'] = F.relu(spectral_radius - max_radius)
        
        # Combine losses with weights
        total_loss = (
            losses.get('prediction', 0) +
            0.1 * losses['density'] +
            0.05 * losses.get('clustering', 0) +
            0.01 * losses['spectral']
        )
        
        losses['total'] = total_loss
        
        return losses


class TheoreticalRALECGNN(nn.Module):
    """
    RALEC-GNN enhanced with theoretical framework.
    
    Integrates:
    1. Theory-guided edge construction
    2. Phase transition detection
    3. Theory-informed loss
    4. Interpretable metrics
    """
    
    def __init__(self, base_model: nn.Module, config: dict):
        super().__init__()
        
        self.base_model = base_model
        self.config = config
        
        # Theoretical components
        self.morphology = FinancialNetworkMorphology()
        self.theory_edge_constructor = TheoryGuidedEdgeConstructor(
            node_features=config['node_features'],
            edge_features=config.get('edge_features', 8),
            hidden_dim=config.get('hidden_dim', 128)
        )
        self.phase_detector = PhaseTransitionDetector()
        self.theory_metrics = TheoreticalMetrics(self.morphology)
        
    def forward(
        self,
        graph_sequence: List[Data],
        return_theory_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass with theoretical enhancements.
        """
        # Get base model predictions
        base_output = self.base_model(graph_sequence)
        
        # Enhance with theoretical components
        theory_outputs = []
        
        for graph in graph_sequence:
            # Theory-guided edge construction
            theory_edges, edge_info = self.theory_edge_constructor(graph)
            
            # Create adjacency matrix
            num_nodes = graph.x.shape[0]
            adj_matrix = torch.zeros(num_nodes, num_nodes, device=graph.x.device)
            if theory_edges.shape[1] > 0:
                adj_matrix[theory_edges[0], theory_edges[1]] = 1.0
            
            # Phase transition detection
            transition_info = self.phase_detector(graph, adj_matrix)
            
            # Theoretical metrics
            if return_theory_metrics:
                metrics = self.theory_metrics.compute_theoretical_metrics(
                    adj_matrix.cpu().numpy(),
                    graph.x.cpu().numpy(),
                    edge_info['market_state']
                )
            else:
                metrics = {}
            
            theory_outputs.append({
                'edge_info': edge_info,
                'transition_info': transition_info,
                'metrics': metrics
            })
        
        # Combine outputs
        output = {
            **base_output,
            'theory_analysis': theory_outputs,
            'market_state': theory_outputs[-1]['edge_info']['market_state'] if theory_outputs else None
        }
        
        return output


if __name__ == "__main__":
    # Example integration test
    logger.info("Testing theoretical framework integration...")
    
    # Create dummy graph
    num_nodes = 50
    num_features = 16
    
    graph = Data(
        x=torch.randn(num_nodes, num_features),
        edge_index=torch.randint(0, num_nodes, (2, 100)),
        edge_attr=torch.randn(100, 8)
    )
    
    # Test edge constructor
    edge_constructor = TheoryGuidedEdgeConstructor(num_features, 8)
    edges, info = edge_constructor(graph)
    
    logger.info(f"Constructed {edges.shape[1]} edges")
    logger.info(f"Market state: {info['market_state']}")
    logger.info(f"Regime probabilities: Bull={info['regime_probs'][0]:.2%}, "
               f"Normal={info['regime_probs'][1]:.2%}, Crisis={info['regime_probs'][2]:.2%}")
    
    # Test phase detector
    phase_detector = PhaseTransitionDetector()
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    adj_matrix[edges[0], edges[1]] = 1.0
    
    transition_info = phase_detector(graph, adj_matrix)
    logger.info(f"Transition probabilities: {transition_info['transition_probs']}")