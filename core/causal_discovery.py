#!/usr/bin/env python3
"""
RALEC-GNN Phase 3: Causal Discovery Module
Implements neural causal discovery to learn directional relationships between assets
Moving beyond correlation to causation for more accurate contagion modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import networkx as nx
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import logging
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CausalEdge:
    """Represents a discovered causal relationship"""
    source: int
    target: int
    strength: float
    confidence: float
    lag: int
    mechanism: str  # 'linear', 'nonlinear', 'threshold'


class NeuralGrangerCausality(nn.Module):
    """
    Neural network-based Granger causality test.
    
    Tests whether past values of X help predict Y better than
    past values of Y alone, using neural networks to capture
    nonlinear relationships.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, max_lag: int = 5):
        super().__init__()
        self.max_lag = max_lag
        
        # Model for Y based on past Y only
        self.model_restricted = nn.Sequential(
            nn.Linear(input_dim * max_lag, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Model for Y based on past Y and X
        self.model_full = nn.Sequential(
            nn.Linear(2 * input_dim * max_lag, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Attention mechanism for lag importance
        self.lag_attention = nn.Sequential(
            nn.Linear(max_lag, max_lag),
            nn.Softmax(dim=-1)
        )
        
    def prepare_lagged_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lag: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create lagged features for causality testing"""
        n = len(x) - lag
        
        # Past values of Y
        y_lagged = torch.stack([y[i:i+lag] for i in range(n)])
        
        # Past values of X
        x_lagged = torch.stack([x[i:i+lag] for i in range(n)])
        
        # Combined past values
        xy_lagged = torch.cat([x_lagged, y_lagged], dim=-1)
        
        # Target: current Y
        y_target = y[lag:].unsqueeze(-1)
        
        return y_lagged.flatten(1), xy_lagged.flatten(1), y_target, x_lagged
        
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Test if X Granger-causes Y.
        
        Returns causality score and optimal lag.
        """
        device = x.device
        causality_scores = []
        
        for lag in range(1, self.max_lag + 1):
            y_lagged, xy_lagged, y_target, x_lagged = self.prepare_lagged_data(x, y, lag)
            
            if len(y_lagged) < 10:  # Not enough data
                causality_scores.append(0.0)
                continue
            
            # Predictions with Y only
            y_pred_restricted = self.model_restricted(y_lagged)
            loss_restricted = F.mse_loss(y_pred_restricted, y_target)
            
            # Predictions with X and Y
            y_pred_full = self.model_full(xy_lagged)
            loss_full = F.mse_loss(y_pred_full, y_target)
            
            # Causality score: improvement in prediction
            improvement = (loss_restricted - loss_full) / (loss_restricted + 1e-8)
            causality_scores.append(improvement.item())
        
        causality_scores = torch.tensor(causality_scores, device=device)
        
        # Apply attention to find optimal lag
        lag_weights = self.lag_attention(torch.arange(self.max_lag, dtype=torch.float32, device=device))
        weighted_causality = torch.sum(causality_scores * lag_weights)
        
        optimal_lag = torch.argmax(causality_scores) + 1
        
        return {
            'causality_score': weighted_causality,
            'optimal_lag': optimal_lag,
            'lag_scores': causality_scores,
            'confidence': torch.sigmoid(weighted_causality * 10)  # Convert to probability
        }


class PCMCI_Neural(nn.Module):
    """
    Neural implementation of PCMCI (PC Momentary Conditional Independence).
    
    PCMCI is a state-of-the-art causal discovery algorithm that:
    1. Uses PC algorithm to find causal parents
    2. Tests conditional independence using MCI test
    3. Handles time series with autocorrelation
    """
    
    def __init__(self, num_variables: int, max_lag: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.num_variables = num_variables
        self.max_lag = max_lag
        
        # Neural conditional independence tester
        self.ci_tester = ConditionalIndependenceNet(hidden_dim)
        
        # Causal strength estimator
        self.strength_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def pc_skeleton_discovery(
        self,
        data: torch.Tensor,
        alpha: float = 0.05
    ) -> torch.Tensor:
        """
        Discover causal skeleton using PC algorithm.
        
        Returns adjacency matrix of potential causal relationships.
        """
        n_vars = self.num_variables
        adjacency = torch.ones(n_vars, n_vars, device=data.device)
        adjacency.fill_diagonal_(0)
        
        # Start with fully connected graph and remove edges
        for depth in range(n_vars):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if adjacency[i, j] == 0:
                        continue
                    
                    # Find separating set
                    neighbors_i = torch.where(adjacency[i] > 0)[0]
                    neighbors_j = torch.where(adjacency[j] > 0)[0]
                    common_neighbors = torch.tensor(
                        list(set(neighbors_i.tolist()) & set(neighbors_j.tolist())),
                        device=data.device
                    )
                    
                    if len(common_neighbors) >= depth:
                        # Test conditional independence
                        ci_result = self.test_conditional_independence(
                            data[:, i], data[:, j],
                            data[:, common_neighbors] if len(common_neighbors) > 0 else None,
                            alpha
                        )
                        
                        if ci_result['independent']:
                            adjacency[i, j] = 0
                            adjacency[j, i] = 0
        
        return adjacency
    
    def test_conditional_independence(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: Optional[torch.Tensor],
        alpha: float
    ) -> Dict[str, Any]:
        """Test if X ⊥ Y | Z using neural networks"""
        if z is None:
            # Unconditional independence test
            score = self.ci_tester(x.unsqueeze(-1), y.unsqueeze(-1), None)
        else:
            score = self.ci_tester(x.unsqueeze(-1), y.unsqueeze(-1), z)
        
        # Convert score to p-value approximation
        p_value = torch.exp(-score * 10).item()
        
        return {
            'independent': p_value > alpha,
            'p_value': p_value,
            'score': score.item()
        }
    
    def mci_test(
        self,
        data: torch.Tensor,
        skeleton: torch.Tensor
    ) -> torch.Tensor:
        """
        Momentary Conditional Independence test to orient edges.
        
        Returns directed adjacency matrix.
        """
        n_vars = self.num_variables
        directed_adj = torch.zeros_like(skeleton)
        
        for tau in range(1, self.max_lag + 1):
            for i in range(n_vars):
                for j in range(n_vars):
                    if skeleton[i, j] == 0:
                        continue
                    
                    # Get past values
                    if data.shape[0] <= tau:
                        continue
                        
                    x_past = data[:-tau, i]
                    y_current = data[tau:, j]
                    
                    # Condition on parents of Y
                    parents_j = torch.where(skeleton[:, j] > 0)[0]
                    parents_j = parents_j[parents_j != i]  # Exclude i
                    
                    if len(parents_j) > 0:
                        z = data[:-tau, parents_j]
                    else:
                        z = None
                    
                    # Test X_past → Y_current | Parents(Y)
                    ci_result = self.test_conditional_independence(
                        x_past, y_current, z, alpha=0.05
                    )
                    
                    if not ci_result['independent']:
                        # Estimate causal strength
                        strength = self.estimate_causal_strength(x_past, y_current, z)
                        directed_adj[i, j] = strength
        
        return directed_adj
    
    def estimate_causal_strength(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: Optional[torch.Tensor]
    ) -> float:
        """Estimate strength of causal relationship X → Y | Z"""
        x_embed = self.ci_tester.encoder(x.unsqueeze(-1))
        y_embed = self.ci_tester.encoder(y.unsqueeze(-1))
        
        combined = torch.cat([x_embed.mean(0), y_embed.mean(0)], dim=-1)
        strength = self.strength_estimator(combined)
        
        return strength.item()
    
    def forward(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Discover causal graph from multivariate time series.
        
        Args:
            data: Time series data of shape (time, variables)
            
        Returns:
            Dictionary containing causal graph and metrics
        """
        # Step 1: PC skeleton discovery
        skeleton = self.pc_skeleton_discovery(data)
        
        # Step 2: MCI test for edge orientation
        causal_graph = self.mci_test(data, skeleton)
        
        # Step 3: Post-processing
        # Remove very weak edges
        causal_graph[causal_graph < 0.1] = 0
        
        return {
            'causal_graph': causal_graph,
            'skeleton': skeleton,
            'num_edges': torch.sum(causal_graph > 0),
            'avg_strength': torch.mean(causal_graph[causal_graph > 0])
            if torch.any(causal_graph > 0) else torch.tensor(0.0)
        }


class ConditionalIndependenceNet(nn.Module):
    """
    Neural network for testing conditional independence.
    
    Uses the principle: X ⊥ Y | Z iff I(X;Y|Z) = 0
    where I is conditional mutual information.
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.conditional_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.independence_score = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute conditional independence score.
        
        High score = dependent, Low score = independent
        """
        x_encoded = self.encoder(x)
        y_encoded = self.encoder(y)
        
        if z is None:
            # Unconditional case
            combined = torch.cat([x_encoded, y_encoded, torch.zeros_like(x_encoded)], dim=-1)
        else:
            if z.dim() == 1:
                z = z.unsqueeze(-1)
            z_encoded = self.encoder(z.mean(dim=-1, keepdim=True) if z.shape[-1] > 1 else z)
            combined = torch.cat([x_encoded, y_encoded, z_encoded], dim=-1)
        
        conditional_features = self.conditional_encoder(combined)
        score = self.independence_score(conditional_features)
        
        return score.squeeze()


class ThresholdCausalityDetector(nn.Module):
    """
    Detects threshold-based causality where relationships change
    dramatically at certain levels (e.g., margin calls, stop losses).
    """
    
    def __init__(self, input_dim: int = 1, num_thresholds: int = 3):
        super().__init__()
        
        self.num_thresholds = num_thresholds
        
        # Learn threshold values
        self.thresholds = nn.Parameter(torch.linspace(-2, 2, num_thresholds))
        
        # Regime-specific causal models
        self.regime_models = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(num_thresholds + 1)
        ])
        
        # Threshold sharpness (learnable)
        self.sharpness = nn.Parameter(torch.tensor(10.0))
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """
        Detect threshold causality from X to Y.
        
        Returns regime-specific causal strengths.
        """
        # Sort thresholds
        sorted_thresholds, _ = torch.sort(self.thresholds)
        
        # Compute regime probabilities for each sample
        regime_probs = []
        
        for i in range(self.num_thresholds + 1):
            if i == 0:
                # Below first threshold
                prob = torch.sigmoid(-self.sharpness * (x - sorted_thresholds[0]))
            elif i == self.num_thresholds:
                # Above last threshold
                prob = torch.sigmoid(self.sharpness * (x - sorted_thresholds[-1]))
            else:
                # Between thresholds
                prob_low = torch.sigmoid(self.sharpness * (x - sorted_thresholds[i-1]))
                prob_high = torch.sigmoid(-self.sharpness * (x - sorted_thresholds[i]))
                prob = prob_low * prob_high
            
            regime_probs.append(prob)
        
        regime_probs = torch.stack(regime_probs, dim=-1)
        
        # Compute regime-specific predictions
        predictions = []
        for i, model in enumerate(self.regime_models):
            pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=-1)
        
        # Weighted prediction
        y_pred = torch.sum(predictions * regime_probs, dim=-1)
        
        # Causal strength per regime
        causal_strengths = []
        for i in range(self.num_thresholds + 1):
            mask = regime_probs[:, i] > 0.5
            if mask.sum() > 0:
                strength = F.mse_loss(predictions[mask, i], y[mask])
                causal_strengths.append(1 - strength.item())  # Higher is stronger
            else:
                causal_strengths.append(0.0)
        
        return {
            'predictions': y_pred,
            'regime_probs': regime_probs,
            'thresholds': sorted_thresholds,
            'causal_strengths': torch.tensor(causal_strengths),
            'total_causality': F.mse_loss(y_pred, y)
        }


class CausalDiscoveryModule(nn.Module):
    """
    Complete causal discovery module for RALEC-GNN.
    
    Combines multiple causal discovery methods:
    1. Neural Granger causality for time-lagged effects
    2. PCMCI for contemporaneous and lagged relationships
    3. Threshold detection for regime-dependent causality
    4. Integration with theoretical framework
    """
    
    def __init__(
        self,
        num_features: int,
        max_lag: int = 5,
        hidden_dim: int = 128,
        use_theory_constraints: bool = True
    ):
        super().__init__()
        
        self.num_features = num_features
        self.max_lag = max_lag
        self.use_theory_constraints = use_theory_constraints
        
        # Causal discovery methods
        self.granger_net = NeuralGrangerCausality(
            input_dim=1,
            hidden_dim=hidden_dim,
            max_lag=max_lag
        )
        
        self.pcmci = PCMCI_Neural(
            num_variables=num_features,
            max_lag=max_lag,
            hidden_dim=hidden_dim
        )
        
        self.threshold_detector = ThresholdCausalityDetector()
        
        # Edge encoder for GNN
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # strength, lag, confidence, mechanism
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 8)  # Edge features for GNN
        )
        
        # Causal graph refiner
        self.graph_refiner = nn.Sequential(
            nn.Linear(num_features * num_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_features * num_features),
            nn.Sigmoid()
        )
        
    def discover_causal_graph(
        self,
        time_series: torch.Tensor,
        market_state: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Discover causal relationships from multivariate time series.
        
        Args:
            time_series: Shape (time, assets, features)
            market_state: Current market regime information
            
        Returns:
            Causal graph and discovered relationships
        """
        n_time, n_assets, n_features = time_series.shape
        
        # 1. PCMCI for overall causal structure
        # Use returns (first feature) for main causal discovery
        returns_data = time_series[:, :, 0]  # Shape: (time, assets)
        pcmci_result = self.pcmci(returns_data)
        causal_adj = pcmci_result['causal_graph']
        
        # 2. Detailed edge analysis with Granger causality
        causal_edges = []
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    continue
                    
                # Test Granger causality
                granger_result = self.granger_net(
                    returns_data[:, i],
                    returns_data[:, j]
                )
                
                if granger_result['causality_score'] > 0.1:  # Threshold
                    # Test for threshold effects
                    threshold_result = self.threshold_detector(
                        returns_data[:, i].unsqueeze(-1),
                        returns_data[:, j]
                    )
                    
                    # Determine mechanism type
                    if threshold_result['causal_strengths'].std() > 0.2:
                        mechanism = 'threshold'
                    elif granger_result['causality_score'] > 0.5:
                        mechanism = 'nonlinear'
                    else:
                        mechanism = 'linear'
                    
                    edge = CausalEdge(
                        source=i,
                        target=j,
                        strength=granger_result['causality_score'].item(),
                        confidence=granger_result['confidence'].item(),
                        lag=granger_result['optimal_lag'].item(),
                        mechanism=mechanism
                    )
                    causal_edges.append(edge)
        
        # 3. Apply theoretical constraints if enabled
        if self.use_theory_constraints and market_state:
            causal_adj = self._apply_theory_constraints(
                causal_adj, market_state, causal_edges
            )
        
        # 4. Refine causal graph with neural network
        refined_adj = self.graph_refiner(causal_adj.flatten()).view(n_assets, n_assets)
        
        # 5. Create edge features for GNN
        edge_index, edge_features = self._create_gnn_edges(causal_edges, n_assets)
        
        return {
            'causal_adjacency': refined_adj,
            'causal_edges': causal_edges,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'num_causal_edges': len(causal_edges),
            'avg_lag': np.mean([e.lag for e in causal_edges]) if causal_edges else 0
        }
    
    def _apply_theory_constraints(
        self,
        causal_adj: torch.Tensor,
        market_state: Dict[str, float],
        edges: List[CausalEdge]
    ) -> torch.Tensor:
        """Apply theoretical constraints based on market regime"""
        # In crisis, increase connectivity
        if market_state.get('crisis_probability', 0) > 0.5:
            # Reduce threshold for edge inclusion
            causal_adj = torch.where(causal_adj > 0.05, causal_adj * 1.5, causal_adj)
            
            # Add edges between highly volatile assets
            volatility_threshold = market_state.get('volatility', 0.3)
            if volatility_threshold > 0.3:
                # Encourage more connections
                causal_adj = causal_adj + 0.1
        
        # Ensure no self-loops
        causal_adj.fill_diagonal_(0)
        
        # Clip values
        causal_adj = torch.clamp(causal_adj, 0, 1)
        
        return causal_adj
    
    def _create_gnn_edges(
        self,
        causal_edges: List[CausalEdge],
        num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert causal edges to GNN format"""
        if not causal_edges:
            # Return empty tensors
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 8)
        
        edge_index = []
        edge_features = []
        
        for edge in causal_edges:
            edge_index.append([edge.source, edge.target])
            
            # Encode edge features
            mechanism_encoding = {
                'linear': 0,
                'nonlinear': 1,
                'threshold': 2
            }
            
            features = torch.tensor([
                edge.strength,
                edge.confidence,
                edge.lag / self.max_lag,  # Normalize
                mechanism_encoding.get(edge.mechanism, 0) / 2.0  # Normalize
            ])
            
            # Expand to full edge feature dimension
            encoded = self.edge_encoder(features)
            edge_features.append(encoded)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_features = torch.stack(edge_features)
        
        return edge_index, edge_features
    
    def forward(
        self,
        graph_sequence: List[Data],
        window_size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Process a sequence of graphs to discover evolving causal relationships.
        """
        results = []
        
        for i, graph in enumerate(graph_sequence):
            # Extract time window of features
            start_idx = max(0, i - window_size + 1)
            
            # Get historical features for causal discovery
            historical_features = []
            for j in range(start_idx, i + 1):
                historical_features.append(graph_sequence[j].x)
            
            if len(historical_features) < 3:  # Need minimum history
                # Return default empty causal graph
                results.append({
                    'causal_adjacency': torch.zeros(graph.num_nodes, graph.num_nodes),
                    'causal_edges': [],
                    'edge_index': graph.edge_index,
                    'edge_features': torch.zeros(graph.edge_index.shape[1], 8)
                })
                continue
            
            # Stack historical features
            time_series = torch.stack(historical_features)  # (time, nodes, features)
            
            # Discover causal graph
            causal_result = self.discover_causal_graph(time_series)
            
            # Update graph with causal edges
            graph.edge_index = causal_result['edge_index']
            graph.edge_attr = causal_result['edge_features']
            
            results.append(causal_result)
        
        return results


class CausalContagionPredictor(nn.Module):
    """
    Uses discovered causal relationships to predict contagion paths.
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        self.path_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def predict_contagion_path(
        self,
        causal_graph: torch.Tensor,
        shock_nodes: List[int],
        node_features: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Predict how shocks will propagate through causal network.
        
        Returns:
            - Contagion probabilities for each node
            - Most likely propagation paths
            - Expected time to reach each node
        """
        n_nodes = causal_graph.shape[0]
        device = causal_graph.device
        
        # Initialize contagion probabilities
        contagion_probs = torch.zeros(n_nodes, device=device)
        contagion_probs[shock_nodes] = 1.0
        
        # Track propagation paths
        paths = {node: [[node]] for node in shock_nodes}
        arrival_times = torch.full((n_nodes,), float('inf'), device=device)
        arrival_times[shock_nodes] = 0
        
        # Iterative propagation
        max_steps = 10
        for step in range(max_steps):
            new_contagion_probs = contagion_probs.clone()
            
            for i in range(n_nodes):
                if contagion_probs[i] > 0:
                    # Find causal children
                    children = torch.where(causal_graph[i] > 0)[0]
                    
                    for j in children:
                        # Compute transmission probability
                        edge_strength = causal_graph[i, j]
                        
                        # Use node features to modulate transmission
                        source_features = node_features[i]
                        target_features = node_features[j]
                        edge_features = torch.tensor([
                            edge_strength,
                            contagion_probs[i],
                            step / max_steps,
                            torch.abs(source_features[0] - target_features[0])  # Return difference
                        ], device=device)
                        
                        combined = torch.cat([
                            source_features,
                            target_features,
                            edge_features
                        ])
                        
                        transmission_prob = self.path_scorer(combined).squeeze()
                        
                        # Update contagion probability
                        new_prob = contagion_probs[i] * transmission_prob * edge_strength
                        if new_prob > new_contagion_probs[j]:
                            new_contagion_probs[j] = new_prob
                            arrival_times[j] = min(arrival_times[j], step + 1)
                            
                            # Update paths
                            if j.item() not in paths:
                                paths[j.item()] = []
                            for path in paths.get(i.item(), []):
                                paths[j.item()].append(path + [j.item()])
            
            contagion_probs = new_contagion_probs
            
            # Early stopping if converged
            if torch.allclose(contagion_probs, new_contagion_probs, atol=1e-4):
                break
        
        # Identify critical transmission nodes (high betweenness in causal graph)
        G = nx.DiGraph()
        for i in range(n_nodes):
            for j in range(n_nodes):
                if causal_graph[i, j] > 0:
                    G.add_edge(i, j, weight=causal_graph[i, j].item())
        
        if G.number_of_nodes() > 0:
            betweenness = nx.betweenness_centrality(G, weight='weight')
            critical_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            critical_nodes = []
        
        return {
            'contagion_probabilities': contagion_probs,
            'arrival_times': arrival_times,
            'propagation_paths': paths,
            'critical_nodes': critical_nodes,
            'num_affected': torch.sum(contagion_probs > 0.1).item()
        }


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing Causal Discovery Module...")
    
    # Generate synthetic data
    n_time = 100
    n_assets = 10
    n_features = 5
    
    # Create synthetic time series with known causal structure
    time_series = torch.randn(n_time, n_assets, n_features)
    
    # Add some causal relationships
    for t in range(1, n_time):
        # Asset 0 causes asset 1 with lag 1
        time_series[t, 1, 0] += 0.7 * time_series[t-1, 0, 0] + 0.3 * torch.randn(1)
        
        # Asset 1 causes asset 2 with threshold effect
        if time_series[t-1, 1, 0] > 0.5:
            time_series[t, 2, 0] += 0.8 * time_series[t-1, 1, 0]
    
    # Test causal discovery
    causal_module = CausalDiscoveryModule(
        num_features=n_assets,
        max_lag=5
    )
    
    result = causal_module.discover_causal_graph(time_series)
    
    logger.info(f"Discovered {result['num_causal_edges']} causal edges")
    logger.info(f"Average lag: {result['avg_lag']:.2f}")
    
    # Test contagion prediction
    contagion_predictor = CausalContagionPredictor()
    node_features = time_series[-1, :, :]  # Latest features
    
    contagion_result = contagion_predictor.predict_contagion_path(
        result['causal_adjacency'],
        shock_nodes=[0],  # Shock to asset 0
        node_features=node_features
    )
    
    logger.info(f"Contagion affects {contagion_result['num_affected']} assets")
    logger.info(f"Critical nodes: {contagion_result['critical_nodes'][:3]}")