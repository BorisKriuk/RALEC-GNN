#!/usr/bin/env python3
"""
RALEC-GNN Phase 2: Theoretical Framework
Financial Network Morphology Theory - Mathematical Formalization

This module provides the theoretical foundation for understanding how financial
networks restructure under stress, including phase transitions, regime boundaries,
and the mathematical principles governing dynamic edge formation.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy, wasserstein_distance
from scipy.special import logsumexp
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import networkx as nx
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketPhaseSpace:
    """
    Defines the phase space for financial markets with three primary dimensions:
    1. Volatility (σ): Market uncertainty
    2. Correlation (ρ): Systemic coupling
    3. Liquidity (λ): Market depth
    """
    volatility: float  # σ ∈ [0, ∞)
    correlation: float  # ρ ∈ [-1, 1]
    liquidity: float   # λ ∈ [0, 1]
    
    def distance_to(self, other: 'MarketPhaseSpace') -> float:
        """Compute distance in phase space using Mahalanobis metric"""
        diff = np.array([
            self.volatility - other.volatility,
            self.correlation - other.correlation,
            self.liquidity - other.liquidity
        ])
        # Normalize by typical scales
        scales = np.array([0.3, 0.5, 0.3])  # Typical stddevs
        return np.sqrt(np.sum((diff / scales) ** 2))


class FinancialNetworkMorphology:
    """
    Core theoretical framework for financial network morphology.
    
    Key Principles:
    1. Networks are dynamic entities that restructure based on market conditions
    2. Phase transitions occur at critical points in the (σ, ρ, λ) space
    3. Edge formation follows preferential attachment during stress
    4. Information flow increases non-linearly with market stress
    """
    
    def __init__(self):
        self.regime_boundaries = self._define_regime_boundaries()
        self.phase_transition_kernel = self._initialize_transition_kernel()
        
    def _define_regime_boundaries(self) -> Dict[str, Any]:
        """
        Define theoretical boundaries between market regimes.
        
        Based on empirical observations and theoretical constraints:
        - Bull/Low Vol: σ < 0.15, ρ < 0.3, λ > 0.7
        - Normal/Bear: 0.15 ≤ σ < 0.3, 0.3 ≤ ρ < 0.6, 0.4 < λ ≤ 0.7
        - Crisis: σ ≥ 0.3, ρ ≥ 0.6, λ ≤ 0.4
        """
        return {
            'bull_boundary': {
                'volatility_threshold': 0.15,
                'correlation_threshold': 0.3,
                'liquidity_threshold': 0.7
            },
            'crisis_boundary': {
                'volatility_threshold': 0.3,
                'correlation_threshold': 0.6,
                'liquidity_threshold': 0.4
            },
            'transition_zones': {
                'volatility_buffer': 0.05,
                'correlation_buffer': 0.1,
                'liquidity_buffer': 0.1
            }
        }
    
    def _initialize_transition_kernel(self) -> nn.Module:
        """Initialize the phase transition probability kernel"""
        return PhaseTransitionKernel()
    
    def compute_regime_probability(self, state: MarketPhaseSpace) -> np.ndarray:
        """
        Compute probability distribution over regimes given current state.
        
        Uses soft boundaries with Gaussian mixture model.
        """
        # Define regime centers
        regime_centers = [
            MarketPhaseSpace(0.1, 0.2, 0.8),   # Bull
            MarketPhaseSpace(0.2, 0.4, 0.6),   # Normal
            MarketPhaseSpace(0.4, 0.7, 0.3)    # Crisis
        ]
        
        # Compute distances to each regime center
        distances = np.array([state.distance_to(center) for center in regime_centers])
        
        # Convert to probabilities using softmax with temperature
        temperature = 0.5
        log_probs = -distances / temperature
        probs = np.exp(log_probs - logsumexp(log_probs))
        
        return probs
    
    def edge_formation_probability(
        self,
        node_i_state: np.ndarray,
        node_j_state: np.ndarray,
        market_state: MarketPhaseSpace
    ) -> float:
        """
        Theoretical probability of edge formation between two nodes.
        
        Key factors:
        1. State similarity (homophily)
        2. Market stress level (contagion amplification)
        3. Information asymmetry
        
        P(edge) = σ(α·similarity + β·stress + γ·asymmetry)
        """
        # State similarity (correlation between node features)
        similarity = np.corrcoef(node_i_state, node_j_state)[0, 1]
        
        # Market stress indicator
        stress = (market_state.volatility * market_state.correlation) / (market_state.liquidity + 0.1)
        
        # Information asymmetry (difference in volatility)
        asymmetry = np.abs(np.std(node_i_state) - np.std(node_j_state))
        
        # Regime-dependent parameters
        regime_probs = self.compute_regime_probability(market_state)
        
        # Parameters: [bull, normal, crisis]
        alpha_values = np.array([0.5, 1.0, 0.3])  # Similarity importance
        beta_values = np.array([0.1, 0.5, 2.0])   # Stress importance
        gamma_values = np.array([0.2, 0.3, 0.8])  # Asymmetry importance
        
        alpha = np.dot(regime_probs, alpha_values)
        beta = np.dot(regime_probs, beta_values)
        gamma = np.dot(regime_probs, gamma_values)
        
        # Compute edge probability
        logit = alpha * similarity + beta * stress - gamma * asymmetry
        probability = 1 / (1 + np.exp(-logit))
        
        return probability
    
    def network_entropy(self, adjacency_matrix: np.ndarray) -> float:
        """
        Compute network entropy as a measure of uncertainty in information flow.
        
        H(G) = -Σ p_ij log(p_ij)
        where p_ij is the normalized edge weight
        """
        # Normalize adjacency matrix
        total_weight = adjacency_matrix.sum()
        if total_weight == 0:
            return 0.0
        
        prob_matrix = adjacency_matrix / total_weight
        
        # Compute entropy
        prob_flat = prob_matrix.flatten()
        prob_flat = prob_flat[prob_flat > 0]  # Remove zeros
        
        return entropy(prob_flat)
    
    def percolation_threshold(self, num_nodes: int, avg_degree: float) -> float:
        """
        Compute the theoretical percolation threshold for crisis propagation.
        
        Based on random graph theory:
        p_c = 1 / (avg_degree - 1)
        """
        if avg_degree <= 1:
            return 1.0  # No percolation possible
        
        return 1.0 / (avg_degree - 1)
    
    def contagion_velocity(
        self,
        adjacency_matrix: np.ndarray,
        initial_shock: np.ndarray
    ) -> float:
        """
        Compute theoretical contagion propagation velocity.
        
        Based on spectral radius of the adjacency matrix:
        v = λ_max(A) * ||shock||
        """
        if adjacency_matrix.size == 0:
            return 0.0
        
        # Compute largest eigenvalue (spectral radius)
        eigenvalues = np.linalg.eigvals(adjacency_matrix)
        spectral_radius = np.max(np.abs(eigenvalues))
        
        # Shock magnitude
        shock_magnitude = np.linalg.norm(initial_shock)
        
        return spectral_radius * shock_magnitude


class PhaseTransitionKernel(nn.Module):
    """
    Neural network kernel for learning phase transition dynamics.
    
    Maps from current state to transition probabilities.
    """
    
    def __init__(self, state_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        
        self.transition_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 9)  # 3x3 transition matrix
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute transition probability matrix given current state.
        
        Returns:
            3x3 transition probability matrix
        """
        logits = self.transition_net(state)
        transition_matrix = logits.view(3, 3)
        
        # Apply softmax to ensure valid probabilities
        transition_probs = torch.softmax(transition_matrix, dim=1)
        
        return transition_probs


class IsingModelRegimeDetector:
    """
    Ising model-inspired regime detection.
    
    Markets as spin systems:
    - Spin up = Risk-on (bullish)
    - Spin down = Risk-off (bearish)
    - Temperature = Market volatility
    """
    
    def __init__(self, num_assets: int):
        self.num_assets = num_assets
        self.interaction_matrix = self._initialize_interactions()
        
    def _initialize_interactions(self) -> np.ndarray:
        """Initialize pairwise interaction strengths"""
        # Start with small random interactions
        J = np.random.randn(self.num_assets, self.num_assets) * 0.1
        # Make symmetric
        J = (J + J.T) / 2
        # Zero diagonal
        np.fill_diagonal(J, 0)
        return J
    
    def compute_energy(self, spins: np.ndarray) -> float:
        """
        Compute system energy: E = -Σ J_ij s_i s_j
        
        Lower energy = more stable configuration
        """
        return -0.5 * np.sum(self.interaction_matrix * np.outer(spins, spins))
    
    def compute_magnetization(self, spins: np.ndarray) -> float:
        """
        Compute system magnetization: M = (1/N) Σ s_i
        
        Indicates overall market sentiment
        """
        return np.mean(spins)
    
    def detect_phase_transition(
        self,
        spin_history: np.ndarray,
        window: int = 20
    ) -> Dict[str, float]:
        """
        Detect phase transitions using order parameter fluctuations.
        
        High susceptibility indicates proximity to critical point.
        """
        if len(spin_history) < window:
            return {'transition_probability': 0.0, 'susceptibility': 0.0}
        
        # Compute magnetization time series
        magnetizations = np.array([
            self.compute_magnetization(spins) for spins in spin_history[-window:]
        ])
        
        # Susceptibility = variance of order parameter
        susceptibility = np.var(magnetizations)
        
        # Transition probability based on susceptibility
        # High susceptibility indicates critical point
        transition_prob = 1 - np.exp(-susceptibility * 10)
        
        return {
            'transition_probability': transition_prob,
            'susceptibility': susceptibility,
            'magnetization': magnetizations[-1],
            'energy': self.compute_energy(spin_history[-1])
        }


class PercolationTheoryAnalyzer:
    """
    Apply percolation theory to understand systemic risk propagation.
    
    Key concepts:
    - Nodes = Financial institutions/assets
    - Edges = Risk channels
    - Percolation = Systemic crisis
    """
    
    def __init__(self):
        self.critical_fraction = None
        
    def find_giant_component(self, adjacency_matrix: np.ndarray) -> Tuple[int, List[int]]:
        """Find the size of the largest connected component"""
        G = nx.from_numpy_array(adjacency_matrix)
        
        if len(G) == 0:
            return 0, []
        
        components = list(nx.connected_components(G))
        if not components:
            return 0, []
        
        largest = max(components, key=len)
        return len(largest), list(largest)
    
    def compute_percolation_probability(
        self,
        adjacency_matrix: np.ndarray,
        shock_nodes: List[int]
    ) -> float:
        """
        Compute probability that shock will percolate through network.
        
        Based on:
        1. Network connectivity
        2. Location of initial shocks
        3. Edge weights (contagion strength)
        """
        n = adjacency_matrix.shape[0]
        if n == 0:
            return 0.0
        
        # Check if shocked nodes are in giant component
        giant_size, giant_nodes = self.find_giant_component(adjacency_matrix)
        
        shocked_in_giant = sum(1 for node in shock_nodes if node in giant_nodes)
        
        # Percolation probability factors
        connectivity_factor = giant_size / n
        shock_centrality = shocked_in_giant / len(shock_nodes) if shock_nodes else 0
        
        # Average edge weight (contagion strength)
        avg_weight = adjacency_matrix[adjacency_matrix > 0].mean() if adjacency_matrix.any() else 0
        
        # Combined probability
        perc_prob = connectivity_factor * shock_centrality * avg_weight
        
        return min(perc_prob, 1.0)
    
    def identify_critical_nodes(
        self,
        adjacency_matrix: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Identify nodes whose removal would most fragment the network.
        
        Uses betweenness centrality as criticality measure.
        """
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Compute betweenness centrality
        centrality = nx.betweenness_centrality(G, weight='weight')
        
        # Sort by centrality
        critical_nodes = sorted(
            centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return critical_nodes


class TheoreticalMetrics:
    """
    Compute theory-based metrics for model evaluation and interpretation.
    """
    
    def __init__(self, morphology: FinancialNetworkMorphology):
        self.morphology = morphology
        self.ising_detector = None
        self.percolation_analyzer = PercolationTheoryAnalyzer()
        
    def compute_theoretical_metrics(
        self,
        adjacency_matrix: np.ndarray,
        node_features: np.ndarray,
        market_state: MarketPhaseSpace
    ) -> Dict[str, float]:
        """Compute all theoretical metrics"""
        
        metrics = {}
        
        # Network morphology metrics
        metrics['network_entropy'] = self.morphology.network_entropy(adjacency_matrix)
        metrics['avg_degree'] = np.sum(adjacency_matrix > 0) / adjacency_matrix.shape[0]
        metrics['percolation_threshold'] = self.morphology.percolation_threshold(
            adjacency_matrix.shape[0],
            metrics['avg_degree']
        )
        
        # Phase space metrics
        regime_probs = self.morphology.compute_regime_probability(market_state)
        metrics['bull_probability'] = regime_probs[0]
        metrics['normal_probability'] = regime_probs[1]
        metrics['crisis_probability'] = regime_probs[2]
        
        # Percolation metrics
        giant_size, _ = self.percolation_analyzer.find_giant_component(adjacency_matrix)
        metrics['giant_component_fraction'] = giant_size / adjacency_matrix.shape[0]
        
        # Contagion potential
        shock = np.ones(adjacency_matrix.shape[0]) / adjacency_matrix.shape[0]
        metrics['contagion_velocity'] = self.morphology.contagion_velocity(
            adjacency_matrix, shock
        )
        
        return metrics
    
    def theoretical_loss(
        self,
        predicted_adjacency: torch.Tensor,
        true_adjacency: torch.Tensor,
        market_state: MarketPhaseSpace
    ) -> torch.Tensor:
        """
        Theory-informed loss function that encourages:
        1. Correct regime-dependent edge patterns
        2. Appropriate network density
        3. Realistic clustering coefficients
        """
        # Standard reconstruction loss
        reconstruction_loss = nn.functional.mse_loss(predicted_adjacency, true_adjacency)
        
        # Density regularization (regime-dependent)
        regime_probs = self.morphology.compute_regime_probability(market_state)
        expected_densities = np.array([0.2, 0.3, 0.5])  # Bull, Normal, Crisis
        expected_density = np.dot(regime_probs, expected_densities)
        
        predicted_density = torch.mean(predicted_adjacency)
        density_loss = (predicted_density - expected_density) ** 2
        
        # Combine losses
        total_loss = reconstruction_loss + 0.1 * density_loss
        
        return total_loss


def create_theoretical_framework() -> Dict[str, Any]:
    """
    Create and return the complete theoretical framework.
    """
    
    framework = {
        'morphology': FinancialNetworkMorphology(),
        'ising_model': IsingModelRegimeDetector,  # Class, not instance
        'percolation': PercolationTheoryAnalyzer(),
        'metrics': None  # Will be initialized with morphology
    }
    
    framework['metrics'] = TheoreticalMetrics(framework['morphology'])
    
    logger.info("Theoretical framework initialized successfully")
    logger.info("Components:")
    logger.info("  - Financial Network Morphology Theory")
    logger.info("  - Ising Model Regime Detection")
    logger.info("  - Percolation Theory Analysis")
    logger.info("  - Theory-based Metrics")
    
    return framework


if __name__ == "__main__":
    # Example usage and validation
    framework = create_theoretical_framework()
    
    # Test with synthetic data
    market_state = MarketPhaseSpace(
        volatility=0.25,
        correlation=0.45,
        liquidity=0.6
    )
    
    regime_probs = framework['morphology'].compute_regime_probability(market_state)
    print(f"\nMarket State: {market_state}")
    print(f"Regime Probabilities:")
    print(f"  Bull: {regime_probs[0]:.2%}")
    print(f"  Normal: {regime_probs[1]:.2%}")
    print(f"  Crisis: {regime_probs[2]:.2%}")
    
    # Test edge formation
    node_i = np.random.randn(10)
    node_j = np.random.randn(10)
    edge_prob = framework['morphology'].edge_formation_probability(
        node_i, node_j, market_state
    )
    print(f"\nEdge Formation Probability: {edge_prob:.2%}")