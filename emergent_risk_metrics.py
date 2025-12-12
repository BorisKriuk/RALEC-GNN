#!/usr/bin/env python3
"""
RALEC-GNN Phase 6: Emergent Risk Metrics and Systemic Risk Indicators
Implements advanced metrics that capture emergent systemic risks arising from
network effects, collective behavior, and complex interactions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import networkx as nx
from scipy import stats
from scipy.linalg import eigh
from collections import defaultdict
import logging
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SystemicRiskIndicators:
    """Container for various systemic risk metrics"""
    # Network-based metrics
    network_fragility: float  # How close network is to breakdown
    contagion_potential: float  # Expected contagion spread
    systemic_importance: torch.Tensor  # Per-asset systemic importance
    clustering_risk: float  # Risk from highly clustered regions
    
    # Collective behavior metrics
    herding_index: float  # Degree of collective movement
    synchronization_risk: float  # Risk from synchronized behavior
    diversity_loss: float  # Loss of strategy diversity
    
    # Information metrics
    information_contagion: float  # Speed of information spread
    uncertainty_propagation: float  # How uncertainty spreads
    
    # Complex system metrics
    emergence_indicator: float  # Likelihood of emergent phenomena
    self_organization: float  # Degree of self-organizing criticality
    cascade_probability: float  # Probability of cascading failures
    
    # Memory and adaptation metrics
    memory_fragility: float  # How past crises affect current risk
    adaptation_capacity: float  # System's ability to adapt
    
    # Composite scores
    overall_systemic_risk: float  # Aggregated systemic risk score
    risk_decomposition: Dict[str, float] = field(default_factory=dict)


class NetworkFragilityAnalyzer(nn.Module):
    """
    Analyzes network fragility using percolation theory,
    spectral analysis, and robustness measures.
    """
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        # Learned fragility scorer
        self.fragility_scorer = nn.Sequential(
            nn.Linear(10, hidden_dim),  # Multiple fragility features
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Component vulnerability predictor
        self.vulnerability_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def compute_network_fragility(self, graph: Data) -> Tuple[float, Dict[str, Any]]:
        """
        Comprehensive network fragility assessment.
        """
        # Convert to NetworkX for analysis
        edge_list = graph.edge_index.T.cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(graph.num_nodes))
        G.add_edges_from(edge_list)
        
        # 1. Percolation Analysis
        percolation_threshold = self._estimate_percolation_threshold(G)
        current_density = G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2)
        percolation_distance = current_density - percolation_threshold
        
        # 2. Spectral Analysis
        try:
            adjacency = nx.adjacency_matrix(G).todense()
            eigenvalues = np.linalg.eigvals(adjacency)
            spectral_gap = sorted(np.abs(eigenvalues))[-1] - sorted(np.abs(eigenvalues))[-2]
            spectral_radius = np.max(np.abs(eigenvalues))
        except:
            spectral_gap = 0
            spectral_radius = 1
        
        # 3. k-Core Analysis
        k_core_structure = self._analyze_k_cores(G)
        core_periphery_ratio = k_core_structure['max_core_size'] / G.number_of_nodes()
        
        # 4. Robustness Metrics
        node_connectivity = nx.node_connectivity(G) if G.number_of_nodes() > 1 else 0
        edge_connectivity = nx.edge_connectivity(G) if G.number_of_nodes() > 1 else 0
        
        # 5. Critical Nodes Identification
        betweenness = nx.betweenness_centrality(G)
        critical_fraction = sum(1 for v in betweenness.values() if v > np.mean(list(betweenness.values())) + 2 * np.std(list(betweenness.values()))) / len(betweenness)
        
        # 6. Small World Properties
        try:
            clustering = nx.average_clustering(G)
            if G.number_of_edges() > 0:
                avg_path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
            else:
                avg_path_length = float('inf')
            
            # Random graph comparison
            random_clustering = 2 * G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1))
            small_world_index = (clustering / random_clustering) if random_clustering > 0 else 0
        except:
            small_world_index = 0
            avg_path_length = float('inf')
        
        # Combine features for neural scoring
        fragility_features = torch.tensor([
            percolation_distance,
            1 / (spectral_gap + 1e-6),
            1 - core_periphery_ratio,
            1 / (node_connectivity + 1),
            1 / (edge_connectivity + 1),
            critical_fraction,
            small_world_index,
            min(avg_path_length / G.number_of_nodes(), 1),
            1 / (spectral_radius + 1e-6),
            k_core_structure['fragmentation']
        ], dtype=torch.float32).unsqueeze(0)
        
        # Neural fragility score
        fragility_score = self.fragility_scorer(fragility_features).item()
        
        details = {
            'percolation_threshold': percolation_threshold,
            'percolation_distance': percolation_distance,
            'spectral_gap': spectral_gap,
            'spectral_radius': spectral_radius,
            'k_core_analysis': k_core_structure,
            'node_connectivity': node_connectivity,
            'edge_connectivity': edge_connectivity,
            'critical_nodes_fraction': critical_fraction,
            'small_world_index': small_world_index
        }
        
        return fragility_score, details
    
    def _estimate_percolation_threshold(self, G: nx.Graph) -> float:
        """Estimate network percolation threshold"""
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        if n <= 1:
            return 0.0
        
        # Mean degree
        mean_degree = 2 * m / n
        
        # For random networks, threshold ≈ 1/<k>
        # Adjust for network structure
        clustering = nx.average_clustering(G)
        
        # Higher clustering increases robustness
        threshold = 1 / (mean_degree + 1) * (1 + clustering)
        
        return min(threshold, 1.0)
    
    def _analyze_k_cores(self, G: nx.Graph) -> Dict[str, Any]:
        """Analyze k-core decomposition for robustness"""
        if G.number_of_nodes() == 0:
            return {'max_k': 0, 'max_core_size': 0, 'fragmentation': 1.0}
        
        # Get core numbers
        core_numbers = nx.core_number(G)
        
        if not core_numbers:
            return {'max_k': 0, 'max_core_size': 0, 'fragmentation': 1.0}
        
        max_k = max(core_numbers.values())
        
        # Size of maximum k-core
        max_core_nodes = [n for n, k in core_numbers.items() if k == max_k]
        max_core_size = len(max_core_nodes)
        
        # Fragmentation: how distributed are the k-values
        k_values = list(core_numbers.values())
        fragmentation = stats.entropy(k_values) / np.log(len(k_values)) if len(k_values) > 1 else 0
        
        return {
            'max_k': max_k,
            'max_core_size': max_core_size,
            'fragmentation': fragmentation,
            'core_distribution': dict(zip(*np.unique(k_values, return_counts=True)))
        }
    
    def identify_vulnerable_components(self, graph: Data, node_features: torch.Tensor) -> torch.Tensor:
        """Identify vulnerable nodes/components in the network"""
        vulnerability_scores = self.vulnerability_net(node_features)
        return vulnerability_scores.squeeze()


class CollectiveBehaviorAnalyzer(nn.Module):
    """
    Analyzes collective behavior patterns that create systemic risk.
    """
    
    def __init__(self, num_assets: int, hidden_dim: int = 128):
        super().__init__()
        self.num_assets = num_assets
        
        # Herding detection network
        self.herding_detector = nn.Sequential(
            nn.Linear(num_assets * 2, hidden_dim),  # Returns and positions
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # No herding, moderate, severe
        )
        
        # Synchronization analyzer
        self.sync_analyzer = nn.Sequential(
            nn.Linear(num_assets + 10, hidden_dim),  # Asset features + sync metrics
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def analyze_collective_behavior(
        self, 
        returns_sequence: torch.Tensor,  # (time, assets)
        positions: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Comprehensive collective behavior analysis.
        """
        # 1. Herding Analysis
        herding_index = self._compute_herding_index(returns_sequence)
        
        # 2. Synchronization Analysis
        sync_metrics = self._analyze_synchronization(returns_sequence)
        
        # 3. Diversity Analysis
        diversity_loss = self._compute_diversity_loss(returns_sequence, positions)
        
        # 4. Correlation Dynamics
        correlation_surge = self._detect_correlation_surge(returns_sequence)
        
        # 5. Phase Coupling
        phase_coupling = self._analyze_phase_coupling(returns_sequence)
        
        # Neural herding classification
        if positions is not None:
            herding_features = torch.cat([
                returns_sequence[-1],
                positions
            ])
        else:
            herding_features = torch.cat([
                returns_sequence[-1],
                torch.zeros(self.num_assets)
            ])
        
        herding_class = self.herding_detector(herding_features.unsqueeze(0))
        herding_severity = F.softmax(herding_class, dim=1)[0, 2].item()  # Severe herding prob
        
        return {
            'herding_index': herding_index,
            'herding_severity': herding_severity,
            'synchronization_risk': sync_metrics['sync_risk'],
            'phase_locking': sync_metrics['phase_locking'],
            'diversity_loss': diversity_loss,
            'correlation_surge': correlation_surge,
            'phase_coupling': phase_coupling,
            'collective_risk': (herding_index + sync_metrics['sync_risk'] + diversity_loss) / 3
        }
    
    def _compute_herding_index(self, returns: torch.Tensor) -> float:
        """Compute herding behavior index"""
        if returns.shape[0] < 2:
            return 0.0
        
        # Cross-sectional dispersion over time
        cross_sectional_std = returns.std(dim=1)
        
        # Herding: decreasing dispersion
        if len(cross_sectional_std) > 1:
            herding_trend = -torch.diff(cross_sectional_std).mean().item()
            
            # Normalize
            avg_dispersion = cross_sectional_std.mean().item()
            herding_index = max(0, min(1, herding_trend / (avg_dispersion + 1e-6) + 0.5))
        else:
            herding_index = 0.0
        
        return herding_index
    
    def _analyze_synchronization(self, returns: torch.Tensor) -> Dict[str, float]:
        """Analyze synchronization patterns"""
        if returns.shape[0] < 10:
            return {'sync_risk': 0.0, 'phase_locking': 0.0}
        
        # Instantaneous correlation
        corr_matrix = torch.corrcoef(returns.T)
        
        # Average pairwise correlation (excluding diagonal)
        mask = ~torch.eye(corr_matrix.shape[0], dtype=bool)
        avg_correlation = corr_matrix[mask].mean().item()
        
        # Phase analysis using Hilbert transform approximation
        # Simplified: use rolling window correlation dynamics
        window = min(20, returns.shape[0] // 2)
        rolling_corrs = []
        
        for i in range(window, returns.shape[0]):
            window_corr = torch.corrcoef(returns[i-window:i].T)
            rolling_corrs.append(window_corr[mask].mean().item())
        
        if rolling_corrs:
            # Persistence of high correlation
            high_corr_persistence = sum(1 for c in rolling_corrs if c > 0.7) / len(rolling_corrs)
            phase_locking = high_corr_persistence
        else:
            phase_locking = 0.0
        
        # Eigenvalue analysis for synchronization
        try:
            eigenvalues = torch.linalg.eigvals(corr_matrix).real
            largest_eigenvalue = eigenvalues.max().item()
            sync_indicator = largest_eigenvalue / len(eigenvalues)
        except:
            sync_indicator = avg_correlation
        
        sync_risk = min(1.0, sync_indicator * avg_correlation)
        
        return {
            'sync_risk': sync_risk,
            'phase_locking': phase_locking,
            'avg_correlation': avg_correlation,
            'sync_indicator': sync_indicator
        }
    
    def _compute_diversity_loss(
        self, 
        returns: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> float:
        """Measure loss of diversity in strategies/positions"""
        # Return pattern diversity
        return_diversity = 1 - torch.corrcoef(returns.T).abs().mean().item()
        
        # Position diversity (if available)
        if positions is not None:
            position_concentration = (positions.abs() / positions.abs().sum()).max().item()
            position_diversity = 1 - position_concentration
        else:
            position_diversity = return_diversity
        
        # Combine metrics
        diversity_loss = 1 - (return_diversity * position_diversity) ** 0.5
        
        return diversity_loss
    
    def _detect_correlation_surge(self, returns: torch.Tensor) -> float:
        """Detect sudden increases in correlation"""
        if returns.shape[0] < 20:
            return 0.0
        
        # Compute rolling correlations
        window = 10
        correlations = []
        
        for i in range(window, returns.shape[0]):
            corr = torch.corrcoef(returns[i-window:i].T)
            mask = ~torch.eye(corr.shape[0], dtype=bool)
            avg_corr = corr[mask].mean().item()
            correlations.append(avg_corr)
        
        if len(correlations) > 10:
            recent_corr = np.mean(correlations[-5:])
            historical_corr = np.mean(correlations[:-5])
            
            if historical_corr > 0:
                surge = (recent_corr - historical_corr) / historical_corr
                return max(0, min(1, surge))
        
        return 0.0
    
    def _analyze_phase_coupling(self, returns: torch.Tensor) -> float:
        """Analyze phase coupling between assets"""
        if returns.shape[0] < 20:
            return 0.0
        
        # Simplified phase coupling using sign concordance
        signs = torch.sign(returns)
        
        # Pairwise sign concordance
        concordance_sum = 0
        count = 0
        
        for i in range(returns.shape[1]):
            for j in range(i + 1, returns.shape[1]):
                concordance = (signs[:, i] == signs[:, j]).float().mean().item()
                concordance_sum += concordance
                count += 1
        
        if count > 0:
            avg_concordance = concordance_sum / count
            # High concordance indicates phase coupling
            phase_coupling = (avg_concordance - 0.5) * 2  # Normalize to [0, 1]
            return max(0, min(1, phase_coupling))
        
        return 0.0


class InformationContagionAnalyzer(nn.Module):
    """
    Analyzes how information and uncertainty propagate through the system.
    """
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        # Information flow network
        self.info_flow_net = nn.Sequential(
            nn.Linear(256, hidden_dim),  # Edge features + node features
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty propagation model
        self.uncertainty_prop = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
    def analyze_information_contagion(
        self,
        graph: Data,
        node_uncertainties: torch.Tensor,
        edge_strengths: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Analyze how information and uncertainty spread.
        """
        # 1. Information Transfer Entropy
        transfer_entropy = self._compute_transfer_entropy(graph, node_uncertainties)
        
        # 2. Contagion Speed
        contagion_speed = self._estimate_contagion_speed(graph, edge_strengths)
        
        # 3. Uncertainty Amplification
        uncertainty_amp = self._compute_uncertainty_amplification(
            graph, node_uncertainties
        )
        
        # 4. Information Bottlenecks
        bottlenecks = self._identify_information_bottlenecks(graph)
        
        # 5. Echo Chamber Effects
        echo_effects = self._detect_echo_chambers(graph)
        
        return {
            'information_contagion': (transfer_entropy + contagion_speed) / 2,
            'uncertainty_propagation': uncertainty_amp,
            'contagion_speed': contagion_speed,
            'transfer_entropy': transfer_entropy,
            'bottleneck_risk': bottlenecks['risk'],
            'echo_chamber_effect': echo_effects
        }
    
    def _compute_transfer_entropy(
        self,
        graph: Data,
        uncertainties: torch.Tensor
    ) -> float:
        """Compute information transfer entropy"""
        # Simplified transfer entropy based on uncertainty flow
        edge_index = graph.edge_index
        
        if edge_index.shape[1] == 0:
            return 0.0
        
        # Source and target uncertainties
        source_uncertainty = uncertainties[edge_index[0]]
        target_uncertainty = uncertainties[edge_index[1]]
        
        # Information flow: high source uncertainty to low target
        info_flow = source_uncertainty - target_uncertainty
        positive_flow = (info_flow > 0).float().mean().item()
        
        # Entropy of flow distribution
        if len(info_flow) > 1:
            flow_entropy = stats.entropy(
                np.histogram(info_flow.cpu().numpy(), bins=10)[0] + 1e-10
            )
            normalized_entropy = flow_entropy / np.log(10)
        else:
            normalized_entropy = 0.0
        
        return positive_flow * normalized_entropy
    
    def _estimate_contagion_speed(
        self,
        graph: Data,
        edge_strengths: Optional[torch.Tensor] = None
    ) -> float:
        """Estimate speed of contagion spread"""
        # Convert to NetworkX
        edge_list = graph.edge_index.T.cpu().numpy()
        G = nx.Graph()
        G.add_edges_from(edge_list)
        
        if G.number_of_nodes() <= 1:
            return 0.0
        
        # Use effective resistance as proxy for contagion speed
        try:
            # Laplacian spectrum
            laplacian = nx.laplacian_matrix(G).todense()
            eigenvalues = np.linalg.eigvalsh(laplacian)
            
            # Algebraic connectivity (2nd smallest eigenvalue)
            algebraic_connectivity = sorted(eigenvalues)[1] if len(eigenvalues) > 1 else 0
            
            # Higher connectivity = faster spread
            speed_indicator = 1 - np.exp(-algebraic_connectivity)
        except:
            speed_indicator = 0.0
        
        # Adjust for edge strengths if provided
        if edge_strengths is not None:
            avg_strength = edge_strengths.mean().item()
            speed_indicator *= avg_strength
        
        return min(1.0, speed_indicator)
    
    def _compute_uncertainty_amplification(
        self,
        graph: Data,
        uncertainties: torch.Tensor
    ) -> float:
        """Compute how uncertainty amplifies through network"""
        if graph.edge_index.shape[1] == 0:
            return 0.0
        
        # Simulate uncertainty propagation
        current_uncertainty = uncertainties.clone()
        amplification_factor = 1.0
        
        # Simple diffusion model
        for _ in range(5):  # 5 propagation steps
            new_uncertainty = current_uncertainty.clone()
            
            for i in range(graph.num_nodes):
                # Get neighbors
                neighbors = graph.edge_index[1][graph.edge_index[0] == i]
                if len(neighbors) > 0:
                    neighbor_uncertainty = current_uncertainty[neighbors].mean()
                    # Uncertainty increases with neighbor average
                    new_uncertainty[i] = 0.7 * current_uncertainty[i] + 0.3 * neighbor_uncertainty
            
            # Measure amplification
            amplification_factor = new_uncertainty.mean() / (uncertainties.mean() + 1e-6)
            current_uncertainty = new_uncertainty
        
        return min(1.0, amplification_factor.item() - 1)
    
    def _identify_information_bottlenecks(self, graph: Data) -> Dict[str, Any]:
        """Identify information flow bottlenecks"""
        edge_list = graph.edge_index.T.cpu().numpy()
        G = nx.Graph()
        G.add_edges_from(edge_list)
        
        if G.number_of_nodes() <= 2:
            return {'risk': 0.0, 'bottleneck_nodes': []}
        
        # Betweenness centrality identifies bottlenecks
        betweenness = nx.betweenness_centrality(G)
        
        # High betweenness nodes are bottlenecks
        mean_between = np.mean(list(betweenness.values()))
        std_between = np.std(list(betweenness.values()))
        
        bottleneck_nodes = [
            n for n, b in betweenness.items() 
            if b > mean_between + 2 * std_between
        ]
        
        bottleneck_risk = len(bottleneck_nodes) / G.number_of_nodes()
        
        return {
            'risk': min(1.0, bottleneck_risk * 3),  # Scale up
            'bottleneck_nodes': bottleneck_nodes,
            'max_betweenness': max(betweenness.values()) if betweenness else 0
        }
    
    def _detect_echo_chambers(self, graph: Data) -> float:
        """Detect echo chamber effects in information flow"""
        edge_list = graph.edge_index.T.cpu().numpy()
        G = nx.Graph()
        G.add_edges_from(edge_list)
        
        if G.number_of_nodes() < 10:
            return 0.0
        
        # Detect communities
        try:
            from networkx.algorithms import community
            communities = list(community.greedy_modularity_communities(G))
            
            # Modularity as echo chamber indicator
            modularity = community.modularity(G, communities)
            
            # High modularity + low inter-community edges = echo chambers
            inter_community_edges = 0
            total_edges = G.number_of_edges()
            
            for i, comm1 in enumerate(communities):
                for j, comm2 in enumerate(communities[i+1:], i+1):
                    inter_community_edges += len([
                        (u, v) for u, v in G.edges() 
                        if (u in comm1 and v in comm2) or (u in comm2 and v in comm1)
                    ])
            
            if total_edges > 0:
                inter_ratio = inter_community_edges / total_edges
                echo_effect = modularity * (1 - inter_ratio)
            else:
                echo_effect = 0.0
        except:
            echo_effect = 0.0
        
        return min(1.0, echo_effect)


class EmergentRiskMetrics(nn.Module):
    """
    Main module that combines all emergent risk analyzers and produces
    comprehensive systemic risk metrics.
    """
    
    def __init__(
        self,
        num_assets: int,
        hidden_dim: int = 256,
        memory_enabled: bool = True
    ):
        super().__init__()
        
        # Component analyzers
        self.network_analyzer = NetworkFragilityAnalyzer(hidden_dim)
        self.behavior_analyzer = CollectiveBehaviorAnalyzer(num_assets, hidden_dim)
        self.information_analyzer = InformationContagionAnalyzer(hidden_dim)
        
        # Risk aggregation network
        self.risk_aggregator = nn.Sequential(
            nn.Linear(20, hidden_dim),  # Multiple risk inputs
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)  # Different risk categories
        )
        
        # Emergence detector
        self.emergence_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # Low, medium, high emergence
        )
        
        # Risk decomposer
        self.risk_decomposer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 10),  # Risk components
            nn.Softmax(dim=-1)
        )
        
        self.memory_enabled = memory_enabled
        if memory_enabled:
            # Risk memory for tracking evolution
            self.risk_memory = nn.LSTM(
                input_size=15,
                hidden_size=hidden_dim // 2,
                num_layers=2,
                batch_first=True
            )
            self.hidden_state = None
        
    def compute_systemic_risk(
        self,
        graph_sequence: List[Data],
        returns_sequence: Optional[torch.Tensor] = None,
        crisis_memory: Optional[Any] = None,
        phase_indicators: Optional[Dict[str, float]] = None
    ) -> SystemicRiskIndicators:
        """
        Compute comprehensive systemic risk indicators.
        """
        current_graph = graph_sequence[-1]
        
        # 1. Network Fragility Analysis
        network_fragility, network_details = self.network_analyzer.compute_network_fragility(
            current_graph
        )
        
        # Compute systemic importance
        if hasattr(current_graph, 'x'):
            systemic_importance = self.network_analyzer.identify_vulnerable_components(
                current_graph, current_graph.x
            )
        else:
            systemic_importance = torch.zeros(current_graph.num_nodes)
        
        # 2. Collective Behavior Analysis
        if returns_sequence is not None:
            behavior_metrics = self.behavior_analyzer.analyze_collective_behavior(
                returns_sequence
            )
        else:
            # Use node features as proxy
            if hasattr(current_graph, 'x') and current_graph.x.shape[0] > 1:
                pseudo_returns = torch.diff(current_graph.x, dim=0)
                behavior_metrics = self.behavior_analyzer.analyze_collective_behavior(
                    pseudo_returns
                )
            else:
                behavior_metrics = {
                    'herding_index': 0.0,
                    'synchronization_risk': 0.0,
                    'diversity_loss': 0.0
                }
        
        # 3. Information Contagion Analysis
        if hasattr(current_graph, 'x'):
            # Use feature variance as uncertainty proxy
            uncertainties = current_graph.x.var(dim=1)
        else:
            uncertainties = torch.ones(current_graph.num_nodes) * 0.5
        
        info_metrics = self.information_analyzer.analyze_information_contagion(
            current_graph, uncertainties
        )
        
        # 4. Memory-based Risk Assessment
        if crisis_memory and self.memory_enabled:
            memory_metrics = self._assess_memory_based_risk(crisis_memory)
        else:
            memory_metrics = {
                'memory_fragility': 0.0,
                'adaptation_capacity': 1.0
            }
        
        # 5. Emergence and Self-Organization
        emergence_metrics = self._assess_emergence(
            current_graph, network_details, behavior_metrics
        )
        
        # 6. Cascade Probability
        cascade_prob = self._estimate_cascade_probability(
            network_fragility,
            behavior_metrics['herding_index'],
            info_metrics['contagion_speed']
        )
        
        # 7. Aggregate Risks
        risk_vector = torch.tensor([
            network_fragility,
            network_details['critical_nodes_fraction'],
            behavior_metrics['herding_index'],
            behavior_metrics['synchronization_risk'],
            behavior_metrics['diversity_loss'],
            info_metrics['information_contagion'],
            info_metrics['uncertainty_propagation'],
            info_metrics['echo_chamber_effect'],
            emergence_metrics['emergence_indicator'],
            emergence_metrics['self_organization'],
            cascade_prob,
            memory_metrics['memory_fragility'],
            memory_metrics['adaptation_capacity'],
            network_details['percolation_distance'],
            info_metrics['bottleneck_risk']
        ]).unsqueeze(0)
        
        # Neural risk aggregation
        aggregated_risks = self.risk_aggregator(risk_vector[:, :20])
        risk_categories = F.softmax(aggregated_risks, dim=-1).squeeze()
        
        # Overall risk score
        overall_risk = self._compute_overall_risk(risk_vector.squeeze())
        
        # Risk decomposition
        risk_components = self.risk_decomposer(
            torch.cat([risk_vector, aggregated_risks], dim=-1)
        ).squeeze()
        
        risk_decomposition = {
            'network': risk_components[0].item(),
            'behavioral': risk_components[1].item(),
            'informational': risk_components[2].item(),
            'emergent': risk_components[3].item(),
            'memory': risk_components[4].item(),
            'cascade': risk_components[5].item(),
            'structural': risk_components[6].item(),
            'dynamic': risk_components[7].item(),
            'adaptive': risk_components[8].item(),
            'systemic': risk_components[9].item()
        }
        
        # Update risk memory
        if self.memory_enabled:
            self._update_risk_memory(risk_vector)
        
        return SystemicRiskIndicators(
            network_fragility=network_fragility,
            contagion_potential=network_details['percolation_distance'],
            systemic_importance=systemic_importance,
            clustering_risk=network_details['k_core_analysis']['fragmentation'],
            herding_index=behavior_metrics['herding_index'],
            synchronization_risk=behavior_metrics['synchronization_risk'],
            diversity_loss=behavior_metrics['diversity_loss'],
            information_contagion=info_metrics['information_contagion'],
            uncertainty_propagation=info_metrics['uncertainty_propagation'],
            emergence_indicator=emergence_metrics['emergence_indicator'],
            self_organization=emergence_metrics['self_organization'],
            cascade_probability=cascade_prob,
            memory_fragility=memory_metrics['memory_fragility'],
            adaptation_capacity=memory_metrics['adaptation_capacity'],
            overall_systemic_risk=overall_risk,
            risk_decomposition=risk_decomposition
        )
    
    def _assess_memory_based_risk(self, crisis_memory: Any) -> Dict[str, float]:
        """Assess risk based on crisis memory"""
        # Memory fragility: how similar current state to past crises
        if hasattr(crisis_memory, 'crisis_probability'):
            memory_fragility = crisis_memory.crisis_probability
        else:
            memory_fragility = 0.0
        
        # Adaptation capacity: inverse of memory rigidity
        if hasattr(crisis_memory, 'adaptation_success_rate'):
            adaptation_capacity = np.mean(crisis_memory.adaptation_success_rate) \
                if crisis_memory.adaptation_success_rate else 0.5
        else:
            adaptation_capacity = 0.5
        
        return {
            'memory_fragility': float(memory_fragility),
            'adaptation_capacity': float(adaptation_capacity)
        }
    
    def _assess_emergence(
        self,
        graph: Data,
        network_details: Dict[str, Any],
        behavior_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Assess emergence and self-organization"""
        # Emergence indicators
        emergence_factors = [
            network_details['small_world_index'] > 2,  # Small world property
            behavior_metrics['synchronization_risk'] > 0.7,  # High sync
            network_details['spectral_gap'] < 0.1,  # Low spectral gap
            behavior_metrics['phase_coupling'] > 0.6  # Phase coupling
        ]
        
        emergence_score = sum(emergence_factors) / len(emergence_factors)
        
        # Self-organization: spontaneous order
        if 'k_core_analysis' in network_details:
            core_structure = network_details['k_core_analysis']['max_k'] > 5
        else:
            core_structure = False
        
        self_org_factors = [
            core_structure,  # Strong core structure
            behavior_metrics.get('herding_severity', 0) > 0.5,  # Spontaneous herding
            network_details.get('clustering', 0) > 0.5  # High clustering
        ]
        
        self_org_score = sum(self_org_factors) / len(self_org_factors)
        
        return {
            'emergence_indicator': emergence_score,
            'self_organization': self_org_score
        }
    
    def _estimate_cascade_probability(
        self,
        network_fragility: float,
        herding_index: float,
        contagion_speed: float
    ) -> float:
        """Estimate probability of cascading failures"""
        # Simple multiplicative model
        cascade_factors = [
            network_fragility,
            herding_index,
            contagion_speed
        ]
        
        # Cascade requires all factors
        cascade_prob = np.prod(cascade_factors) ** (1/3)
        
        # Threshold effect
        if cascade_prob > 0.5:
            cascade_prob = cascade_prob ** 0.5  # Accelerate past threshold
        
        return min(1.0, cascade_prob)
    
    def _compute_overall_risk(self, risk_vector: torch.Tensor) -> float:
        """Compute overall systemic risk score"""
        # Weighted average with emphasis on critical metrics
        weights = torch.tensor([
            2.0,  # network_fragility
            1.5,  # critical_nodes
            1.5,  # herding
            2.0,  # synchronization
            1.0,  # diversity_loss
            1.5,  # info_contagion
            1.0,  # uncertainty_prop
            0.5,  # echo_chamber
            2.0,  # emergence
            1.5,  # self_organization
            2.5,  # cascade_prob
            1.0,  # memory_fragility
            0.5,  # adaptation_capacity (inverse)
            1.5,  # percolation_distance
            1.0   # bottleneck_risk
        ])
        
        # Invert adaptation capacity
        adjusted_vector = risk_vector.clone()
        adjusted_vector[12] = 1 - adjusted_vector[12]
        
        # Weighted mean
        overall = (adjusted_vector * weights).sum() / weights.sum()
        
        # Non-linearity for extreme risks
        if overall > 0.7:
            overall = overall ** 0.8  # Amplify high risk
        
        return min(1.0, overall.item())
    
    def _update_risk_memory(self, risk_vector: torch.Tensor):
        """Update LSTM memory with risk evolution"""
        if self.memory_enabled:
            risk_input = risk_vector[:15].unsqueeze(0).unsqueeze(0)
            
            if self.hidden_state is None:
                output, self.hidden_state = self.risk_memory(risk_input)
            else:
                output, self.hidden_state = self.risk_memory(
                    risk_input, self.hidden_state
                )


def create_risk_alert_system(
    risk_indicators: SystemicRiskIndicators,
    thresholds: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Generate risk alerts based on indicators.
    """
    if thresholds is None:
        thresholds = {
            'overall': 0.7,
            'network': 0.8,
            'cascade': 0.6,
            'emergence': 0.75,
            'synchronization': 0.8
        }
    
    alerts = []
    
    # Overall systemic risk alert
    if risk_indicators.overall_systemic_risk > thresholds['overall']:
        alerts.append({
            'level': 'CRITICAL',
            'type': 'SYSTEMIC_RISK',
            'message': f"Critical systemic risk level: {risk_indicators.overall_systemic_risk:.2f}",
            'components': risk_indicators.risk_decomposition,
            'recommended_actions': [
                'Reduce portfolio concentration',
                'Increase cash reserves',
                'Implement defensive hedges',
                'Monitor contagion paths'
            ]
        })
    
    # Network fragility alert
    if risk_indicators.network_fragility > thresholds['network']:
        alerts.append({
            'level': 'HIGH',
            'type': 'NETWORK_FRAGILITY',
            'message': f"Network approaching critical fragility: {risk_indicators.network_fragility:.2f}",
            'details': {
                'contagion_potential': risk_indicators.contagion_potential,
                'clustering_risk': risk_indicators.clustering_risk
            },
            'recommended_actions': [
                'Identify and protect critical nodes',
                'Reduce interconnectedness',
                'Prepare firebreaks'
            ]
        })
    
    # Cascade risk alert
    if risk_indicators.cascade_probability > thresholds['cascade']:
        alerts.append({
            'level': 'HIGH',
            'type': 'CASCADE_RISK',
            'message': f"High probability of cascading failures: {risk_indicators.cascade_probability:.2f}",
            'contributing_factors': {
                'herding': risk_indicators.herding_index,
                'synchronization': risk_indicators.synchronization_risk,
                'info_contagion': risk_indicators.information_contagion
            },
            'recommended_actions': [
                'Implement circuit breakers',
                'Diversify strategies',
                'Reduce leverage'
            ]
        })
    
    # Emergence alert
    if risk_indicators.emergence_indicator > thresholds['emergence']:
        alerts.append({
            'level': 'WARNING',
            'type': 'EMERGENT_BEHAVIOR',
            'message': f"Detecting emergent risk patterns: {risk_indicators.emergence_indicator:.2f}",
            'details': {
                'self_organization': risk_indicators.self_organization,
                'diversity_loss': risk_indicators.diversity_loss
            },
            'recommended_actions': [
                'Monitor for regime changes',
                'Prepare for non-linear effects',
                'Increase monitoring frequency'
            ]
        })
    
    # Synchronization alert
    if risk_indicators.synchronization_risk > thresholds['synchronization']:
        alerts.append({
            'level': 'WARNING',
            'type': 'SYNCHRONIZATION',
            'message': f"Dangerous synchronization detected: {risk_indicators.synchronization_risk:.2f}",
            'details': {
                'herding': risk_indicators.herding_index,
                'diversity_loss': risk_indicators.diversity_loss
            },
            'recommended_actions': [
                'Encourage diverse strategies',
                'Implement position limits',
                'Monitor correlation surges'
            ]
        })
    
    return alerts


if __name__ == "__main__":
    # Test emergent risk metrics
    logger.info("Testing Emergent Risk Metrics System...")
    
    # Create test data
    num_assets = 50
    num_time_steps = 100
    
    # Simulate returns
    returns = torch.randn(num_time_steps, num_assets) * 0.02
    
    # Add crisis period with high correlation
    crisis_start = 70
    crisis_factor = torch.randn(1, num_assets) * 0.05
    returns[crisis_start:] += crisis_factor
    
    # Create graphs
    graphs = []
    for t in range(num_time_steps):
        # Increase connectivity during crisis
        if t >= crisis_start:
            num_edges = 500
        else:
            num_edges = 200
        
        edge_index = torch.randint(0, num_assets, (2, num_edges))
        
        # Node features from returns
        if t > 10:
            node_features = torch.stack([
                returns[t-i] for i in range(10)
            ], dim=1)
        else:
            node_features = torch.randn(num_assets, 10) * 0.01
        
        graph = Data(
            x=node_features,
            edge_index=edge_index
        )
        graphs.append(graph)
    
    # Initialize system
    risk_system = EmergentRiskMetrics(
        num_assets=num_assets,
        hidden_dim=256,
        memory_enabled=True
    )
    
    # Compute risks at different time points
    logger.info("\nSystemic Risk Evolution:")
    logger.info("-" * 50)
    
    for t in [30, 60, 75, 85, 95]:
        risk_indicators = risk_system.compute_systemic_risk(
            graphs[:t+1],
            returns_sequence=returns[:t+1]
        )
        
        logger.info(f"\nTime {t}:")
        logger.info(f"Overall Systemic Risk: {risk_indicators.overall_systemic_risk:.3f}")
        logger.info(f"Network Fragility: {risk_indicators.network_fragility:.3f}")
        logger.info(f"Cascade Probability: {risk_indicators.cascade_probability:.3f}")
        logger.info(f"Herding Index: {risk_indicators.herding_index:.3f}")
        logger.info(f"Synchronization Risk: {risk_indicators.synchronization_risk:.3f}")
        
        # Generate alerts
        alerts = create_risk_alert_system(risk_indicators)
        if alerts:
            logger.info(f"\nAlerts: {len(alerts)}")
            for alert in alerts:
                logger.info(f"  [{alert['level']}] {alert['type']}: {alert['message']}")
    
    # Risk decomposition at crisis
    crisis_indicators = risk_system.compute_systemic_risk(
        graphs[:85],
        returns_sequence=returns[:85]
    )
    
    logger.info("\nRisk Decomposition at Crisis:")
    logger.info("-" * 30)
    for component, weight in crisis_indicators.risk_decomposition.items():
        logger.info(f"{component:15s}: {weight:.3f}")
    
    logger.info("\nEmergent Risk Metrics test complete!")