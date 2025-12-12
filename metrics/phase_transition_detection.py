#!/usr/bin/env python3
"""
RALEC-GNN Phase 4: Phase Transition Detection Mechanism
Advanced detection and prediction of market regime transitions using
critical phenomena theory, early warning signals, and neural architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats, signal
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
from collections import deque
import logging

from theoretical_framework import (
    MarketPhaseSpace,
    FinancialNetworkMorphology,
    IsingModelRegimeDetector,
    PercolationTheoryAnalyzer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PhaseTransitionIndicators:
    """Collection of early warning indicators for phase transitions"""
    autocorrelation: float  # AR(1) coefficient
    variance: float  # Rolling variance
    skewness: float  # Distribution asymmetry
    critical_slowing_down: float  # Recovery rate from perturbations
    spatial_correlation: float  # Cross-asset correlation
    entropy_rate: float  # Information production rate
    flickering: float  # Regime switching frequency
    hysteresis_gap: float  # Path dependence measure


class EarlyWarningSystem(nn.Module):
    """
    Detects approaching phase transitions using critical phenomena theory.
    
    Based on:
    1. Critical slowing down
    2. Increasing autocorrelation
    3. Rising variance
    4. Spatial correlation increases
    5. Flickering between states
    """
    
    def __init__(
        self,
        window_size: int = 50,
        n_indicators: int = 8,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.window_size = window_size
        
        # Indicator processor
        self.indicator_net = nn.Sequential(
            nn.Linear(n_indicators, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # Distance to each regime boundary
        )
        
        # Transition probability estimator
        self.transition_net = nn.Sequential(
            nn.Linear(n_indicators + 3, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6)  # Transition probabilities between regimes
        )
        
        # Memory for historical indicators
        self.indicator_history = deque(maxlen=window_size)
        
    def compute_indicators(
        self,
        time_series: torch.Tensor,
        causal_adjacency: Optional[torch.Tensor] = None
    ) -> PhaseTransitionIndicators:
        """
        Compute early warning indicators from time series data.
        
        Args:
            time_series: Shape (time, features) 
            causal_adjacency: Optional causal structure
        """
        if len(time_series) < 10:
            # Return default indicators if not enough data
            return PhaseTransitionIndicators(
                autocorrelation=0.0,
                variance=1.0,
                skewness=0.0,
                critical_slowing_down=0.0,
                spatial_correlation=0.0,
                entropy_rate=0.0,
                flickering=0.0,
                hysteresis_gap=0.0
            )
        
        # Convert to numpy for scipy operations
        ts_np = time_series.cpu().numpy()
        
        # 1. Autocorrelation (AR(1) coefficient)
        # High autocorrelation indicates critical slowing down
        autocorr_values = []
        for i in range(ts_np.shape[1]):
            series = ts_np[:, i]
            if np.std(series) > 0:
                autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                autocorr_values.append(autocorr)
        autocorrelation = np.mean(autocorr_values) if autocorr_values else 0.0
        
        # 2. Variance (rolling window)
        # Increasing variance indicates approaching transition
        if len(ts_np) > self.window_size:
            recent = ts_np[-self.window_size:]
            old = ts_np[-2*self.window_size:-self.window_size]
            variance_ratio = np.var(recent) / (np.var(old) + 1e-8)
        else:
            variance_ratio = 1.0
        
        # 3. Skewness
        # Asymmetry in distribution indicates one-sided fluctuations
        skewness = stats.skew(ts_np.flatten())
        
        # 4. Critical slowing down
        # Measure recovery rate from perturbations
        csd = self._compute_critical_slowing_down(ts_np)
        
        # 5. Spatial correlation
        # Increasing correlation across assets
        if ts_np.shape[1] > 1:
            corr_matrix = np.corrcoef(ts_np.T)
            # Average off-diagonal correlations
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            spatial_corr = np.mean(np.abs(corr_matrix[mask]))
        else:
            spatial_corr = 0.0
        
        # 6. Entropy rate
        # Information production in the system
        entropy_rate = self._compute_entropy_rate(ts_np)
        
        # 7. Flickering
        # Frequency of regime switching
        flickering = self._compute_flickering(ts_np)
        
        # 8. Hysteresis gap
        # Memory effects in the system
        hysteresis = self._compute_hysteresis(ts_np, causal_adjacency)
        
        return PhaseTransitionIndicators(
            autocorrelation=autocorrelation,
            variance=variance_ratio,
            skewness=skewness,
            critical_slowing_down=csd,
            spatial_correlation=spatial_corr,
            entropy_rate=entropy_rate,
            flickering=flickering,
            hysteresis_gap=hysteresis
        )
    
    def _compute_critical_slowing_down(self, time_series: np.ndarray) -> float:
        """
        Measure how slowly system recovers from perturbations.
        Uses Ornstein-Uhlenbeck process fitting.
        """
        if len(time_series) < 20:
            return 0.0
        
        # Detrend the series
        detrended = signal.detrend(time_series, axis=0)
        
        # Fit AR(1) model to each dimension
        recovery_rates = []
        for i in range(detrended.shape[1]):
            series = detrended[:, i]
            if np.std(series) > 0:
                # Estimate mean reversion speed
                # dx = -theta*(x-mu)*dt + sigma*dW
                # theta is recovery rate
                x_t = series[1:]
                x_prev = series[:-1]
                
                # Linear regression: x_t = alpha + beta*x_prev
                # theta = -log(beta)
                try:
                    beta = np.corrcoef(x_prev, x_t)[0, 1]
                    if 0 < beta < 1:
                        theta = -np.log(beta)
                        recovery_rates.append(theta)
                except:
                    pass
        
        if recovery_rates:
            # Lower recovery rate = critical slowing down
            avg_recovery = np.mean(recovery_rates)
            csd_score = 1 / (1 + avg_recovery)  # Normalize to [0,1]
        else:
            csd_score = 0.5
        
        return csd_score
    
    def _compute_entropy_rate(self, time_series: np.ndarray) -> float:
        """
        Compute entropy rate using symbolic dynamics.
        Higher entropy = more unpredictable.
        """
        # Discretize time series into symbols
        n_symbols = 3  # Low, medium, high
        
        # Use quantiles for discretization
        symbols = np.zeros_like(time_series, dtype=int)
        for i in range(time_series.shape[1]):
            series = time_series[:, i]
            quantiles = np.percentile(series, [33, 67])
            symbols[:, i] = np.digitize(series, quantiles)
        
        # Compute transition probabilities
        transitions = {}
        for t in range(len(symbols) - 1):
            current_state = tuple(symbols[t])
            next_state = tuple(symbols[t + 1])
            
            if current_state not in transitions:
                transitions[current_state] = {}
            
            if next_state not in transitions[current_state]:
                transitions[current_state][next_state] = 0
            
            transitions[current_state][next_state] += 1
        
        # Compute entropy rate
        entropy_rate = 0.0
        total_transitions = len(symbols) - 1
        
        for current_state, next_states in transitions.items():
            state_prob = sum(next_states.values()) / total_transitions
            
            # Conditional entropy H(X_t+1|X_t=current_state)
            state_entropy = 0.0
            total_from_state = sum(next_states.values())
            
            for next_state, count in next_states.items():
                prob = count / total_from_state
                if prob > 0:
                    state_entropy -= prob * np.log(prob)
            
            entropy_rate += state_prob * state_entropy
        
        return entropy_rate
    
    def _compute_flickering(self, time_series: np.ndarray) -> float:
        """
        Measure frequency of regime switching.
        High flickering indicates bistability near transition.
        """
        # Compute rolling mean
        window = min(10, len(time_series) // 4)
        if window < 3:
            return 0.0
        
        # Identify regime based on mean level
        rolling_mean = np.array([
            np.mean(time_series[max(0, i-window):i+1], axis=0)
            for i in range(len(time_series))
        ])
        
        # Detect crossings of median
        median_level = np.median(rolling_mean, axis=0)
        above_median = rolling_mean > median_level
        
        # Count regime switches
        switches = 0
        for i in range(1, len(above_median)):
            switches += np.sum(above_median[i] != above_median[i-1])
        
        # Normalize by time and dimensions
        flickering_rate = switches / (len(time_series) * time_series.shape[1])
        
        return min(flickering_rate, 1.0)
    
    def _compute_hysteresis(
        self,
        time_series: np.ndarray,
        causal_adjacency: Optional[torch.Tensor]
    ) -> float:
        """
        Measure hysteresis (path dependence) in the system.
        Uses causal structure if available.
        """
        if len(time_series) < 20:
            return 0.0
        
        # Split into up and down phases
        mid_point = len(time_series) // 2
        first_half = time_series[:mid_point]
        second_half = time_series[mid_point:]
        
        # Compare statistics in each half
        mean_diff = np.abs(np.mean(first_half) - np.mean(second_half))
        std_diff = np.abs(np.std(first_half) - np.std(second_half))
        
        # If causal structure available, check for feedback loops
        if causal_adjacency is not None:
            adj_np = causal_adjacency.cpu().numpy()
            # Feedback loops create hysteresis
            feedback_strength = np.trace(adj_np @ adj_np.T) / (adj_np.size + 1e-8)
        else:
            feedback_strength = 0.0
        
        # Combine measures
        hysteresis_score = (mean_diff + std_diff) * (1 + feedback_strength)
        
        return np.tanh(hysteresis_score)  # Normalize to [0,1]
    
    def forward(
        self,
        indicators: PhaseTransitionIndicators,
        current_regime: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict phase transition probabilities.
        
        Returns:
            - Distance to regime boundaries
            - Transition probabilities
            - Warning level
        """
        # Convert indicators to tensor
        indicator_tensor = torch.tensor([
            indicators.autocorrelation,
            indicators.variance,
            indicators.skewness,
            indicators.critical_slowing_down,
            indicators.spatial_correlation,
            indicators.entropy_rate,
            indicators.flickering,
            indicators.hysteresis_gap
        ], dtype=torch.float32)
        
        # Compute distances to regime boundaries
        distances = self.indicator_net(indicator_tensor.unsqueeze(0))
        
        # Compute transition probabilities
        combined_input = torch.cat([indicator_tensor, distances.squeeze()], dim=0)
        transition_logits = self.transition_net(combined_input.unsqueeze(0))
        
        # Reshape to transition matrix (3x3, excluding self-transitions)
        # Order: Bull->Normal, Bull->Crisis, Normal->Bull, Normal->Crisis, Crisis->Bull, Crisis->Normal
        transition_probs = F.softmax(transition_logits, dim=1).squeeze()
        
        # Compute warning level based on indicators
        warning_components = [
            indicators.autocorrelation > 0.8,  # High autocorrelation
            indicators.variance > 1.5,  # Increasing variance
            abs(indicators.skewness) > 1.0,  # High skewness
            indicators.critical_slowing_down > 0.7,  # Slow recovery
            indicators.spatial_correlation > 0.6,  # High correlation
            indicators.flickering > 0.3,  # Frequent regime switches
        ]
        warning_level = sum(warning_components) / len(warning_components)
        
        return {
            'distances': distances.squeeze(),
            'transition_probs': transition_probs,
            'warning_level': torch.tensor(warning_level),
            'indicators': indicator_tensor
        }


class LandscapeReconstructor(nn.Module):
    """
    Reconstructs the effective potential landscape of market states.
    Uses dynamical systems theory to understand stability and transitions.
    """
    
    def __init__(self, state_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        
        # Potential function approximator
        self.potential_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Drift function for SDE
        self.drift_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Diffusion coefficient estimator
        self.diffusion_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim),
            nn.Softplus()  # Ensure positive
        )
        
    def compute_potential(self, state: torch.Tensor) -> torch.Tensor:
        """Compute effective potential at given state"""
        return self.potential_net(state)
    
    def compute_drift(self, state: torch.Tensor) -> torch.Tensor:
        """Compute drift term: -∇V(x)"""
        return self.drift_net(state)
    
    def compute_diffusion(self, state: torch.Tensor) -> torch.Tensor:
        """Compute state-dependent noise level"""
        return self.diffusion_net(state)
    
    def find_fixed_points(
        self,
        bounds: Tuple[torch.Tensor, torch.Tensor],
        n_samples: int = 1000
    ) -> List[torch.Tensor]:
        """
        Find stable fixed points (potential minima) in the landscape.
        """
        lower, upper = bounds
        
        # Sample random initial points
        samples = torch.rand(n_samples, lower.shape[0])
        samples = samples * (upper - lower) + lower
        
        fixed_points = []
        tolerance = 1e-3
        
        # Gradient descent to find local minima
        for sample in samples:
            x = sample.clone().requires_grad_(True)
            
            for _ in range(100):  # Max iterations
                pot = self.compute_potential(x)
                pot.backward()
                
                if x.grad.norm() < tolerance:
                    # Found a fixed point
                    # Check if it's stable (positive definite Hessian)
                    if self._is_stable(x):
                        # Check if it's a new fixed point
                        is_new = True
                        for fp in fixed_points:
                            if torch.dist(x, fp) < 0.1:
                                is_new = False
                                break
                        
                        if is_new:
                            fixed_points.append(x.detach().clone())
                    break
                
                # Gradient descent step
                x.data -= 0.01 * x.grad
                x.grad.zero_()
        
        return fixed_points
    
    def _is_stable(self, point: torch.Tensor) -> bool:
        """Check stability by examining Hessian eigenvalues"""
        # Approximate check: perturb and see if drift points back
        eps = 1e-4
        perturbations = torch.eye(point.shape[0]) * eps
        
        stable = True
        for pert in perturbations:
            perturbed = point + pert
            drift = self.compute_drift(perturbed)
            
            # Check if drift points back towards fixed point
            if torch.dot(drift, pert) > 0:
                stable = False
                break
        
        return stable
    
    def compute_barrier_height(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        n_steps: int = 100
    ) -> float:
        """
        Compute potential barrier between two states.
        Uses string method to find minimum energy path.
        """
        # Linear interpolation as initial path
        path = []
        for i in range(n_steps):
            alpha = i / (n_steps - 1)
            state = (1 - alpha) * start_state + alpha * end_state
            path.append(state)
        
        # Optimize path to find minimum energy path
        # Simplified: just evaluate along linear path
        potentials = []
        for state in path:
            pot = self.compute_potential(state)
            potentials.append(pot.item())
        
        # Barrier height is max potential minus start potential
        barrier = max(potentials) - potentials[0]
        
        return barrier
    
    def forward(self, trajectory: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze trajectory and reconstruct landscape.
        
        Args:
            trajectory: Time series of states (time, state_dim)
        """
        # Compute potentials along trajectory
        potentials = []
        drifts = []
        diffusions = []
        
        for state in trajectory:
            pot = self.compute_potential(state)
            drift = self.compute_drift(state)
            diff = self.compute_diffusion(state)
            
            potentials.append(pot)
            drifts.append(drift)
            diffusions.append(diff)
        
        potentials = torch.stack(potentials)
        drifts = torch.stack(drifts)
        diffusions = torch.stack(diffusions)
        
        # Find potential minima (stable states)
        bounds = (trajectory.min(dim=0)[0], trajectory.max(dim=0)[0])
        fixed_points = self.find_fixed_points(bounds)
        
        # Classify current position relative to basins
        current_state = trajectory[-1]
        current_potential = self.compute_potential(current_state)
        
        # Find nearest basin
        if fixed_points:
            distances = [torch.dist(current_state, fp) for fp in fixed_points]
            nearest_basin_idx = np.argmin(distances)
            distance_to_basin = distances[nearest_basin_idx]
        else:
            nearest_basin_idx = -1
            distance_to_basin = float('inf')
        
        return {
            'potentials': potentials,
            'drifts': drifts,
            'diffusions': diffusions,
            'fixed_points': fixed_points,
            'current_basin': nearest_basin_idx,
            'distance_to_basin': distance_to_basin,
            'landscape_roughness': potentials.std().item()
        }


class RegimeMemoryNetwork(nn.Module):
    """
    Captures regime-specific patterns and memory effects.
    Uses LSTM with regime conditioning for path-dependent dynamics.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_regimes: int = 3,
        memory_length: int = 50
    ):
        super().__init__()
        
        self.num_regimes = num_regimes
        self.memory_length = memory_length
        
        # Regime-specific LSTMs
        self.regime_lstms = nn.ModuleList([
            nn.LSTM(input_dim, hidden_dim, batch_first=True)
            for _ in range(num_regimes)
        ])
        
        # Regime transition memory
        self.transition_lstm = nn.LSTM(
            num_regimes + hidden_dim,
            hidden_dim,
            batch_first=True
        )
        
        # Output heads
        self.regime_predictor = nn.Linear(hidden_dim, num_regimes)
        self.transition_predictor = nn.Linear(hidden_dim * 2, num_regimes * num_regimes)
        
        # Memory banks for each regime
        self.regime_memories = [
            deque(maxlen=memory_length) for _ in range(num_regimes)
        ]
        
    def update_memory(self, features: torch.Tensor, regime: int):
        """Update regime-specific memory bank"""
        self.regime_memories[regime].append(features.detach())
    
    def get_regime_context(self, regime: int) -> Optional[torch.Tensor]:
        """Get historical context for specific regime"""
        if len(self.regime_memories[regime]) > 0:
            context = torch.stack(list(self.regime_memories[regime]))
            return context.mean(dim=0)  # Average representation
        return None
    
    def forward(
        self,
        sequence: torch.Tensor,
        regime_sequence: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Process sequence with regime-aware memory.
        
        Args:
            sequence: Features over time (batch, time, features)
            regime_sequence: Regime indicators (batch, time, num_regimes)
        """
        batch_size, seq_len, _ = sequence.shape
        
        # Process through regime-specific LSTMs
        regime_outputs = []
        for regime_idx in range(self.num_regimes):
            # Weight by regime probability
            regime_weight = regime_sequence[:, :, regime_idx:regime_idx+1]
            weighted_input = sequence * regime_weight
            
            output, (h_n, c_n) = self.regime_lstms[regime_idx](weighted_input)
            regime_outputs.append(output * regime_weight)
        
        # Combine regime outputs
        combined_output = torch.stack(regime_outputs, dim=0).sum(dim=0)
        
        # Process transitions
        transition_input = torch.cat([regime_sequence, combined_output], dim=-1)
        transition_output, (h_t, c_t) = self.transition_lstm(transition_input)
        
        # Predictions
        next_regime_logits = self.regime_predictor(combined_output[:, -1])
        
        # Transition matrix prediction
        transition_features = torch.cat([
            combined_output[:, -1],
            transition_output[:, -1]
        ], dim=-1)
        transition_matrix_flat = self.transition_predictor(transition_features)
        transition_matrix = transition_matrix_flat.view(
            batch_size, self.num_regimes, self.num_regimes
        )
        
        # Compute stability measure (how long in current regime)
        current_regime = regime_sequence[:, -1].argmax(dim=-1)
        stability = self._compute_regime_stability(regime_sequence)
        
        return {
            'next_regime_logits': next_regime_logits,
            'transition_matrix': F.softmax(transition_matrix, dim=-1),
            'regime_stability': stability,
            'hidden_states': combined_output
        }
    
    def _compute_regime_stability(self, regime_sequence: torch.Tensor) -> torch.Tensor:
        """Compute how stable current regime is"""
        # Find duration in current regime
        current_regime = regime_sequence[:, -1].argmax(dim=-1)
        
        stability_scores = []
        for b in range(regime_sequence.shape[0]):
            current = current_regime[b]
            duration = 1
            
            # Count backwards
            for t in range(regime_sequence.shape[1] - 2, -1, -1):
                if regime_sequence[b, t].argmax() == current:
                    duration += 1
                else:
                    break
            
            # Normalize by sequence length
            stability = duration / regime_sequence.shape[1]
            stability_scores.append(stability)
        
        return torch.tensor(stability_scores)


class PhaseTransitionDetector(nn.Module):
    """
    Complete phase transition detection system combining:
    1. Early warning indicators
    2. Landscape reconstruction
    3. Regime memory
    4. Causal structure analysis
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_regimes: int = 3
    ):
        super().__init__()
        
        # Components
        self.early_warning = EarlyWarningSystem(
            window_size=50,
            n_indicators=8,
            hidden_dim=hidden_dim
        )
        
        self.landscape = LandscapeReconstructor(
            state_dim=3,  # (volatility, correlation, liquidity)
            hidden_dim=hidden_dim
        )
        
        self.regime_memory = RegimeMemoryNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_regimes=num_regimes
        )
        
        # Integration network
        self.integration_net = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 8, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_regimes + 1)  # Regimes + transition prob
        )
        
        # Theory components
        self.morphology = FinancialNetworkMorphology()
        self.ising_model = IsingModelRegimeDetector(num_assets=100)
        
    def detect_phase_transition(
        self,
        time_series: torch.Tensor,
        causal_adjacency: Optional[torch.Tensor] = None,
        current_regime: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive phase transition detection.
        
        Returns:
            - Transition probability
            - Warning indicators
            - Landscape analysis
            - Regime predictions
        """
        # 1. Compute early warning indicators
        indicators = self.early_warning.compute_indicators(
            time_series, causal_adjacency
        )
        
        if current_regime is None:
            # Estimate current regime from data
            market_state = self._estimate_market_state(time_series)
            regime_probs = self.morphology.compute_regime_probability(market_state)
            current_regime = torch.tensor(regime_probs)
        
        warning_output = self.early_warning(indicators, current_regime)
        
        # 2. Reconstruct potential landscape
        # Project to phase space (volatility, correlation, liquidity)
        phase_trajectory = self._project_to_phase_space(time_series)
        landscape_output = self.landscape(phase_trajectory)
        
        # 3. Regime memory analysis
        # Create regime sequence (simplified: use estimated regimes)
        regime_sequence = self._estimate_regime_sequence(time_series)
        memory_output = self.regime_memory(
            time_series.unsqueeze(0),
            regime_sequence.unsqueeze(0)
        )
        
        # 4. Integrate all signals
        integration_input = torch.cat([
            warning_output['indicators'],
            landscape_output['potentials'][-1].unsqueeze(0),
            torch.tensor([landscape_output['landscape_roughness']]),
            memory_output['hidden_states'][0, -1]
        ])
        
        integration_output = self.integration_net(integration_input)
        
        # Parse output
        regime_logits = integration_output[:-1]
        transition_prob = torch.sigmoid(integration_output[-1])
        
        # 5. Theory-based analysis
        if causal_adjacency is not None:
            percolation = PercolationTheoryAnalyzer()
            giant_size, _ = percolation.find_giant_component(
                causal_adjacency.cpu().numpy()
            )
            percolation_risk = giant_size / causal_adjacency.shape[0]
        else:
            percolation_risk = 0.0
        
        return {
            # Predictions
            'transition_probability': transition_prob,
            'next_regime_logits': regime_logits,
            'next_regime_probs': F.softmax(regime_logits, dim=0),
            
            # Warning signals
            'warning_level': warning_output['warning_level'],
            'distances_to_boundaries': warning_output['distances'],
            'indicators': indicators,
            
            # Landscape analysis
            'potential_wells': landscape_output['fixed_points'],
            'current_basin': landscape_output['current_basin'],
            'landscape_roughness': landscape_output['landscape_roughness'],
            
            # Memory effects
            'regime_stability': memory_output['regime_stability'],
            'transition_matrix': memory_output['transition_matrix'],
            
            # Theory metrics
            'percolation_risk': percolation_risk,
            'critical_indicators': {
                'autocorrelation': indicators.autocorrelation,
                'variance_ratio': indicators.variance,
                'flickering': indicators.flickering,
                'spatial_correlation': indicators.spatial_correlation
            }
        }
    
    def _estimate_market_state(self, time_series: torch.Tensor) -> MarketPhaseSpace:
        """Estimate market phase space coordinates from time series"""
        # Simplified estimation
        volatility = torch.std(time_series).item()
        
        if time_series.shape[1] > 1:
            corr_matrix = torch.corrcoef(time_series.T)
            avg_correlation = corr_matrix[~torch.eye(corr_matrix.shape[0], dtype=bool)].mean().item()
        else:
            avg_correlation = 0.0
        
        # Liquidity proxy (inverse of volatility spread)
        vol_spread = torch.std(torch.std(time_series, dim=0)).item()
        liquidity = 1 / (1 + vol_spread)
        
        return MarketPhaseSpace(
            volatility=min(volatility, 1.0),
            correlation=np.clip(avg_correlation, -1, 1),
            liquidity=min(liquidity, 1.0)
        )
    
    def _project_to_phase_space(self, time_series: torch.Tensor) -> torch.Tensor:
        """Project time series to (volatility, correlation, liquidity) space"""
        phase_trajectory = []
        
        window = min(20, len(time_series) // 2)
        
        for t in range(window, len(time_series)):
            window_data = time_series[t-window:t]
            
            # Volatility
            vol = torch.std(window_data).item()
            
            # Correlation
            if window_data.shape[1] > 1:
                corr = torch.corrcoef(window_data.T).mean().item()
            else:
                corr = 0.0
            
            # Liquidity proxy
            liq = 1 / (1 + torch.std(torch.std(window_data, dim=0)).item())
            
            phase_trajectory.append(torch.tensor([vol, corr, liq]))
        
        return torch.stack(phase_trajectory)
    
    def _estimate_regime_sequence(self, time_series: torch.Tensor) -> torch.Tensor:
        """Estimate regime probabilities over time"""
        regime_sequence = []
        
        window = min(20, len(time_series) // 2)
        
        for t in range(len(time_series)):
            start = max(0, t - window)
            window_data = time_series[start:t+1]
            
            market_state = self._estimate_market_state(window_data)
            regime_probs = self.morphology.compute_regime_probability(market_state)
            regime_sequence.append(torch.tensor(regime_probs))
        
        return torch.stack(regime_sequence)
    
    def forward(
        self,
        graph_sequence: List[Any],
        lookback: int = 50
    ) -> Dict[str, Any]:
        """
        Process graph sequence for phase transition detection.
        """
        # Extract time series from graph sequence
        time_series_list = []
        causal_adjacencies = []
        
        for graph in graph_sequence[-lookback:]:
            # Assuming graph has node features
            time_series_list.append(graph.x.mean(dim=0))  # Aggregate across nodes
            
            # Get causal structure if available
            if hasattr(graph, 'causal_adjacency'):
                causal_adjacencies.append(graph.causal_adjacency)
        
        if not time_series_list:
            return self._empty_output()
        
        time_series = torch.stack(time_series_list)
        
        # Use latest causal structure
        causal_adj = causal_adjacencies[-1] if causal_adjacencies else None
        
        # Detect phase transition
        return self.detect_phase_transition(time_series, causal_adj)
    
    def _empty_output(self) -> Dict[str, Any]:
        """Return empty output structure"""
        return {
            'transition_probability': torch.tensor(0.0),
            'next_regime_probs': torch.tensor([0.33, 0.33, 0.34]),
            'warning_level': torch.tensor(0.0),
            'critical_indicators': {}
        }


if __name__ == "__main__":
    # Test phase transition detection
    logger.info("Testing Phase Transition Detection System...")
    
    # Create synthetic data with regime change
    n_time = 200
    n_features = 10
    
    # Simulate regime transition
    time_series = []
    
    # Bull regime (low volatility, low correlation)
    for t in range(80):
        features = torch.randn(n_features) * 0.1 + torch.sin(torch.tensor(t * 0.1))
        time_series.append(features)
    
    # Transition period (increasing volatility and correlation)
    for t in range(80, 120):
        alpha = (t - 80) / 40  # Transition progress
        vol = 0.1 + alpha * 0.4
        
        base = torch.randn(1) * vol
        features = base + torch.randn(n_features) * vol * (1 - alpha * 0.5)
        time_series.append(features)
    
    # Crisis regime (high volatility, high correlation)
    for t in range(120, n_time):
        base = torch.randn(1) * 0.5
        features = base + torch.randn(n_features) * 0.2
        time_series.append(features)
    
    time_series = torch.stack(time_series)
    
    # Initialize detector
    detector = PhaseTransitionDetector(
        input_dim=n_features,
        hidden_dim=64,
        num_regimes=3
    )
    
    # Detect transitions at different points
    for checkpoint in [60, 100, 140, 180]:
        window = time_series[:checkpoint]
        result = detector.detect_phase_transition(window)
        
        logger.info(f"\nTime {checkpoint}:")
        logger.info(f"Transition probability: {result['transition_probability']:.3f}")
        logger.info(f"Warning level: {result['warning_level']:.3f}")
        logger.info(f"Next regime probs: {result['next_regime_probs']}")
        logger.info(f"Critical indicators: {result['critical_indicators']}")