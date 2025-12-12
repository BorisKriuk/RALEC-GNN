#!/usr/bin/env python3
"""
Visualization tools for Financial Network Morphology Theory
Creates publication-quality figures demonstrating theoretical concepts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Optional
import torch

from theoretical_framework import (
    FinancialNetworkMorphology,
    MarketPhaseSpace,
    IsingModelRegimeDetector
)


class TheoryVisualizer:
    """
    Create visualizations for theoretical concepts and results.
    """
    
    def __init__(self):
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.family'] = 'sans-serif'
        
        # Color schemes
        self.regime_colors = {
            'Bull': '#2ecc71',
            'Normal': '#f39c12', 
            'Crisis': '#e74c3c'
        }
        
        self.phase_space_cmap = plt.cm.RdYlGn_r
        
    def plot_phase_space_3d(
        self,
        trajectory: List[MarketPhaseSpace] = None,
        save_path: str = "output/theory/phase_space_3d.png"
    ):
        """
        Plot 3D phase space with regime boundaries and optional trajectory.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create grid for regime boundaries
        vol_range = np.linspace(0, 0.6, 50)
        corr_range = np.linspace(-0.2, 1, 50)
        liq_range = np.linspace(0, 1, 50)
        
        # Plot regime boundaries as surfaces
        morphology = FinancialNetworkMorphology()
        
        # Bull boundary surface
        V, C = np.meshgrid(vol_range[:20], corr_range[:25])
        L = np.ones_like(V) * 0.7
        ax.plot_surface(V, C, L, alpha=0.3, color=self.regime_colors['Bull'])
        
        # Crisis boundary surface  
        V, C = np.meshgrid(vol_range[30:], corr_range[30:])
        L = np.ones_like(V) * 0.4
        ax.plot_surface(V, C, L, alpha=0.3, color=self.regime_colors['Crisis'])
        
        # Plot trajectory if provided
        if trajectory:
            vols = [state.volatility for state in trajectory]
            corrs = [state.correlation for state in trajectory]
            liqs = [state.liquidity for state in trajectory]
            
            # Color by regime
            colors = []
            for state in trajectory:
                probs = morphology.compute_regime_probability(state)
                regime_idx = np.argmax(probs)
                colors.append(['#2ecc71', '#f39c12', '#e74c3c'][regime_idx])
            
            # Plot trajectory
            for i in range(len(trajectory)-1):
                ax.plot(
                    vols[i:i+2], corrs[i:i+2], liqs[i:i+2],
                    color=colors[i], linewidth=2, alpha=0.8
                )
            
            # Mark start and end
            ax.scatter(*trajectory[0].__dict__.values(), s=100, c='green', marker='o', label='Start')
            ax.scatter(*trajectory[-1].__dict__.values(), s=100, c='red', marker='s', label='End')
        
        # Labels and formatting
        ax.set_xlabel('Volatility (σ)', fontsize=12)
        ax.set_ylabel('Correlation (ρ)', fontsize=12)
        ax.set_zlabel('Liquidity (λ)', fontsize=12)
        ax.set_title('Financial Market Phase Space', fontsize=14, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        if trajectory:
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_regime_boundaries_2d(
        self,
        save_path: str = "output/theory/regime_boundaries_2d.png"
    ):
        """
        Plot 2D projections of regime boundaries with smooth transitions.
        """
        morphology = FinancialNetworkMorphology()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        projections = [
            ('volatility', 'correlation', 0.6),  # Fix liquidity
            ('volatility', 'liquidity', 0.5),     # Fix correlation
            ('correlation', 'liquidity', 0.25)    # Fix volatility
        ]
        
        for ax, (x_param, y_param, fixed_val) in zip(axes, projections):
            # Create grid
            x_range = np.linspace(0, 0.6 if x_param == 'volatility' else 1, 100)
            y_range = np.linspace(
                -0.2 if y_param == 'correlation' else 0,
                1,
                100
            )
            X, Y = np.meshgrid(x_range, y_range)
            
            # Compute regime probabilities for each point
            regime_map = np.zeros((100, 100, 3))
            
            for i in range(100):
                for j in range(100):
                    # Create state
                    state_dict = {
                        'volatility': 0.25,
                        'correlation': 0.5,
                        'liquidity': 0.6
                    }
                    
                    state_dict[x_param] = X[i, j]
                    state_dict[y_param] = Y[i, j]
                    
                    state = MarketPhaseSpace(**state_dict)
                    probs = morphology.compute_regime_probability(state)
                    regime_map[i, j] = probs
            
            # Create RGB image from regime probabilities
            rgb_image = np.zeros((100, 100, 3))
            rgb_image[:, :, 1] = regime_map[:, :, 0]  # Bull = Green
            rgb_image[:, :, 0] = regime_map[:, :, 2]  # Crisis = Red
            rgb_image[:, :, 2] = regime_map[:, :, 1] * 0.5  # Normal = Blue tint
            
            # Normalize
            rgb_image = rgb_image / (rgb_image.max() + 1e-8)
            
            ax.imshow(rgb_image, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
                     origin='lower', aspect='auto')
            
            # Add contour lines for boundaries
            bull_contour = ax.contour(X, Y, regime_map[:, :, 0], levels=[0.5],
                                     colors=['green'], linewidths=2)
            crisis_contour = ax.contour(X, Y, regime_map[:, :, 2], levels=[0.5],
                                       colors=['red'], linewidths=2)
            
            # Labels
            ax.set_xlabel(x_param.capitalize())
            ax.set_ylabel(y_param.capitalize())
            ax.set_title(f'Regime Boundaries\n(fixed: {fixed_val:.1f})')
        
        plt.suptitle('Market Regime Boundaries - 2D Projections', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_edge_formation_dynamics(
        self,
        market_states: List[MarketPhaseSpace],
        save_path: str = "output/theory/edge_dynamics.png"
    ):
        """
        Visualize how edge formation probability changes across regimes.
        """
        morphology = FinancialNetworkMorphology()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Generate sample nodes
        np.random.seed(42)
        node_i = np.random.randn(10)
        node_j_similar = node_i + np.random.randn(10) * 0.1
        node_j_different = np.random.randn(10)
        
        # 1. Edge probability vs market stress
        ax = axes[0, 0]
        stress_levels = np.linspace(0, 2, 50)
        
        for regime_name, regime_state in [
            ('Bull', MarketPhaseSpace(0.1, 0.2, 0.8)),
            ('Normal', MarketPhaseSpace(0.2, 0.4, 0.6)),
            ('Crisis', MarketPhaseSpace(0.4, 0.7, 0.3))
        ]:
            edge_probs = []
            for stress in stress_levels:
                # Adjust market state based on stress
                adjusted_state = MarketPhaseSpace(
                    volatility=min(stress * 0.2, 1.0),
                    correlation=min(stress * 0.3, 1.0),
                    liquidity=max(1.0 - stress * 0.3, 0.1)
                )
                
                prob = morphology.edge_formation_probability(
                    node_i, node_j_similar, adjusted_state
                )
                edge_probs.append(prob)
            
            ax.plot(stress_levels, edge_probs, label=regime_name,
                   color=self.regime_colors[regime_name], linewidth=2)
        
        ax.set_xlabel('Market Stress Level')
        ax.set_ylabel('Edge Formation Probability')
        ax.set_title('Edge Formation vs Market Stress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Network density evolution
        ax = axes[0, 1]
        time_steps = range(len(market_states))
        densities = []
        regime_probs_history = []
        
        for state in market_states:
            # Simulate network density
            regime_probs = morphology.compute_regime_probability(state)
            regime_probs_history.append(regime_probs)
            
            # Expected density based on regime
            expected_densities = [0.2, 0.3, 0.5]  # Bull, Normal, Crisis
            density = np.dot(regime_probs, expected_densities)
            densities.append(density)
        
        ax.plot(time_steps, densities, 'k-', linewidth=2, label='Network Density')
        
        # Color background by regime
        regime_probs_array = np.array(regime_probs_history)
        for i, (name, color) in enumerate(self.regime_colors.items()):
            ax.fill_between(time_steps, 0, 1, 
                          where=regime_probs_array[:, i] > 0.5,
                          color=color, alpha=0.2, label=f'{name} Regime')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Network Density')
        ax.set_title('Network Density Evolution')
        ax.set_ylim(0, 0.6)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 3. Similarity vs asymmetry trade-off
        ax = axes[1, 0]
        similarities = np.linspace(-1, 1, 50)
        asymmetries = np.linspace(0, 1, 50)
        
        S, A = np.meshgrid(similarities, asymmetries)
        
        # Fix a crisis state for visualization
        crisis_state = MarketPhaseSpace(0.4, 0.7, 0.3)
        
        # Compute edge probabilities
        edge_prob_map = np.zeros_like(S)
        for i in range(50):
            for j in range(50):
                # Create nodes with specified similarity and asymmetry
                node_1 = np.zeros(10)
                node_2 = node_1 * S[i, j] + np.random.randn(10) * A[i, j]
                
                prob = morphology.edge_formation_probability(
                    node_1, node_2, crisis_state
                )
                edge_prob_map[i, j] = prob
        
        im = ax.contourf(S, A, edge_prob_map, levels=20, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Edge Probability')
        
        ax.set_xlabel('Node Similarity')
        ax.set_ylabel('Information Asymmetry')
        ax.set_title('Edge Formation Trade-offs (Crisis Regime)')
        
        # 4. Percolation visualization
        ax = axes[1, 1]
        edge_densities = np.linspace(0, 1, 100)
        giant_component_sizes = []
        
        n_nodes = 100
        for density in edge_densities:
            # Create random graph with given density
            n_edges = int(density * n_nodes * (n_nodes - 1) / 2)
            G = nx.gnm_random_graph(n_nodes, n_edges)
            
            # Find giant component
            if len(G) > 0 and nx.is_connected(G):
                giant_size = len(max(nx.connected_components(G), key=len))
            else:
                components = list(nx.connected_components(G))
                giant_size = len(max(components, key=len)) if components else 0
            
            giant_component_sizes.append(giant_size / n_nodes)
        
        ax.plot(edge_densities, giant_component_sizes, 'b-', linewidth=2)
        
        # Mark percolation threshold
        avg_degree = 2  # Approximate
        threshold = 1 / (avg_degree * n_nodes)
        ax.axvline(threshold, color='red', linestyle='--', 
                  label=f'Theoretical Threshold: {threshold:.3f}')
        
        ax.set_xlabel('Edge Density')
        ax.set_ylabel('Giant Component Size (fraction)')
        ax.set_title('Percolation Transition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Edge Formation and Network Dynamics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_ising_model_visualization(
        self,
        spin_configurations: List[np.ndarray],
        temperatures: List[float],
        save_path: str = "output/theory/ising_model.png"
    ):
        """
        Visualize Ising model representation of market states.
        """
        n_configs = min(len(spin_configurations), 4)
        
        fig, axes = plt.subplots(2, n_configs, figsize=(4*n_configs, 8))
        
        if n_configs == 1:
            axes = axes.reshape(-1, 1)
        
        ising_model = IsingModelRegimeDetector(num_assets=len(spin_configurations[0]))
        
        for i in range(n_configs):
            spins = spin_configurations[i]
            temp = temperatures[i]
            
            # Top: Spin configuration
            ax = axes[0, i]
            
            # Reshape spins to 2D grid for visualization
            grid_size = int(np.sqrt(len(spins)))
            if grid_size ** 2 < len(spins):
                grid_size += 1
            
            spin_grid = np.zeros((grid_size, grid_size))
            spin_grid.flat[:len(spins)] = spins
            
            im = ax.imshow(spin_grid, cmap='RdBu', vmin=-1, vmax=1)
            ax.set_title(f'T = {temp:.2f}\n({"Low" if temp < 0.2 else "High"} Volatility)')
            ax.set_xticks([])
            ax.set_yticks([])
            
            if i == 0:
                ax.set_ylabel('Spin Configuration\n(Red=Risk-on, Blue=Risk-off)')
            
            # Bottom: Metrics
            ax = axes[1, i]
            
            magnetization = ising_model.compute_magnetization(spins)
            energy = ising_model.compute_energy(spins)
            
            metrics_text = f'Magnetization: {magnetization:.3f}\n'
            metrics_text += f'Energy: {energy:.1f}\n'
            metrics_text += f'Sentiment: {"Bullish" if magnetization > 0.3 else "Bearish" if magnetization < -0.3 else "Mixed"}'
            
            ax.text(0.5, 0.5, metrics_text, transform=ax.transAxes,
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))
            ax.axis('off')
        
        plt.suptitle('Market as Ising Model: Spin Configurations', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_theoretical_predictions_vs_empirical(
        self,
        theoretical_predictions: Dict[str, List[float]],
        empirical_observations: Dict[str, List[float]],
        save_path: str = "output/theory/theory_vs_empirical.png"
    ):
        """
        Compare theoretical predictions with empirical observations.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        metrics = [
            ('edge_density', 'Edge Density', 'Network Density'),
            ('spectral_radius', 'Spectral Radius', 'Contagion Potential'),
            ('clustering', 'Clustering Coefficient', 'Local Connectivity'),
            ('giant_component', 'Giant Component Size', 'Systemic Risk')
        ]
        
        for ax, (metric_key, metric_name, description) in zip(axes, metrics):
            if metric_key in theoretical_predictions and metric_key in empirical_observations:
                theory = theoretical_predictions[metric_key]
                empirical = empirical_observations[metric_key]
                
                # Scatter plot
                ax.scatter(theory, empirical, alpha=0.6, s=50)
                
                # Perfect prediction line
                min_val = min(min(theory), min(empirical))
                max_val = max(max(theory), max(empirical))
                ax.plot([min_val, max_val], [min_val, max_val], 
                       'k--', alpha=0.5, label='Perfect Prediction')
                
                # Regression line
                z = np.polyfit(theory, empirical, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min_val, max_val, 100)
                ax.plot(x_line, p(x_line), 'r-', alpha=0.8,
                       label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
                
                # Correlation
                correlation = np.corrcoef(theory, empirical)[0, 1]
                ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                       transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_xlabel(f'Theoretical {metric_name}')
                ax.set_ylabel(f'Empirical {metric_name}')
                ax.set_title(description)
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Theoretical Predictions vs Empirical Observations',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_theory_visualizations(output_dir: str = "output/theory"):
    """
    Generate all theoretical visualizations.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = TheoryVisualizer()
    
    # 1. Phase space visualization
    # Create sample trajectory
    trajectory = []
    for t in range(50):
        # Simulate crisis development
        if t < 20:
            # Bull phase
            vol = 0.1 + np.random.randn() * 0.02
            corr = 0.2 + np.random.randn() * 0.05
            liq = 0.8 - t * 0.005
        elif t < 35:
            # Transition
            vol = 0.1 + (t - 20) * 0.02 + np.random.randn() * 0.02
            corr = 0.2 + (t - 20) * 0.03 + np.random.randn() * 0.05
            liq = 0.7 - (t - 20) * 0.02
        else:
            # Crisis
            vol = 0.4 + np.random.randn() * 0.05
            corr = 0.7 + np.random.randn() * 0.05
            liq = 0.3 - (t - 35) * 0.01
        
        trajectory.append(MarketPhaseSpace(
            volatility=max(0, vol),
            correlation=np.clip(corr, -1, 1),
            liquidity=np.clip(liq, 0, 1)
        ))
    
    visualizer.plot_phase_space_3d(trajectory, f"{output_dir}/phase_space_3d.png")
    
    # 2. Regime boundaries
    visualizer.plot_regime_boundaries_2d(f"{output_dir}/regime_boundaries_2d.png")
    
    # 3. Edge dynamics
    visualizer.plot_edge_formation_dynamics(trajectory, f"{output_dir}/edge_dynamics.png")
    
    # 4. Ising model
    # Generate spin configurations at different temperatures
    n_assets = 100
    spin_configs = []
    temperatures = [0.1, 0.2, 0.4, 0.8]  # Low to high volatility
    
    for temp in temperatures:
        # Generate spins with temperature-dependent correlation
        if temp < 0.2:
            # Low temp: mostly aligned (bullish)
            spins = np.ones(n_assets)
            spins[np.random.rand(n_assets) < 0.1] = -1
        elif temp < 0.4:
            # Medium temp: mixed
            spins = np.sign(np.random.randn(n_assets))
        else:
            # High temp: random
            spins = np.sign(np.random.randn(n_assets) - 0.2)
        
        spin_configs.append(spins)
    
    visualizer.plot_ising_model_visualization(
        spin_configs, temperatures, f"{output_dir}/ising_model.png"
    )
    
    # 5. Theory vs empirical (with synthetic data for demonstration)
    # In practice, these would come from actual model results
    n_points = 100
    theoretical = {
        'edge_density': np.random.uniform(0.1, 0.5, n_points),
        'spectral_radius': np.random.uniform(0.5, 2.0, n_points),
        'clustering': np.random.uniform(0.2, 0.8, n_points),
        'giant_component': np.random.uniform(0.3, 1.0, n_points)
    }
    
    # Add noise to create "empirical" observations
    empirical = {}
    for key, theory_vals in theoretical.items():
        empirical[key] = theory_vals + np.random.randn(n_points) * 0.1
        empirical[key] = np.clip(empirical[key], 0, 1 if key != 'spectral_radius' else 3)
    
    visualizer.plot_theoretical_predictions_vs_empirical(
        theoretical, empirical, f"{output_dir}/theory_vs_empirical.png"
    )
    
    print(f"Theoretical visualizations saved to {output_dir}/")


if __name__ == "__main__":
    create_theory_visualizations()