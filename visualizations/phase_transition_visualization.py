#!/usr/bin/env python3
"""
Visualization tools for phase transition detection
Creates publication-quality figures showing regime transitions and early warning signals
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import torch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata

from phase_transition_detection import PhaseTransitionIndicators


class PhaseTransitionVisualizer:
    """
    Comprehensive visualization for phase transition detection results.
    """
    
    def __init__(self):
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.family'] = 'sans-serif'
        
        # Color schemes
        self.regime_colors = {
            0: '#2ecc71',  # Bull - Green
            1: '#f39c12',  # Normal - Orange
            2: '#e74c3c',  # Crisis - Red
            'Bull': '#2ecc71',
            'Normal': '#f39c12',
            'Crisis': '#e74c3c'
        }
        
        self.indicator_colors = {
            'autocorrelation': '#3498db',
            'variance': '#e74c3c',
            'skewness': '#9b59b6',
            'critical_slowing_down': '#e67e22',
            'spatial_correlation': '#16a085',
            'entropy_rate': '#f1c40f',
            'flickering': '#95a5a6',
            'hysteresis_gap': '#d35400'
        }
    
    def plot_early_warning_dashboard(
        self,
        indicators_history: List[PhaseTransitionIndicators],
        transition_probs: List[float],
        warning_levels: List[float],
        save_path: str = "output/phase_transitions/early_warning_dashboard.png"
    ):
        """
        Comprehensive dashboard showing all early warning indicators.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(4, 3, height_ratios=[1.5, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        time_steps = range(len(indicators_history))
        
        # Top panel: Transition probability and warning level
        ax_top = fig.add_subplot(gs[0, :])
        
        # Plot transition probability
        ax_top.plot(time_steps, transition_probs, 'k-', linewidth=2,
                   label='Transition Probability')
        ax_top.fill_between(time_steps, 0, transition_probs, alpha=0.3, color='gray')
        
        # Plot warning level
        ax_twin = ax_top.twinx()
        ax_twin.plot(time_steps, warning_levels, 'r--', linewidth=2,
                    label='Warning Level')
        
        # Add danger zones
        ax_top.axhline(0.5, color='orange', linestyle=':', alpha=0.5)
        ax_top.axhline(0.8, color='red', linestyle=':', alpha=0.5)
        
        ax_top.set_ylabel('Transition Probability', fontsize=12)
        ax_twin.set_ylabel('Warning Level', fontsize=12, color='red')
        ax_top.set_title('Phase Transition Risk Assessment', fontsize=14, fontweight='bold')
        ax_top.legend(loc='upper left')
        ax_twin.legend(loc='upper right')
        ax_top.grid(True, alpha=0.3)
        ax_top.set_ylim(0, 1)
        ax_twin.set_ylim(0, 1)
        
        # Extract indicator time series
        indicator_names = [
            'autocorrelation', 'variance', 'skewness', 'critical_slowing_down',
            'spatial_correlation', 'entropy_rate', 'flickering', 'hysteresis_gap'
        ]
        
        indicator_series = {name: [] for name in indicator_names}
        for ind in indicators_history:
            for name in indicator_names:
                indicator_series[name].append(getattr(ind, name))
        
        # Plot individual indicators
        positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1)]
        
        for idx, (name, pos) in enumerate(zip(indicator_names, positions)):
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            
            values = indicator_series[name]
            color = self.indicator_colors[name]
            
            ax.plot(time_steps, values, color=color, linewidth=2)
            ax.fill_between(time_steps, 0, values, alpha=0.3, color=color)
            
            # Add threshold lines based on indicator
            if name == 'autocorrelation':
                ax.axhline(0.8, color='red', linestyle='--', alpha=0.5)
            elif name == 'variance':
                ax.axhline(1.5, color='red', linestyle='--', alpha=0.5)
            elif name == 'critical_slowing_down':
                ax.axhline(0.7, color='red', linestyle='--', alpha=0.5)
            
            ax.set_title(name.replace('_', ' ').title(), fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            
            if pos[0] == 3:  # Bottom row
                ax.set_xlabel('Time')
        
        # Summary statistics box
        ax_summary = fig.add_subplot(gs[3, 2])
        
        latest_indicators = indicators_history[-1] if indicators_history else None
        if latest_indicators:
            summary_text = "Latest Values:\n"
            for name in indicator_names[:4]:  # Show top 4
                value = getattr(latest_indicators, name)
                summary_text += f"{name.replace('_', ' ').title()}: {value:.3f}\n"
        else:
            summary_text = "No data available"
        
        ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_summary.axis('off')
        
        plt.suptitle('Early Warning System Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_potential_landscape_3d(
        self,
        landscape_data: Dict[str, Any],
        trajectory: Optional[torch.Tensor] = None,
        save_path: str = "output/phase_transitions/potential_landscape_3d.png"
    ):
        """
        Plot 3D potential landscape with trajectory.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create grid for potential surface
        x_range = np.linspace(-2, 2, 50)
        y_range = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Compute potential on grid (simplified - using Gaussian mixture)
        Z = np.zeros_like(X)
        
        # Add potential wells at fixed points
        fixed_points = landscape_data.get('fixed_points', [])
        if not fixed_points:
            # Default wells for visualization
            fixed_points = [
                torch.tensor([-1.0, -1.0, 0.0]),  # Bull
                torch.tensor([0.0, 0.0, 0.0]),    # Normal
                torch.tensor([1.0, 1.0, 0.0])     # Crisis
            ]
        
        for i, fp in enumerate(fixed_points):
            fp_2d = fp[:2].numpy()
            # Gaussian well
            Z += -np.exp(-((X - fp_2d[0])**2 + (Y - fp_2d[1])**2))
        
        # Add barriers between wells
        Z += 0.5 * np.exp(-((X**2 + Y**2) / 4))
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7,
                             linewidth=0, antialiased=True)
        
        # Add contour lines at bottom
        contours = ax.contour(X, Y, Z, zdir='z', offset=Z.min() - 0.5,
                            colors='black', alpha=0.3, linewidths=0.5)
        
        # Plot trajectory if provided
        if trajectory is not None and len(trajectory) > 0:
            traj_np = trajectory.numpy()
            if traj_np.shape[1] >= 2:
                # Add z-coordinates (potential values)
                z_coords = []
                for point in traj_np:
                    # Interpolate potential at trajectory points
                    z = griddata((X.ravel(), Y.ravel()), Z.ravel(),
                               (point[0], point[1]), method='linear')
                    z_coords.append(z if not np.isnan(z) else 0)
                
                ax.plot(traj_np[:, 0], traj_np[:, 1], z_coords,
                       'r-', linewidth=2, label='System Trajectory')
                
                # Mark start and end
                ax.scatter(*traj_np[0, :2], z_coords[0], color='green',
                          s=100, marker='o', label='Start')
                ax.scatter(*traj_np[-1, :2], z_coords[-1], color='red',
                          s=100, marker='s', label='End')
        
        # Mark potential wells
        for i, fp in enumerate(fixed_points):
            fp_2d = fp[:2].numpy()
            z_well = -1  # Depth of well
            ax.scatter(*fp_2d, z_well, color='yellow', s=200, marker='*',
                      edgecolors='black', linewidth=2)
            ax.text(fp_2d[0], fp_2d[1], z_well + 0.5,
                   f'Well {i+1}', fontsize=9)
        
        ax.set_xlabel('Volatility', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_zlabel('Potential V(x)', fontsize=12)
        ax.set_title('Market Potential Landscape', fontsize=14, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        if trajectory is not None:
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_regime_transition_timeline(
        self,
        regime_probs_history: List[torch.Tensor],
        transition_events: Optional[List[int]] = None,
        save_path: str = "output/phase_transitions/regime_timeline.png"
    ):
        """
        Plot regime evolution over time with transition events.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Convert to numpy
        regime_probs_array = torch.stack(regime_probs_history).numpy()
        time_steps = range(len(regime_probs_history))
        
        # Plot 1: Stacked area chart of regime probabilities
        ax1.stackplot(time_steps,
                     regime_probs_array[:, 0],  # Bull
                     regime_probs_array[:, 1],  # Normal
                     regime_probs_array[:, 2],  # Crisis
                     labels=['Bull', 'Normal', 'Crisis'],
                     colors=[self.regime_colors[0],
                            self.regime_colors[1],
                            self.regime_colors[2]],
                     alpha=0.8)
        
        # Mark transition events
        if transition_events:
            for event_time in transition_events:
                ax1.axvline(event_time, color='black', linestyle='--',
                          linewidth=2, alpha=0.7)
                ax1.text(event_time, 0.5, 'Transition', rotation=90,
                        va='center', ha='right', fontsize=9)
        
        ax1.set_ylabel('Regime Probability', fontsize=12)
        ax1.set_title('Market Regime Evolution', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Dominant regime
        dominant_regime = regime_probs_array.argmax(axis=1)
        
        # Create colored bars for dominant regime
        for t in range(len(time_steps)):
            regime = dominant_regime[t]
            ax2.add_patch(Rectangle((t, 0), 1, 1,
                                  facecolor=self.regime_colors[regime],
                                  edgecolor='none'))
        
        # Add regime labels
        ax2.text(0.25, 0.5, 'Bull', transform=ax2.transAxes,
                ha='center', va='center', fontweight='bold')
        ax2.text(0.5, 0.5, 'Normal', transform=ax2.transAxes,
                ha='center', va='center', fontweight='bold')
        ax2.text(0.75, 0.5, 'Crisis', transform=ax2.transAxes,
                ha='center', va='center', fontweight='bold')
        
        ax2.set_xlim(0, len(time_steps))
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Regime', fontsize=12)
        ax2.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_critical_phenomena_analysis(
        self,
        detection_results: List[Dict[str, Any]],
        save_path: str = "output/phase_transitions/critical_phenomena.png"
    ):
        """
        Analyze and visualize critical phenomena indicators.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        time_steps = range(len(detection_results))
        
        # Extract time series
        autocorr = [r['critical_indicators'].get('autocorrelation', 0)
                   for r in detection_results]
        variance = [r['critical_indicators'].get('variance_ratio', 1)
                   for r in detection_results]
        flickering = [r['critical_indicators'].get('flickering', 0)
                     for r in detection_results]
        spatial_corr = [r['critical_indicators'].get('spatial_correlation', 0)
                       for r in detection_results]
        
        # 1. Autocorrelation (critical slowing down)
        ax = axes[0, 0]
        ax.plot(time_steps, autocorr, 'b-', linewidth=2)
        ax.fill_between(time_steps, 0, autocorr, alpha=0.3)
        ax.axhline(0.8, color='red', linestyle='--', label='Critical threshold')
        ax.set_ylabel('AR(1) Coefficient')
        ax.set_title('Critical Slowing Down')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Variance increase
        ax = axes[0, 1]
        ax.plot(time_steps, variance, 'r-', linewidth=2)
        ax.fill_between(time_steps, 1, variance, alpha=0.3, color='red')
        ax.axhline(1.5, color='red', linestyle='--', label='Warning threshold')
        ax.set_ylabel('Variance Ratio')
        ax.set_title('Variance Amplification')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Flickering
        ax = axes[1, 0]
        ax.plot(time_steps, flickering, 'g-', linewidth=2)
        ax.fill_between(time_steps, 0, flickering, alpha=0.3, color='green')
        ax.set_xlabel('Time')
        ax.set_ylabel('Flickering Rate')
        ax.set_title('Regime Flickering')
        ax.grid(True, alpha=0.3)
        
        # 4. Phase diagram
        ax = axes[1, 1]
        
        # Create phase space plot
        scatter = ax.scatter(autocorr, variance, c=time_steps,
                           cmap='viridis', s=50, alpha=0.7)
        
        # Add regions
        ax.axvline(0.8, color='red', linestyle=':', alpha=0.5)
        ax.axhline(1.5, color='red', linestyle=':', alpha=0.5)
        
        # Label regions
        ax.text(0.4, 2.5, 'Pre-transition', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        ax.text(0.85, 2.5, 'Critical', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
        
        ax.set_xlabel('Autocorrelation')
        ax.set_ylabel('Variance Ratio')
        ax.set_title('Phase Space Evolution')
        
        # Colorbar for time
        cbar = plt.colorbar(scatter, ax=ax, label='Time')
        
        plt.suptitle('Critical Phenomena Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_transition_matrix_evolution(
        self,
        transition_matrices: List[torch.Tensor],
        save_path: str = "output/phase_transitions/transition_matrix_evolution.png"
    ):
        """
        Visualize how transition probabilities evolve over time.
        """
        n_timepoints = min(len(transition_matrices), 6)  # Show up to 6 timepoints
        indices = np.linspace(0, len(transition_matrices)-1, n_timepoints, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()
        
        regime_names = ['Bull', 'Normal', 'Crisis']
        
        for idx, (ax, t_idx) in enumerate(zip(axes[:n_timepoints], indices)):
            matrix = transition_matrices[t_idx].numpy()
            
            # Plot heatmap
            im = ax.imshow(matrix, cmap='RdBu_r', vmin=0, vmax=1)
            
            # Add text annotations
            for i in range(3):
                for j in range(3):
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                 ha='center', va='center',
                                 color='white' if matrix[i, j] > 0.5 else 'black')
            
            # Labels
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(regime_names)
            ax.set_yticklabels(regime_names)
            ax.set_xlabel('To', fontsize=10)
            ax.set_ylabel('From', fontsize=10)
            ax.set_title(f'Time {t_idx}', fontsize=10)
        
        # Add colorbar
        fig.colorbar(im, ax=axes, label='Transition Probability',
                    fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for idx in range(n_timepoints, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Evolution of Regime Transition Probabilities',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_warning_signal_comparison(
        self,
        detection_results: List[Dict[str, Any]],
        actual_transitions: List[int],
        save_path: str = "output/phase_transitions/warning_comparison.png"
    ):
        """
        Compare warning signals with actual transition events.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        time_steps = range(len(detection_results))
        
        # Extract signals
        transition_probs = [r['transition_probability'].item()
                          if torch.is_tensor(r['transition_probability'])
                          else r['transition_probability']
                          for r in detection_results]
        warning_levels = [r['warning_level'].item()
                        if torch.is_tensor(r['warning_level'])
                        else r['warning_level']
                        for r in detection_results]
        
        # Plot 1: Warning signals
        ax1.plot(time_steps, transition_probs, 'b-', linewidth=2,
                label='Transition Probability')
        ax1.plot(time_steps, warning_levels, 'r--', linewidth=2,
                label='Warning Level')
        
        # Mark actual transitions
        for t in actual_transitions:
            ax1.axvline(t, color='black', linestyle='-', alpha=0.5, linewidth=2)
            ax1.text(t, 0.9, 'Actual\nTransition', rotation=0,
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax1.fill_between(time_steps, 0, transition_probs, alpha=0.3, color='blue')
        ax1.set_ylabel('Probability / Level', fontsize=12)
        ax1.set_title('Warning Signals vs Actual Transitions', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Lead time analysis
        lead_times = []
        detection_times = []
        
        # Find when warnings exceeded threshold before each transition
        threshold = 0.5
        for trans_time in actual_transitions:
            # Look back from transition
            for t in range(max(0, trans_time-50), trans_time):
                if transition_probs[t] > threshold:
                    lead_time = trans_time - t
                    lead_times.append(lead_time)
                    detection_times.append(t)
                    break
        
        if lead_times:
            ax2.bar(detection_times, lead_times, width=1, alpha=0.7,
                   color='green', label='Lead Time')
            
            avg_lead = np.mean(lead_times)
            ax2.axhline(avg_lead, color='red', linestyle='--',
                       label=f'Average Lead Time: {avg_lead:.1f}')
        
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Lead Time (steps)', fontsize=12)
        ax2.set_title('Early Warning Lead Times', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_phase_transition_report(
    detection_results: List[Dict[str, Any]],
    indicators_history: List[PhaseTransitionIndicators],
    output_dir: str = "output/phase_transitions"
):
    """
    Generate comprehensive phase transition analysis report.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = PhaseTransitionVisualizer()
    
    # Extract time series from results
    transition_probs = [r['transition_probability'].item()
                       if torch.is_tensor(r['transition_probability'])
                       else r['transition_probability']
                       for r in detection_results]
    
    warning_levels = [r['warning_level'].item()
                     if torch.is_tensor(r['warning_level'])
                     else r['warning_level']
                     for r in detection_results]
    
    regime_probs = [r['next_regime_probs'] for r in detection_results]
    
    # Detect actual transitions (simplified)
    transitions = []
    for i in range(1, len(regime_probs)):
        prev_regime = torch.argmax(regime_probs[i-1])
        curr_regime = torch.argmax(regime_probs[i])
        if prev_regime != curr_regime:
            transitions.append(i)
    
    # 1. Early warning dashboard
    visualizer.plot_early_warning_dashboard(
        indicators_history,
        transition_probs,
        warning_levels,
        save_path=f"{output_dir}/early_warning_dashboard.png"
    )
    
    # 2. Regime timeline
    visualizer.plot_regime_transition_timeline(
        regime_probs,
        transitions,
        save_path=f"{output_dir}/regime_timeline.png"
    )
    
    # 3. Critical phenomena
    visualizer.plot_critical_phenomena_analysis(
        detection_results,
        save_path=f"{output_dir}/critical_phenomena.png"
    )
    
    # 4. Warning comparison
    visualizer.plot_warning_signal_comparison(
        detection_results,
        transitions,
        save_path=f"{output_dir}/warning_comparison.png"
    )
    
    # 5. Potential landscape (if available)
    if detection_results and 'potential_wells' in detection_results[0]:
        landscape_data = {
            'fixed_points': detection_results[-1].get('potential_wells', []),
            'landscape_roughness': detection_results[-1].get('landscape_roughness', 0)
        }
        visualizer.plot_potential_landscape_3d(
            landscape_data,
            save_path=f"{output_dir}/potential_landscape.png"
        )
    
    # Generate summary statistics
    summary = {
        'num_transitions_detected': len(transitions),
        'avg_warning_lead_time': 0,
        'max_transition_prob': max(transition_probs) if transition_probs else 0,
        'critical_periods': sum(1 for w in warning_levels if w > 0.7),
        'regime_distribution': {}
    }
    
    # Calculate regime distribution
    for probs in regime_probs:
        regime = torch.argmax(probs).item()
        regime_name = ['Bull', 'Normal', 'Crisis'][regime]
        summary['regime_distribution'][regime_name] = \
            summary['regime_distribution'].get(regime_name, 0) + 1
    
    # Normalize distribution
    total = sum(summary['regime_distribution'].values())
    if total > 0:
        for regime in summary['regime_distribution']:
            summary['regime_distribution'][regime] /= total
    
    print(f"\nPhase Transition Analysis Summary:")
    print(f"Transitions detected: {summary['num_transitions_detected']}")
    print(f"Critical periods: {summary['critical_periods']}")
    print(f"Max transition probability: {summary['max_transition_prob']:.3f}")
    print(f"Regime distribution: {summary['regime_distribution']}")
    print(f"\nVisualizations saved to {output_dir}/")
    
    return summary


if __name__ == "__main__":
    # Example usage with synthetic data
    from phase_transition_detection import PhaseTransitionIndicators
    
    # Create synthetic detection results
    n_steps = 100
    detection_results = []
    indicators_history = []
    
    for t in range(n_steps):
        # Simulate approaching transition
        progress = t / n_steps
        
        # Indicators that increase near transition
        indicators = PhaseTransitionIndicators(
            autocorrelation=0.3 + 0.5 * progress + 0.1 * np.random.randn(),
            variance=1.0 + progress + 0.2 * np.random.randn(),
            skewness=0.5 * np.sin(t * 0.1),
            critical_slowing_down=0.2 + 0.6 * progress,
            spatial_correlation=0.3 + 0.4 * progress,
            entropy_rate=0.5 + 0.3 * progress,
            flickering=0.1 + 0.5 * progress * (1 - progress),
            hysteresis_gap=0.2 * progress
        )
        indicators_history.append(indicators)
        
        # Detection results
        result = {
            'transition_probability': torch.tensor(0.1 + 0.8 * progress + 0.1 * np.random.randn()),
            'warning_level': torch.tensor(min(1.0, 0.2 + 0.7 * progress)),
            'next_regime_probs': torch.tensor([
                1 - progress,  # Bull
                0.5 * progress * (1 - progress),  # Normal
                progress * progress  # Crisis
            ]),
            'critical_indicators': {
                'autocorrelation': indicators.autocorrelation,
                'variance_ratio': indicators.variance,
                'flickering': indicators.flickering,
                'spatial_correlation': indicators.spatial_correlation
            }
        }
        detection_results.append(result)
    
    # Generate report
    create_phase_transition_report(detection_results, indicators_history)