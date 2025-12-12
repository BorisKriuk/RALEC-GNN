#!/usr/bin/env python3
"""
Visualization tools for emergent risk metrics
Provides comprehensive dashboards and visualizations for systemic risk monitoring
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Polygon
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import torch
import networkx as nx
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.subplots as ps
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from emergent_risk_metrics import SystemicRiskIndicators


class EmergentRiskVisualizer:
    """
    Comprehensive visualization system for emergent risk metrics.
    """
    
    def __init__(self):
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.family'] = 'sans-serif'
        
        # Risk color schemes
        self.risk_colors = {
            'low': '#2ecc71',
            'medium': '#f39c12',
            'high': '#e74c3c',
            'critical': '#c0392b'
        }
        
        self.component_colors = {
            'network': '#3498db',
            'behavioral': '#9b59b6',
            'informational': '#1abc9c',
            'emergent': '#e74c3c',
            'memory': '#f39c12',
            'cascade': '#e67e22',
            'structural': '#34495e',
            'dynamic': '#16a085',
            'adaptive': '#27ae60',
            'systemic': '#8e44ad'
        }
    
    def create_systemic_risk_dashboard(
        self,
        risk_history: List[SystemicRiskIndicators],
        save_path: str = "output/emergent_risk/systemic_dashboard.html",
        time_labels: Optional[List[str]] = None
    ):
        """
        Create interactive systemic risk dashboard using Plotly.
        """
        if not risk_history:
            return
        
        # Prepare data
        n_times = len(risk_history)
        if time_labels is None:
            time_labels = [f"T{i}" for i in range(n_times)]
        
        # Extract time series
        overall_risk = [r.overall_systemic_risk for r in risk_history]
        network_fragility = [r.network_fragility for r in risk_history]
        cascade_prob = [r.cascade_probability for r in risk_history]
        herding = [r.herding_index for r in risk_history]
        synchronization = [r.synchronization_risk for r in risk_history]
        emergence = [r.emergence_indicator for r in risk_history]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Overall Systemic Risk', 'Risk Decomposition',
                'Network & Cascade Risk', 'Behavioral Risks',
                'Emergence Indicators', 'Risk Alerts'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'indicator'}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.12
        )
        
        # 1. Overall Systemic Risk
        fig.add_trace(
            go.Scatter(
                x=time_labels,
                y=overall_risk,
                mode='lines+markers',
                name='Systemic Risk',
                line=dict(color='red', width=3),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ),
            row=1, col=1
        )
        
        # Add risk thresholds
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                     annotation_text="High Risk", row=1, col=1)
        fig.add_hline(y=0.9, line_dash="dash", line_color="red", 
                     annotation_text="Critical", row=1, col=1)
        
        # 2. Risk Decomposition (stacked area)
        if len(risk_history) > 0 and risk_history[-1].risk_decomposition:
            components = list(risk_history[-1].risk_decomposition.keys())
            
            for i, component in enumerate(components):
                values = [r.risk_decomposition.get(component, 0) for r in risk_history]
                
                fig.add_trace(
                    go.Scatter(
                        x=time_labels,
                        y=values,
                        mode='lines',
                        name=component.capitalize(),
                        stackgroup='risk',
                        fillcolor=self.component_colors.get(component, '#333')
                    ),
                    row=1, col=2
                )
        
        # 3. Network & Cascade Risk
        fig.add_trace(
            go.Scatter(
                x=time_labels, y=network_fragility,
                mode='lines+markers', name='Network Fragility',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=time_labels, y=cascade_prob,
                mode='lines+markers', name='Cascade Probability',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1
        )
        
        # 4. Behavioral Risks
        fig.add_trace(
            go.Scatter(
                x=time_labels, y=herding,
                mode='lines+markers', name='Herding Index',
                line=dict(color='purple', width=2)
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=time_labels, y=synchronization,
                mode='lines+markers', name='Synchronization Risk',
                line=dict(color='green', width=2)
            ),
            row=2, col=2
        )
        
        # 5. Emergence Indicators
        fig.add_trace(
            go.Scatter(
                x=time_labels, y=emergence,
                mode='lines+markers', name='Emergence Indicator',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.2)'
            ),
            row=3, col=1
        )
        
        # 6. Current Risk Gauge
        current_risk = risk_history[-1].overall_systemic_risk
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=current_risk,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Current Risk Level"},
                delta={'reference': risk_history[-2].overall_systemic_risk if len(risk_history) > 1 else 0},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': self._get_risk_color(current_risk)},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgreen"},
                        {'range': [0.3, 0.7], 'color': "lightyellow"},
                        {'range': [0.7, 0.9], 'color': "lightcoral"},
                        {'range': [0.9, 1], 'color': "darkred"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Systemic Risk Dashboard",
            height=1200,
            showlegend=True,
            template='plotly_dark'
        )
        
        # Save
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        
        return fig
    
    def plot_network_fragility_3d(
        self,
        graphs: List[Any],
        risk_indicators: List[SystemicRiskIndicators],
        save_path: str = "output/emergent_risk/network_fragility_3d.png"
    ):
        """
        3D visualization of network fragility evolution.
        """
        fig = plt.figure(figsize=(14, 10))
        
        # Main 3D plot
        ax = fig.add_subplot(121, projection='3d')
        
        # Extract data
        times = np.arange(len(risk_indicators))
        fragilities = [r.network_fragility for r in risk_indicators]
        contagion_potentials = [r.contagion_potential for r in risk_indicators]
        cascade_probs = [r.cascade_probability for r in risk_indicators]
        
        # Create 3D scatter
        scatter = ax.scatter(
            times,
            fragilities,
            cascade_probs,
            c=contagion_potentials,
            cmap='hot',
            s=100,
            alpha=0.6,
            edgecolors='black'
        )
        
        # Add trajectory
        ax.plot(times, fragilities, cascade_probs, 'k-', alpha=0.3, linewidth=2)
        
        # Add projections
        ax.plot(times, fragilities, np.zeros_like(times), 'b--', alpha=0.3)
        ax.plot(times, np.ones_like(times), cascade_probs, 'r--', alpha=0.3)
        
        # Labels
        ax.set_xlabel('Time')
        ax.set_ylabel('Network Fragility')
        ax.set_zlabel('Cascade Probability')
        ax.set_title('Network Risk Evolution in 3D', fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Contagion Potential')
        
        # Side plot: Network snapshots
        ax2 = fig.add_subplot(122)
        
        # Show network at different risk levels
        risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
        risk_indices = [
            np.argmin(fragilities),
            len(fragilities) // 2,
            np.argmax(fragilities)
        ]
        
        for i, (level, idx) in enumerate(zip(risk_levels, risk_indices)):
            if idx < len(graphs):
                G = nx.Graph()
                edge_list = graphs[idx].edge_index.T.cpu().numpy()
                G.add_edges_from(edge_list)
                
                # Subplot position
                pos = nx.spring_layout(G, k=0.5, iterations=20)
                
                # Adjust positions for side-by-side display
                for node in pos:
                    pos[node][0] += i * 2.5
                    pos[node][1] *= 0.3
                    pos[node][1] += (1 - i) * 0.6
                
                # Draw network
                nx.draw_networkx_nodes(
                    G, pos, node_size=20,
                    node_color=self._get_risk_color(fragilities[idx]),
                    alpha=0.7, ax=ax2
                )
                nx.draw_networkx_edges(
                    G, pos, alpha=0.2, ax=ax2
                )
                
                # Label
                ax2.text(
                    i * 2.5 + 0.5, -0.5,
                    f"{level}\n(t={idx})",
                    ha='center', fontsize=10
                )
        
        ax2.set_xlim(-1, 7)
        ax2.set_ylim(-0.6, 1.2)
        ax2.axis('off')
        ax2.set_title('Network Structure at Different Risk Levels', fontsize=12)
        
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_collective_behavior_heatmap(
        self,
        risk_history: List[SystemicRiskIndicators],
        save_path: str = "output/emergent_risk/collective_behavior.png"
    ):
        """
        Heatmap visualization of collective behavior risks.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Prepare data matrix
        n_times = len(risk_history)
        behavior_metrics = [
            'herding_index',
            'synchronization_risk',
            'diversity_loss',
            'information_contagion',
            'uncertainty_propagation',
            'emergence_indicator'
        ]
        
        data_matrix = np.zeros((len(behavior_metrics), n_times))
        
        for t, risk in enumerate(risk_history):
            data_matrix[0, t] = risk.herding_index
            data_matrix[1, t] = risk.synchronization_risk
            data_matrix[2, t] = risk.diversity_loss
            data_matrix[3, t] = risk.information_contagion
            data_matrix[4, t] = risk.uncertainty_propagation
            data_matrix[5, t] = risk.emergence_indicator
        
        # 1. Main heatmap
        ax = axes[0, 0]
        im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlBu_r')
        
        ax.set_yticks(range(len(behavior_metrics)))
        ax.set_yticklabels([m.replace('_', ' ').title() for m in behavior_metrics])
        ax.set_xlabel('Time')
        ax.set_title('Collective Behavior Risk Evolution', fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Risk Level')
        
        # 2. Correlation matrix
        ax = axes[0, 1]
        
        # Compute correlations
        corr_matrix = np.corrcoef(data_matrix)
        
        # Plot correlation
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            ax=ax,
            cbar_kws={'label': 'Correlation'}
        )
        
        ax.set_xticklabels([m.replace('_', ' ')[:10] for m in behavior_metrics], rotation=45)
        ax.set_yticklabels([m.replace('_', ' ')[:10] for m in behavior_metrics])
        ax.set_title('Risk Metric Correlations', fontsize=12, fontweight='bold')
        
        # 3. Phase space plot
        ax = axes[1, 0]
        
        # Plot in herding-synchronization space
        herding = [r.herding_index for r in risk_history]
        sync = [r.synchronization_risk for r in risk_history]
        overall = [r.overall_systemic_risk for r in risk_history]
        
        scatter = ax.scatter(
            herding, sync, c=overall, s=100,
            cmap='RdYlBu_r', alpha=0.6, edgecolors='black'
        )
        
        # Add trajectory
        ax.plot(herding, sync, 'k-', alpha=0.3, linewidth=1)
        
        # Add risk regions
        ax.axhspan(0.7, 1.0, alpha=0.1, color='red', label='High Risk Zone')
        ax.axvspan(0.7, 1.0, alpha=0.1, color='red')
        
        ax.set_xlabel('Herding Index')
        ax.set_ylabel('Synchronization Risk')
        ax.set_title('Behavioral Risk Phase Space', fontsize=12, fontweight='bold')
        ax.legend()
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Overall Risk')
        
        # 4. Time series comparison
        ax = axes[1, 1]
        
        times = np.arange(n_times)
        
        # Plot key behavioral metrics
        ax.plot(times, herding, 'b-', linewidth=2, label='Herding', marker='o', markersize=4)
        ax.plot(times, sync, 'r-', linewidth=2, label='Synchronization', marker='s', markersize=4)
        ax.plot(times, [r.diversity_loss for r in risk_history], 
                'g-', linewidth=2, label='Diversity Loss', marker='^', markersize=4)
        
        # Add shaded risk periods
        high_risk_periods = np.array(overall) > 0.7
        ax.fill_between(times, 0, 1, where=high_risk_periods, 
                       alpha=0.2, color='red', label='High Risk Period')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Risk Level')
        ax.set_title('Behavioral Risk Components', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cascade_risk_network(
        self,
        graph: Any,
        risk_indicators: SystemicRiskIndicators,
        save_path: str = "output/emergent_risk/cascade_network.png"
    ):
        """
        Visualize cascade risk propagation through the network.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Convert to NetworkX
        edge_list = graph.edge_index.T.cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(graph.num_nodes))
        G.add_edges_from(edge_list)
        
        # 1. Network with cascade risk visualization
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Node colors based on systemic importance
        node_importance = risk_indicators.systemic_importance.cpu().numpy()
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_color=node_importance,
            node_size=100 + 500 * node_importance,
            cmap='YlOrRd',
            alpha=0.8,
            ax=ax1
        )
        
        # Draw edges with varying thickness based on contagion potential
        edge_weights = []
        for edge in G.edges():
            # Simulate edge weight based on node importance
            weight = (node_importance[edge[0]] + node_importance[edge[1]]) / 2
            edge_weights.append(weight)
        
        edges = nx.draw_networkx_edges(
            G, pos,
            width=[1 + 3 * w for w in edge_weights],
            alpha=[0.2 + 0.6 * w for w in edge_weights],
            edge_color=edge_weights,
            edge_cmap=plt.cm.Reds,
            ax=ax1
        )
        
        # Highlight critical nodes
        critical_nodes = np.where(node_importance > 0.8)[0]
        if len(critical_nodes) > 0:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=critical_nodes,
                node_color='red',
                node_size=300,
                node_shape='^',
                alpha=0.9,
                ax=ax1,
                label='Critical Nodes'
            )
        
        ax1.set_title(f'Cascade Risk Network\n(Cascade Probability: {risk_indicators.cascade_probability:.2f})',
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Colorbar for node importance
        sm = plt.cm.ScalarMappable(cmap='YlOrRd', 
                                   norm=plt.Normalize(vmin=node_importance.min(), 
                                                     vmax=node_importance.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Systemic Importance')
        
        # 2. Cascade simulation visualization
        ax2.set_title('Cascade Risk Factors', fontsize=14, fontweight='bold')
        
        # Risk factor bars
        factors = {
            'Network\nFragility': risk_indicators.network_fragility,
            'Herding\nIndex': risk_indicators.herding_index,
            'Sync\nRisk': risk_indicators.synchronization_risk,
            'Info\nContagion': risk_indicators.information_contagion,
            'Emergence': risk_indicators.emergence_indicator
        }
        
        x_pos = np.arange(len(factors))
        values = list(factors.values())
        
        bars = ax2.bar(x_pos, values, alpha=0.7, edgecolor='black')
        
        # Color bars by risk level
        for i, (bar, val) in enumerate(zip(bars, values)):
            bar.set_facecolor(self._get_risk_color(val))
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(list(factors.keys()))
        ax2.set_ylabel('Risk Level')
        ax2.set_ylim(0, 1)
        
        # Add cascade probability as horizontal line
        ax2.axhline(risk_indicators.cascade_probability, color='red', 
                   linestyle='--', linewidth=2, label=f'Cascade Probability: {risk_indicators.cascade_probability:.2f}')
        
        # Add threshold line
        ax2.axhline(0.6, color='orange', linestyle=':', linewidth=2, 
                   label='Critical Threshold')
        
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add risk equation
        equation = f"P(cascade) = f(fragility × herding × contagion)"
        ax2.text(0.5, 0.95, equation, transform=ax2.transAxes,
                ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_risk_decomposition_sunburst(
        self,
        risk_indicators: SystemicRiskIndicators,
        save_path: str = "output/emergent_risk/risk_sunburst.html"
    ):
        """
        Create sunburst chart showing risk decomposition.
        """
        # Prepare hierarchical data
        labels = []
        parents = []
        values = []
        colors = []
        
        # Root
        labels.append("Total Risk")
        parents.append("")
        values.append(risk_indicators.overall_systemic_risk)
        colors.append(self._get_risk_color(risk_indicators.overall_systemic_risk))
        
        # Main categories
        categories = {
            'Network': ['network', 'structural'],
            'Behavioral': ['behavioral', 'herding', 'synchronization'],
            'Information': ['informational', 'cascade'],
            'Emergent': ['emergent', 'adaptive', 'memory']
        }
        
        for category, components in categories.items():
            # Category level
            cat_value = sum(risk_indicators.risk_decomposition.get(c, 0) for c in components)
            labels.append(category)
            parents.append("Total Risk")
            values.append(cat_value)
            colors.append(self.component_colors.get(components[0], '#666'))
            
            # Components level
            for comp in components:
                if comp in risk_indicators.risk_decomposition:
                    labels.append(comp.capitalize())
                    parents.append(category)
                    values.append(risk_indicators.risk_decomposition[comp])
                    colors.append(self.component_colors.get(comp, '#999'))
        
        # Create sunburst
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            textinfo="label+percent parent",
            hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<br>%{percentParent}',
        ))
        
        fig.update_layout(
            title="Risk Decomposition",
            width=800,
            height=800
        )
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        
        return fig
    
    def plot_emergence_landscape(
        self,
        risk_history: List[SystemicRiskIndicators],
        save_path: str = "output/emergent_risk/emergence_landscape.png"
    ):
        """
        Visualize emergence and self-organization landscape.
        """
        fig = plt.figure(figsize=(14, 10))
        
        # 3D landscape
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Create grid
        n_points = 50
        sync_range = np.linspace(0, 1, n_points)
        herd_range = np.linspace(0, 1, n_points)
        sync_grid, herd_grid = np.meshgrid(sync_range, herd_range)
        
        # Compute emergence landscape (simplified model)
        emergence_grid = np.zeros_like(sync_grid)
        for i in range(n_points):
            for j in range(n_points):
                s = sync_grid[i, j]
                h = herd_grid[i, j]
                # Emergence increases non-linearly with synchronization and herding
                emergence_grid[i, j] = (s * h) ** 0.5 * (1 + 0.5 * s * h)
        
        # Plot surface
        surf = ax1.plot_surface(
            sync_grid, herd_grid, emergence_grid,
            cmap='hot', alpha=0.7, antialiased=True
        )
        
        # Plot actual trajectory
        if risk_history:
            sync_traj = [r.synchronization_risk for r in risk_history]
            herd_traj = [r.herding_index for r in risk_history]
            emrg_traj = [r.emergence_indicator for r in risk_history]
            
            ax1.plot(sync_traj, herd_traj, emrg_traj,
                    'b-', linewidth=3, label='System Trajectory')
            
            # Mark current position
            ax1.scatter(sync_traj[-1], herd_traj[-1], emrg_traj[-1],
                       c='red', s=200, marker='o', edgecolors='black',
                       linewidth=2, label='Current State')
        
        ax1.set_xlabel('Synchronization Risk')
        ax1.set_ylabel('Herding Index')
        ax1.set_zlabel('Emergence Indicator')
        ax1.set_title('Emergence Landscape', fontsize=14, fontweight='bold')
        ax1.legend()
        
        # 2D projections
        ax2 = fig.add_subplot(122)
        
        # Contour plot
        contour = ax2.contour(sync_grid, herd_grid, emergence_grid, 
                             levels=10, cmap='hot')
        ax2.clabel(contour, inline=True, fontsize=8)
        
        # Plot trajectory
        if risk_history:
            ax2.plot(sync_traj, herd_traj, 'b-', linewidth=2, 
                    marker='o', markersize=5, label='System Path')
            
            # Color points by time
            scatter = ax2.scatter(sync_traj, herd_traj, 
                                c=range(len(sync_traj)), 
                                cmap='viridis', s=50)
            
            # Current position
            ax2.scatter(sync_traj[-1], herd_traj[-1], 
                       c='red', s=200, marker='*', 
                       edgecolors='black', linewidth=2)
        
        # Add risk zones
        ax2.add_patch(Rectangle((0.7, 0.7), 0.3, 0.3, 
                               fill=True, alpha=0.2, color='red',
                               label='Critical Zone'))
        
        ax2.set_xlabel('Synchronization Risk')
        ax2.set_ylabel('Herding Index')
        ax2.set_title('Emergence Risk Map', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_alert_timeline(
        self,
        alerts_history: List[List[Dict[str, Any]]],
        save_path: str = "output/emergent_risk/alert_timeline.png"
    ):
        """
        Visualize alert timeline and patterns.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Process alerts
        alert_types = set()
        alert_data = []
        
        for t, alerts in enumerate(alerts_history):
            for alert in alerts:
                alert_types.add(alert['type'])
                alert_data.append({
                    'time': t,
                    'type': alert['type'],
                    'level': alert['level'],
                    'message': alert['message']
                })
        
        alert_types = sorted(list(alert_types))
        
        # 1. Alert timeline
        type_to_y = {atype: i for i, atype in enumerate(alert_types)}
        level_colors = {
            'WARNING': '#f39c12',
            'HIGH': '#e74c3c',
            'CRITICAL': '#c0392b'
        }
        
        for alert in alert_data:
            y = type_to_y[alert['type']]
            color = level_colors.get(alert['level'], '#95a5a6')
            
            # Plot alert marker
            ax1.scatter(alert['time'], y, c=color, s=200, 
                       marker='o', edgecolors='black', linewidth=2,
                       alpha=0.8, zorder=3)
            
            # Add alert level text
            ax1.text(alert['time'], y + 0.1, alert['level'][0],
                    ha='center', va='bottom', fontsize=8,
                    fontweight='bold')
        
        # Connect consecutive alerts of same type
        for atype in alert_types:
            type_alerts = [a for a in alert_data if a['type'] == atype]
            if len(type_alerts) > 1:
                times = [a['time'] for a in type_alerts]
                y_vals = [type_to_y[atype]] * len(times)
                ax1.plot(times, y_vals, 'k--', alpha=0.3, linewidth=1)
        
        ax1.set_yticks(range(len(alert_types)))
        ax1.set_yticklabels(alert_types)
        ax1.set_xlabel('Time')
        ax1.set_title('Risk Alert Timeline', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=color, markersize=12,
                      label=level, markeredgecolor='black', markeredgewidth=2)
            for level, color in level_colors.items()
        ]
        ax1.legend(handles=legend_elements, loc='upper left')
        
        # 2. Alert frequency
        time_bins = np.arange(0, len(alerts_history) + 1, 5)
        alert_counts = []
        
        for i in range(len(time_bins) - 1):
            count = sum(1 for a in alert_data 
                       if time_bins[i] <= a['time'] < time_bins[i+1])
            alert_counts.append(count)
        
        ax2.bar(time_bins[:-1], alert_counts, width=4, 
               color='red', alpha=0.6, edgecolor='black')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Alert Count')
        ax2.set_title('Alert Frequency Over Time', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_risk_color(self, risk_value: float) -> str:
        """Get color based on risk level."""
        if risk_value < 0.3:
            return self.risk_colors['low']
        elif risk_value < 0.7:
            return self.risk_colors['medium']
        elif risk_value < 0.9:
            return self.risk_colors['high']
        else:
            return self.risk_colors['critical']


def create_emergent_risk_report(
    risk_history: List[SystemicRiskIndicators],
    graphs: List[Any],
    alerts_history: Optional[List[List[Dict[str, Any]]]] = None,
    output_dir: str = "output/emergent_risk"
):
    """
    Generate comprehensive emergent risk visualization report.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = EmergentRiskVisualizer()
    
    # 1. Interactive dashboard
    visualizer.create_systemic_risk_dashboard(
        risk_history,
        save_path=f"{output_dir}/systemic_dashboard.html"
    )
    
    # 2. Network fragility 3D
    if len(graphs) >= 3:
        visualizer.plot_network_fragility_3d(
            graphs,
            risk_history,
            save_path=f"{output_dir}/network_fragility_3d.png"
        )
    
    # 3. Collective behavior heatmap
    visualizer.plot_collective_behavior_heatmap(
        risk_history,
        save_path=f"{output_dir}/collective_behavior.png"
    )
    
    # 4. Cascade risk network
    if graphs and risk_history:
        visualizer.plot_cascade_risk_network(
            graphs[-1],
            risk_history[-1],
            save_path=f"{output_dir}/cascade_network.png"
        )
    
    # 5. Risk decomposition sunburst
    if risk_history:
        visualizer.plot_risk_decomposition_sunburst(
            risk_history[-1],
            save_path=f"{output_dir}/risk_sunburst.html"
        )
    
    # 6. Emergence landscape
    visualizer.plot_emergence_landscape(
        risk_history,
        save_path=f"{output_dir}/emergence_landscape.png"
    )
    
    # 7. Alert timeline
    if alerts_history:
        visualizer.create_alert_timeline(
            alerts_history,
            save_path=f"{output_dir}/alert_timeline.png"
        )
    
    # Generate summary statistics
    if risk_history:
        current_risk = risk_history[-1]
        
        summary = f"""
Emergent Risk Analysis Report
============================

Current Systemic Risk Level: {current_risk.overall_systemic_risk:.3f}

Key Risk Indicators:
- Network Fragility: {current_risk.network_fragility:.3f}
- Cascade Probability: {current_risk.cascade_probability:.3f}
- Herding Index: {current_risk.herding_index:.3f}
- Synchronization Risk: {current_risk.synchronization_risk:.3f}
- Emergence Indicator: {current_risk.emergence_indicator:.3f}

Risk Decomposition:
"""
        for component, weight in current_risk.risk_decomposition.items():
            summary += f"- {component.capitalize()}: {weight:.3f}\n"
        
        with open(f"{output_dir}/risk_summary.txt", 'w') as f:
            f.write(summary)
        
        print(summary)
    
    print(f"\nEmergent risk visualizations saved to {output_dir}/")


if __name__ == "__main__":
    # Test visualization with dummy data
    from emergent_risk_metrics import EmergentRiskMetrics, create_risk_alert_system
    import torch
    from torch_geometric.data import Data
    
    # Create dummy risk history
    num_steps = 50
    num_assets = 30
    
    # Initialize system
    risk_system = EmergentRiskMetrics(num_assets=num_assets)
    
    risk_history = []
    alerts_history = []
    graphs = []
    
    # Simulate evolution
    for t in range(num_steps):
        # Create graph
        if t < 30:
            num_edges = 100
        else:
            num_edges = 300  # Crisis period
        
        edge_index = torch.randint(0, num_assets, (2, num_edges))
        node_features = torch.randn(num_assets, 10)
        
        graph = Data(x=node_features, edge_index=edge_index)
        graphs.append(graph)
        
        # Compute risk
        returns = torch.randn(20, num_assets) * (0.02 if t < 30 else 0.05)
        
        risk_indicators = risk_system.compute_systemic_risk(
            [graph],
            returns_sequence=returns
        )
        risk_history.append(risk_indicators)
        
        # Generate alerts
        alerts = create_risk_alert_system(risk_indicators)
        alerts_history.append(alerts)
    
    # Generate report
    create_emergent_risk_report(
        risk_history,
        graphs,
        alerts_history
    )