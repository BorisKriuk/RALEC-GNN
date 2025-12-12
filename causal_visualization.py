#!/usr/bin/env python3
"""
Visualization tools for causal discovery results
Creates publication-quality visualizations of discovered causal relationships
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
import networkx as nx
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import torch
from scipy.stats import gaussian_kde

from causal_discovery import CausalEdge


class CausalGraphVisualizer:
    """
    Comprehensive visualization suite for causal discovery results.
    """
    
    def __init__(self):
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.family'] = 'sans-serif'
        
        # Color schemes
        self.edge_colors = {
            'linear': '#3498db',      # Blue
            'nonlinear': '#e74c3c',   # Red
            'threshold': '#9b59b6'    # Purple
        }
        
        self.node_colors = {
            'source': '#27ae60',      # Green
            'sink': '#e67e22',        # Orange
            'intermediate': '#34495e'  # Dark gray
        }
    
    def plot_causal_graph(
        self,
        causal_edges: List[CausalEdge],
        node_names: Optional[List[str]] = None,
        save_path: str = "output/causal/causal_graph.png",
        layout: str = 'spring'
    ):
        """
        Plot the discovered causal graph with edge types and strengths.
        """
        fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=(16, 10),
                                                 gridspec_kw={'width_ratios': [4, 1]})
        
        # Build networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        all_nodes = set()
        for edge in causal_edges:
            all_nodes.add(edge.source)
            all_nodes.add(edge.target)
        
        G.add_nodes_from(all_nodes)
        
        # Add edges with attributes
        for edge in causal_edges:
            G.add_edge(edge.source, edge.target,
                      strength=edge.strength,
                      confidence=edge.confidence,
                      lag=edge.lag,
                      mechanism=edge.mechanism)
        
        # Layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'hierarchical':
            pos = self._hierarchical_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)
        
        # Draw nodes
        node_sizes = []
        node_colors_list = []
        
        for node in G.nodes():
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            
            # Size based on total degree
            size = 500 + (in_degree + out_degree) * 200
            node_sizes.append(size)
            
            # Color based on role
            if in_degree == 0 and out_degree > 0:
                color = self.node_colors['source']
            elif out_degree == 0 and in_degree > 0:
                color = self.node_colors['sink']
            else:
                color = self.node_colors['intermediate']
            node_colors_list.append(color)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                             node_color=node_colors_list,
                             alpha=0.9, ax=ax_main)
        
        # Draw edges by mechanism type
        for mechanism, color in self.edge_colors.items():
            edges = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get('mechanism') == mechanism]
            
            if edges:
                # Edge widths based on strength
                widths = [G[u][v]['strength'] * 5 for u, v in edges]
                
                # Draw edges with arrows
                nx.draw_networkx_edges(G, pos, edgelist=edges,
                                     edge_color=color, width=widths,
                                     alpha=0.7, arrows=True,
                                     arrowsize=20, arrowstyle='-|>',
                                     connectionstyle='arc3,rad=0.1',
                                     ax=ax_main)
        
        # Add labels
        if node_names:
            labels = {i: name for i, name in enumerate(node_names[:len(G.nodes())])}
        else:
            labels = {i: f'Asset {i}' for i in G.nodes()}
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8,
                              font_weight='bold', ax=ax_main)
        
        # Add lag annotations to edges
        edge_labels = {}
        for u, v, d in G.edges(data=True):
            if d['lag'] > 1:
                edge_labels[(u, v)] = f"lag={d['lag']}"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                   font_size=7, ax=ax_main)
        
        ax_main.set_title("Discovered Causal Graph", fontsize=14, fontweight='bold')
        ax_main.axis('off')
        
        # Create legend
        ax_legend.axis('off')
        
        # Edge type legend
        y_pos = 0.9
        ax_legend.text(0.1, y_pos, "Edge Types:", fontsize=12, fontweight='bold')
        y_pos -= 0.1
        
        for mechanism, color in self.edge_colors.items():
            ax_legend.add_patch(mpatches.FancyArrowPatch(
                (0.1, y_pos), (0.3, y_pos),
                mutation_scale=20, color=color, linewidth=2
            ))
            ax_legend.text(0.35, y_pos, mechanism.capitalize(), va='center')
            y_pos -= 0.08
        
        # Node type legend
        y_pos -= 0.1
        ax_legend.text(0.1, y_pos, "Node Types:", fontsize=12, fontweight='bold')
        y_pos -= 0.1
        
        for node_type, color in self.node_colors.items():
            circle = Circle((0.2, y_pos), 0.05, color=color, alpha=0.9)
            ax_legend.add_patch(circle)
            ax_legend.text(0.35, y_pos, node_type.capitalize(), va='center')
            y_pos -= 0.08
        
        # Statistics
        y_pos -= 0.1
        ax_legend.text(0.1, y_pos, "Statistics:", fontsize=12, fontweight='bold')
        y_pos -= 0.08
        
        stats_text = f"Nodes: {G.number_of_nodes()}\n"
        stats_text += f"Edges: {G.number_of_edges()}\n"
        stats_text += f"Density: {nx.density(G):.3f}\n"
        
        if G.number_of_nodes() > 0:
            try:
                stats_text += f"Avg Clustering: {nx.average_clustering(G):.3f}\n"
            except:
                stats_text += "Avg Clustering: N/A\n"
        
        ax_legend.text(0.1, y_pos, stats_text, fontsize=10, va='top')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _hierarchical_layout(self, G: nx.DiGraph) -> Dict:
        """Create hierarchical layout based on causal structure"""
        # Find layers based on topological sort
        try:
            topo_order = list(nx.topological_sort(G))
        except nx.NetworkXError:
            # Graph has cycles, use spring layout
            return nx.spring_layout(G)
        
        # Assign nodes to layers
        layers = {}
        for node in topo_order:
            # Find maximum layer of predecessors
            pred_layers = [layers.get(pred, -1) for pred in G.predecessors(node)]
            layer = max(pred_layers) + 1 if pred_layers else 0
            layers[node] = layer
        
        # Position nodes
        pos = {}
        layer_counts = {}
        
        for node, layer in layers.items():
            if layer not in layer_counts:
                layer_counts[layer] = 0
            
            x = layer
            y = layer_counts[layer]
            pos[node] = (x, y)
            layer_counts[layer] += 1
        
        # Center each layer
        max_height = max(layer_counts.values())
        for node, (x, y) in pos.items():
            layer_size = layer_counts[layers[node]]
            offset = (max_height - layer_size) / 2
            pos[node] = (x, y + offset)
        
        return pos
    
    def plot_causal_strength_matrix(
        self,
        causal_adjacency: torch.Tensor,
        node_names: Optional[List[str]] = None,
        save_path: str = "output/causal/strength_matrix.png"
    ):
        """
        Plot causal strength matrix as a heatmap.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert to numpy
        matrix = causal_adjacency.detach().cpu().numpy()
        
        # Create mask for zero values
        mask = matrix == 0
        
        # Plot heatmap
        sns.heatmap(matrix, mask=mask, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8, "label": "Causal Strength"},
                   annot=True, fmt='.2f', annot_kws={'size': 8},
                   ax=ax)
        
        # Labels
        n_nodes = matrix.shape[0]
        if node_names and len(node_names) >= n_nodes:
            ax.set_xticklabels(node_names[:n_nodes], rotation=45, ha='right')
            ax.set_yticklabels(node_names[:n_nodes], rotation=0)
        else:
            ax.set_xticklabels([f'Asset {i}' for i in range(n_nodes)], rotation=45, ha='right')
            ax.set_yticklabels([f'Asset {i}' for i in range(n_nodes)], rotation=0)
        
        ax.set_xlabel("Target Asset", fontsize=12)
        ax.set_ylabel("Source Asset", fontsize=12)
        ax.set_title("Causal Strength Matrix", fontsize=14, fontweight='bold')
        
        # Add text annotation
        textstr = f'Non-zero edges: {(matrix > 0).sum()}\nAvg strength: {matrix[matrix > 0].mean():.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_lag_distribution(
        self,
        causal_edges: List[CausalEdge],
        save_path: str = "output/causal/lag_distribution.png"
    ):
        """
        Plot distribution of causal lags and their relationship to edge strength.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract data
        lags = [edge.lag for edge in causal_edges]
        strengths = [edge.strength for edge in causal_edges]
        mechanisms = [edge.mechanism for edge in causal_edges]
        
        # 1. Lag histogram by mechanism
        mechanism_types = list(set(mechanisms))
        x = np.arange(1, max(lags) + 1)
        width = 0.8 / len(mechanism_types)
        
        for i, mechanism in enumerate(mechanism_types):
            mechanism_lags = [edge.lag for edge in causal_edges if edge.mechanism == mechanism]
            lag_counts = [mechanism_lags.count(lag) for lag in x]
            
            ax1.bar(x + i * width - 0.4, lag_counts, width,
                   label=mechanism.capitalize(),
                   color=self.edge_colors[mechanism], alpha=0.8)
        
        ax1.set_xlabel("Lag (time steps)")
        ax1.set_ylabel("Number of Edges")
        ax1.set_title("Causal Lag Distribution by Mechanism")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x)
        
        # 2. Lag vs Strength scatter
        colors = [self.edge_colors[m] for m in mechanisms]
        
        ax2.scatter(lags, strengths, c=colors, alpha=0.6, s=100)
        
        # Add trend line
        z = np.polyfit(lags, strengths, 1)
        p = np.poly1d(z)
        ax2.plot(sorted(lags), p(sorted(lags)), "k--", alpha=0.5,
                label=f'Trend: strength = {z[0]:.3f}×lag + {z[1]:.3f}')
        
        ax2.set_xlabel("Lag (time steps)")
        ax2.set_ylabel("Causal Strength")
        ax2.set_title("Causal Strength vs Lag")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle("Temporal Characteristics of Causal Relationships",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_contagion_simulation(
        self,
        contagion_result: Dict[str, Any],
        node_names: Optional[List[str]] = None,
        save_path: str = "output/causal/contagion_simulation.png"
    ):
        """
        Visualize contagion propagation through causal network.
        """
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
        
        ax_main = fig.add_subplot(gs[0, :])
        ax_prob = fig.add_subplot(gs[1, 0])
        ax_time = fig.add_subplot(gs[1, 1])
        
        # Get data
        contagion_probs = contagion_result['contagion_probabilities'].detach().cpu().numpy()
        arrival_times = contagion_result['arrival_times'].detach().cpu().numpy()
        critical_nodes = contagion_result.get('critical_nodes', [])
        
        n_nodes = len(contagion_probs)
        
        # Create network for visualization
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        
        # Add edges from propagation paths
        paths = contagion_result.get('propagation_paths', {})
        for target, target_paths in paths.items():
            for path in target_paths:
                for i in range(len(path) - 1):
                    G.add_edge(path[i], path[i+1])
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Main plot: Network with contagion probabilities
        # Node colors based on contagion probability
        node_colors = plt.cm.Reds(contagion_probs)
        node_sizes = 300 + contagion_probs * 1000
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                      node_size=node_sizes,
                                      cmap='Reds', vmin=0, vmax=1,
                                      ax=ax_main)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True,
                             arrowsize=15, ax=ax_main)
        
        # Highlight critical nodes
        if critical_nodes:
            critical_node_list = [node[0] for node in critical_nodes[:5]]
            nx.draw_networkx_nodes(G, pos, nodelist=critical_node_list,
                                 node_color='none',
                                 edgecolors='black',
                                 linewidths=3,
                                 node_size=[node_sizes[i] for i in critical_node_list],
                                 ax=ax_main)
        
        # Labels
        if node_names:
            labels = {i: name for i, name in enumerate(node_names[:n_nodes])}
        else:
            labels = {i: f'{i}' for i in range(n_nodes)}
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax_main)
        
        # Color bar
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_main, label='Contagion Probability')
        
        ax_main.set_title("Contagion Propagation Through Causal Network",
                         fontsize=14, fontweight='bold')
        ax_main.axis('off')
        
        # Bottom left: Contagion probability distribution
        ax_prob.bar(range(n_nodes), contagion_probs,
                   color=plt.cm.Reds(contagion_probs))
        ax_prob.set_xlabel("Node")
        ax_prob.set_ylabel("Contagion Probability")
        ax_prob.set_title("Contagion Risk by Node")
        ax_prob.grid(True, alpha=0.3)
        
        # Bottom right: Arrival time distribution
        valid_times = arrival_times[arrival_times < float('inf')]
        if len(valid_times) > 0:
            ax_time.hist(valid_times, bins=10, color='steelblue', alpha=0.7,
                        edgecolor='black')
            ax_time.set_xlabel("Arrival Time (steps)")
            ax_time.set_ylabel("Number of Nodes")
            ax_time.set_title("Contagion Arrival Times")
            ax_time.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_causal_evolution(
        self,
        causal_results_sequence: List[Dict[str, Any]],
        save_path: str = "output/causal/causal_evolution.png"
    ):
        """
        Plot how causal relationships evolve over time.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract time series data
        num_edges = []
        avg_strengths = []
        mechanism_counts = {m: [] for m in ['linear', 'nonlinear', 'threshold']}
        density = []
        
        for result in causal_results_sequence:
            edges = result.get('causal_edges', [])
            num_edges.append(len(edges))
            
            if edges:
                avg_strengths.append(np.mean([e.strength for e in edges]))
                
                for mechanism in mechanism_counts:
                    count = sum(1 for e in edges if e.mechanism == mechanism)
                    mechanism_counts[mechanism].append(count)
            else:
                avg_strengths.append(0)
                for mechanism in mechanism_counts:
                    mechanism_counts[mechanism].append(0)
            
            # Network density
            adj = result.get('causal_adjacency', torch.zeros(1, 1))
            n = adj.shape[0]
            if n > 1:
                density.append((adj > 0).sum().item() / (n * (n - 1)))
            else:
                density.append(0)
        
        time_steps = range(len(causal_results_sequence))
        
        # 1. Number of causal edges over time
        ax = axes[0, 0]
        ax.plot(time_steps, num_edges, 'b-', linewidth=2)
        ax.fill_between(time_steps, 0, num_edges, alpha=0.3)
        ax.set_xlabel("Time")
        ax.set_ylabel("Number of Causal Edges")
        ax.set_title("Causal Edge Count Evolution")
        ax.grid(True, alpha=0.3)
        
        # 2. Average causal strength
        ax = axes[0, 1]
        ax.plot(time_steps, avg_strengths, 'r-', linewidth=2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Average Causal Strength")
        ax.set_title("Causal Strength Evolution")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 3. Mechanism type distribution
        ax = axes[1, 0]
        bottom = np.zeros(len(time_steps))
        
        for mechanism, color in self.edge_colors.items():
            counts = mechanism_counts[mechanism]
            ax.fill_between(time_steps, bottom, bottom + counts,
                          label=mechanism.capitalize(),
                          color=color, alpha=0.7)
            bottom += np.array(counts)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Number of Edges")
        ax.set_title("Causal Mechanism Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Network density
        ax = axes[1, 1]
        ax.plot(time_steps, density, 'g-', linewidth=2)
        ax.fill_between(time_steps, 0, density, alpha=0.3, color='green')
        ax.set_xlabel("Time")
        ax.set_ylabel("Network Density")
        ax.set_title("Causal Network Density")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(density) * 1.1 if density else 1)
        
        plt.suptitle("Evolution of Causal Structure", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_causal_vs_correlation(
        self,
        causal_adjacency: torch.Tensor,
        correlation_matrix: np.ndarray,
        save_path: str = "output/causal/causal_vs_correlation.png"
    ):
        """
        Compare discovered causal relationships with correlations.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Convert causal adjacency to numpy
        causal_np = causal_adjacency.detach().cpu().numpy()
        
        # 1. Correlation matrix
        ax = axes[0]
        im1 = ax.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title("Correlation Matrix")
        ax.set_xlabel("Asset")
        ax.set_ylabel("Asset")
        plt.colorbar(im1, ax=ax, label="Correlation")
        
        # 2. Causal adjacency matrix
        ax = axes[1]
        im2 = ax.imshow(causal_np, cmap='Reds', vmin=0, vmax=1)
        ax.set_title("Causal Adjacency Matrix")
        ax.set_xlabel("Target Asset")
        ax.set_ylabel("Source Asset")
        plt.colorbar(im2, ax=ax, label="Causal Strength")
        
        # 3. Scatter plot: Correlation vs Causality
        ax = axes[2]
        
        # Extract pairwise values
        n = causal_np.shape[0]
        correlations = []
        causalities = []
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    correlations.append(abs(correlation_matrix[i, j]))
                    causalities.append(causal_np[i, j])
        
        # Scatter plot with density
        if len(correlations) > 0:
            # Add some jitter for visualization
            correlations = np.array(correlations)
            causalities = np.array(causalities)
            
            # Create hexbin plot for density
            hb = ax.hexbin(correlations, causalities, gridsize=20, cmap='YlOrBr')
            plt.colorbar(hb, ax=ax, label='Count')
            
            # Add reference lines
            ax.axhline(y=0.1, color='k', linestyle='--', alpha=0.5,
                      label='Causality threshold')
            ax.axvline(x=0.3, color='r', linestyle='--', alpha=0.5,
                      label='Correlation threshold')
            
            ax.set_xlabel("Absolute Correlation")
            ax.set_ylabel("Causal Strength")
            ax.set_title("Correlation vs Causality")
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        plt.suptitle("Causal Discovery vs Correlation Analysis",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_causal_discovery_report(
    causal_results: Dict[str, Any],
    output_dir: str = "output/causal"
):
    """
    Generate comprehensive visualization report for causal discovery results.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = CausalGraphVisualizer()
    
    # Extract components
    causal_edges = causal_results.get('causal_edges', [])
    causal_adjacency = causal_results.get('causal_adjacency')
    
    if not causal_edges:
        print("No causal edges found to visualize.")
        return
    
    # 1. Main causal graph
    visualizer.plot_causal_graph(
        causal_edges,
        save_path=f"{output_dir}/causal_graph.png",
        layout='hierarchical'
    )
    
    # 2. Strength matrix
    if causal_adjacency is not None:
        visualizer.plot_causal_strength_matrix(
            causal_adjacency,
            save_path=f"{output_dir}/strength_matrix.png"
        )
    
    # 3. Lag distribution
    visualizer.plot_lag_distribution(
        causal_edges,
        save_path=f"{output_dir}/lag_distribution.png"
    )
    
    print(f"Causal discovery visualizations saved to {output_dir}/")
    
    # Generate summary statistics
    summary = {
        'num_nodes': len(set([e.source for e in causal_edges] + 
                            [e.target for e in causal_edges])),
        'num_edges': len(causal_edges),
        'avg_strength': np.mean([e.strength for e in causal_edges]),
        'avg_confidence': np.mean([e.confidence for e in causal_edges]),
        'avg_lag': np.mean([e.lag for e in causal_edges]),
        'mechanism_distribution': {}
    }
    
    for mechanism in ['linear', 'nonlinear', 'threshold']:
        count = sum(1 for e in causal_edges if e.mechanism == mechanism)
        summary['mechanism_distribution'][mechanism] = count
    
    print("\nCausal Discovery Summary:")
    print(f"Nodes: {summary['num_nodes']}")
    print(f"Edges: {summary['num_edges']}")
    print(f"Average Strength: {summary['avg_strength']:.3f}")
    print(f"Average Lag: {summary['avg_lag']:.2f}")
    print(f"Mechanisms: {summary['mechanism_distribution']}")
    
    return summary


if __name__ == "__main__":
    # Example usage with synthetic data
    from causal_discovery import CausalEdge
    
    # Create synthetic causal edges
    causal_edges = []
    np.random.seed(42)
    
    for i in range(20):
        edge = CausalEdge(
            source=np.random.randint(0, 10),
            target=np.random.randint(0, 10),
            strength=np.random.uniform(0.1, 0.9),
            confidence=np.random.uniform(0.5, 1.0),
            lag=np.random.randint(1, 6),
            mechanism=np.random.choice(['linear', 'nonlinear', 'threshold'])
        )
        if edge.source != edge.target:
            causal_edges.append(edge)
    
    # Create synthetic adjacency matrix
    causal_adjacency = torch.zeros(10, 10)
    for edge in causal_edges:
        causal_adjacency[edge.source, edge.target] = edge.strength
    
    # Generate report
    causal_results = {
        'causal_edges': causal_edges,
        'causal_adjacency': causal_adjacency
    }
    
    create_causal_discovery_report(causal_results)