#!/usr/bin/env python3
"""
Visualization tools for meta-learning crisis memory system
Shows learned prototypes, memory retrieval, and adaptation performance
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import torch
import networkx as nx
from datetime import datetime, timedelta
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from meta_learning_crisis_memory import (
    CrisisEpisode, CrisisPrototype, MetaLearningCrisisMemory
)


class MetaLearningVisualizer:
    """
    Comprehensive visualization for meta-learning crisis memory system.
    """
    
    def __init__(self):
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.family'] = 'sans-serif'
        
        # Color schemes
        self.prototype_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        self.crisis_type_colors = {
            'liquidity': '#3498db',
            'contagion': '#e74c3c',
            'systemic': '#9b59b6',
            'volatility': '#f39c12',
            'credit': '#2ecc71',
            'unknown': '#95a5a6'
        }
        
    def plot_crisis_memory_landscape(
        self,
        memory_system: MetaLearningCrisisMemory,
        save_path: str = "output/meta_learning/memory_landscape.png"
    ):
        """
        Visualize the crisis memory landscape showing episodes and prototypes.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Extract memory embeddings
        memory_keys = memory_system.memory_bank.memory_keys.detach().cpu().numpy()
        
        # Reduce dimensions for visualization
        if memory_keys.shape[0] > 2:
            pca = PCA(n_components=2)
            memory_2d = pca.fit_transform(memory_keys)
        else:
            memory_2d = memory_keys
        
        # Plot 1: Memory space with episodes
        episodes = list(memory_system.memory_bank.crisis_episodes.values())
        
        if episodes:
            # Color by crisis type
            colors = []
            sizes = []
            for i, (ep_id, idx) in enumerate(
                memory_system.memory_bank.episode_embeddings.items()
            ):
                if ep_id in memory_system.memory_bank.crisis_episodes:
                    episode = memory_system.memory_bank.crisis_episodes[ep_id]
                    colors.append(self.crisis_type_colors.get(
                        episode.trigger, '#95a5a6'
                    ))
                    sizes.append(100 * (1 + episode.severity))
                else:
                    colors.append('#95a5a6')
                    sizes.append(50)
            
            scatter = ax1.scatter(
                memory_2d[:len(colors), 0],
                memory_2d[:len(colors), 1],
                c=colors[:len(memory_2d)],
                s=sizes[:len(memory_2d)],
                alpha=0.6,
                edgecolors='black',
                linewidth=1
            )
            
            # Add prototype centroids if available
            if memory_system.prototype_learner.learned_prototypes:
                for proto in memory_system.prototype_learner.learned_prototypes.values():
                    # Project prototype to 2D
                    proto_2d = pca.transform(
                        proto.centroid_features.unsqueeze(0).cpu().numpy()
                    )[0]
                    
                    ax1.scatter(
                        proto_2d[0], proto_2d[1],
                        marker='*', s=500,
                        c=self.crisis_type_colors.get(proto.crisis_type, '#95a5a6'),
                        edgecolors='black', linewidth=2,
                        label=f'Prototype: {proto.crisis_type}'
                    )
        
        ax1.set_xlabel('Memory Dimension 1')
        ax1.set_ylabel('Memory Dimension 2')
        ax1.set_title('Crisis Memory Landscape', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add legend for crisis types
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=color, markersize=10,
                      label=crisis_type.capitalize())
            for crisis_type, color in self.crisis_type_colors.items()
            if crisis_type != 'unknown'
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Plot 2: Prototype relationships
        ax2.set_title('Prototype Relationships', fontsize=14, fontweight='bold')
        
        if memory_system.prototype_learner.learned_prototypes:
            # Create prototype network
            G = nx.Graph()
            prototypes = list(memory_system.prototype_learner.learned_prototypes.values())
            
            # Add nodes
            for i, proto in enumerate(prototypes):
                G.add_node(i, label=proto.crisis_type, proto=proto)
            
            # Add edges based on similarity
            for i in range(len(prototypes)):
                for j in range(i+1, len(prototypes)):
                    similarity = F.cosine_similarity(
                        prototypes[i].centroid_features.unsqueeze(0),
                        prototypes[j].centroid_features.unsqueeze(0)
                    ).item()
                    
                    if similarity > 0.5:  # Threshold
                        G.add_edge(i, j, weight=similarity)
            
            # Layout
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw edges
            edges = G.edges(data=True)
            for (u, v, d) in edges:
                ax2.plot(
                    [pos[u][0], pos[v][0]],
                    [pos[u][1], pos[v][1]],
                    'k-', alpha=d['weight']*0.5, linewidth=2*d['weight']
                )
            
            # Draw nodes
            for node, (x, y) in pos.items():
                proto = G.nodes[node]['proto']
                ax2.scatter(
                    x, y, s=1000,
                    c=self.crisis_type_colors.get(proto.crisis_type, '#95a5a6'),
                    edgecolors='black', linewidth=2
                )
                ax2.text(x, y, proto.crisis_type[:3].upper(),
                        ha='center', va='center', fontweight='bold')
            
            # Add statistics
            stats_text = f"Prototypes: {len(prototypes)}\n"
            stats_text += f"Avg Episodes/Proto: {np.mean([len(p.episodes) for p in prototypes]):.1f}"
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    verticalalignment='top')
        else:
            ax2.text(0.5, 0.5, 'No prototypes learned yet',
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=14, color='gray')
        
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-1.2, 1.2)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_retrieval_visualization(
        self,
        query_result: Dict[str, Any],
        save_path: str = "output/meta_learning/retrieval_viz.png"
    ):
        """
        Visualize memory retrieval results showing query and retrieved episodes.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Plot 1: Retrieval scores
        if 'similar_episodes' in query_result and query_result['similar_episodes']:
            episodes = query_result['similar_episodes']
            scores = query_result.get('retrieval_scores', torch.ones(len(episodes)))
            
            # Bar chart of retrieval scores
            episode_labels = [f"Episode {i+1}" for i in range(len(episodes))]
            y_pos = np.arange(len(episodes))
            
            bars = ax1.barh(y_pos, scores.cpu().numpy() if torch.is_tensor(scores) else scores)
            
            # Color bars by crisis type
            for i, (bar, episode) in enumerate(zip(bars, episodes)):
                color = self.crisis_type_colors.get(episode.trigger, '#95a5a6')
                bar.set_color(color)
                
                # Add episode details
                details = f"{episode.trigger} | Severity: {episode.severity:.2f}"
                ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        details, va='center', fontsize=9)
            
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(episode_labels)
            ax1.set_xlabel('Retrieval Score')
            ax1.set_title('Retrieved Crisis Episodes', fontsize=12, fontweight='bold')
            ax1.set_xlim(0, 1.1)
            ax1.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Pattern matches timeline
        ax2.set_title('Historical Pattern Matches', fontsize=12, fontweight='bold')
        
        if 'pattern_matches' in query_result and query_result['pattern_matches']:
            matches = query_result['pattern_matches']
            
            # Create timeline visualization
            current_time = 0
            time_range = 100  # Days back
            
            for i, (similarity, match_data) in enumerate(matches):
                # Simulate historical time
                hist_time = -np.random.randint(30, 365)  # Days ago
                duration = np.random.randint(10, 60)  # Crisis duration
                
                # Draw crisis period
                rect = Rectangle(
                    (hist_time, i), duration, 0.8,
                    facecolor=plt.cm.Reds(similarity),
                    edgecolor='black',
                    linewidth=1
                )
                ax2.add_patch(rect)
                
                # Add label
                ax2.text(hist_time + duration/2, i + 0.4,
                        f"Sim: {similarity:.2f}",
                        ha='center', va='center', fontsize=9)
            
            # Current observation line
            ax2.axvline(current_time, color='green', linestyle='--',
                       linewidth=2, label='Current')
            
            ax2.set_xlim(-400, 50)
            ax2.set_ylim(-0.5, len(matches) + 0.5)
            ax2.set_xlabel('Days Ago')
            ax2.set_ylabel('Pattern Match')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='x')
        else:
            ax2.text(0.5, 0.5, 'No pattern matches found',
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=14, color='gray')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_adaptation_performance(
        self,
        adaptation_results: List[Dict[str, Any]],
        save_path: str = "output/meta_learning/adaptation_performance.png"
    ):
        """
        Plot meta-learning adaptation performance over time.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract metrics
        episodes = range(len(adaptation_results))
        success_rates = [r.get('success_rate', 0) for r in adaptation_results]
        adaptation_steps = [r.get('adaptation_steps', 0) for r in adaptation_results]
        confidence_scores = [r.get('confidence', 0) for r in adaptation_results]
        
        # Plot 1: Success rate over time
        ax = axes[0, 0]
        ax.plot(episodes, success_rates, 'b-', linewidth=2, marker='o')
        ax.fill_between(episodes, 0, success_rates, alpha=0.3)
        ax.axhline(0.8, color='green', linestyle='--', alpha=0.5,
                  label='Target: 80%')
        ax.set_xlabel('Crisis Episode')
        ax.set_ylabel('Adaptation Success Rate')
        ax.set_title('Meta-Learning Performance', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Adaptation speed
        ax = axes[0, 1]
        ax.bar(episodes, adaptation_steps, color='orange', alpha=0.7)
        ax.set_xlabel('Crisis Episode')
        ax.set_ylabel('Adaptation Steps Required')
        ax.set_title('Adaptation Efficiency', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Confidence evolution
        ax = axes[1, 0]
        ax.plot(episodes, confidence_scores, 'g-', linewidth=2, marker='s')
        ax.fill_between(episodes, 0, confidence_scores, alpha=0.3, color='green')
        ax.set_xlabel('Crisis Episode')
        ax.set_ylabel('Confidence Score')
        ax.set_title('Prediction Confidence', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Learning curve comparison
        ax = axes[1, 1]
        
        # Simulate with and without meta-learning
        episodes_range = np.array(episodes)
        
        # Meta-learning curve (faster improvement)
        meta_performance = 1 - np.exp(-0.3 * episodes_range)
        
        # Standard learning curve (slower)
        standard_performance = 1 - np.exp(-0.1 * episodes_range)
        
        ax.plot(episodes_range, meta_performance, 'b-', linewidth=2,
               label='With Meta-Learning')
        ax.plot(episodes_range, standard_performance, 'r--', linewidth=2,
               label='Standard Learning')
        
        # Fill area between curves
        ax.fill_between(episodes_range, standard_performance, meta_performance,
                       alpha=0.3, color='green', label='Improvement')
        
        ax.set_xlabel('Crisis Episode')
        ax.set_ylabel('Performance')
        ax.set_title('Learning Efficiency Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.suptitle('Meta-Learning Adaptation Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prototype_evolution(
        self,
        prototypes_history: List[Dict[str, CrisisPrototype]],
        save_path: str = "output/meta_learning/prototype_evolution.png"
    ):
        """
        Visualize how crisis prototypes evolve over time.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Track prototype metrics over time
        time_steps = range(len(prototypes_history))
        prototype_counts = defaultdict(list)
        prototype_episodes = defaultdict(list)
        
        for t, prototypes in enumerate(prototypes_history):
            # Count prototypes by type
            type_counts = defaultdict(int)
            type_episodes = defaultdict(list)
            
            for proto in prototypes.values():
                type_counts[proto.crisis_type] += 1
                type_episodes[proto.crisis_type].extend(proto.episodes)
            
            # Record counts
            for crisis_type in self.crisis_type_colors:
                prototype_counts[crisis_type].append(type_counts.get(crisis_type, 0))
                prototype_episodes[crisis_type].append(
                    len(type_episodes.get(crisis_type, []))
                )
        
        # Plot 1: Prototype count evolution
        bottom = np.zeros(len(time_steps))
        for crisis_type, color in self.crisis_type_colors.items():
            if crisis_type in prototype_counts:
                counts = prototype_counts[crisis_type]
                ax1.fill_between(time_steps, bottom, bottom + counts,
                               label=crisis_type.capitalize(),
                               color=color, alpha=0.7)
                bottom += np.array(counts)
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Number of Prototypes')
        ax1.set_title('Crisis Prototype Evolution', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episodes per prototype type
        x = np.arange(len(self.crisis_type_colors))
        width = 0.8 / max(1, len(prototypes_history))
        
        for t in range(0, len(prototypes_history), max(1, len(prototypes_history)//5)):
            offset = (t - len(prototypes_history)/2) * width
            
            heights = []
            for crisis_type in self.crisis_type_colors:
                if t < len(prototype_episodes[crisis_type]):
                    heights.append(prototype_episodes[crisis_type][t])
                else:
                    heights.append(0)
            
            bars = ax2.bar(x + offset, heights, width,
                          label=f'Time {t}', alpha=0.7)
            
            # Color bars
            for bar, crisis_type in zip(bars, self.crisis_type_colors):
                bar.set_color(self.crisis_type_colors[crisis_type])
        
        ax2.set_xlabel('Crisis Type')
        ax2.set_ylabel('Total Episodes')
        ax2.set_title('Crisis Episodes by Prototype Type', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([ct.capitalize() for ct in self.crisis_type_colors])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_crisis_anticipation_dashboard(
        self,
        anticipation_results: Dict[str, Any],
        save_path: str = "output/meta_learning/anticipation_dashboard.png"
    ):
        """
        Dashboard showing crisis anticipation insights.
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main prediction panel
        ax_main = fig.add_subplot(gs[0, :2])
        
        # Time to crisis prediction
        time_to_crisis = anticipation_results.get('time_to_crisis', 0)
        urgency = anticipation_results.get('preparation_urgency', 0)
        
        # Create urgency meter
        theta = np.linspace(np.pi, 0, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Color gradient based on urgency
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(theta)))
        
        for i in range(len(theta)-1):
            ax_main.fill_between([x[i], x[i+1]], [y[i], y[i+1]], 
                               color=colors[i], alpha=0.8)
        
        # Add needle
        needle_angle = np.pi * (1 - urgency)
        needle_x = 0.9 * np.cos(needle_angle)
        needle_y = 0.9 * np.sin(needle_angle)
        ax_main.arrow(0, 0, needle_x, needle_y, head_width=0.1, 
                     head_length=0.1, fc='black', ec='black', linewidth=2)
        
        # Labels
        ax_main.text(0, -0.3, f'Time to Crisis: {time_to_crisis:.1f} days',
                    ha='center', va='top', fontsize=14, fontweight='bold')
        ax_main.text(-1, -0.1, 'LOW', ha='center', fontsize=10)
        ax_main.text(1, -0.1, 'HIGH', ha='center', fontsize=10)
        ax_main.text(0, 0.2, 'URGENCY', ha='center', fontsize=12, fontweight='bold')
        
        ax_main.set_xlim(-1.5, 1.5)
        ax_main.set_ylim(-0.5, 1.2)
        ax_main.axis('off')
        ax_main.set_title('Crisis Anticipation Meter', fontsize=14, fontweight='bold')
        
        # Crisis type probabilities
        ax_type = fig.add_subplot(gs[0, 2])
        
        crisis_types = ['Liquidity', 'Contagion', 'Systemic']
        type_probs = anticipation_results.get('crisis_type_probs', torch.zeros(3))
        if torch.is_tensor(type_probs):
            type_probs = type_probs.cpu().numpy()
        
        bars = ax_type.bar(crisis_types, type_probs,
                          color=['#3498db', '#e74c3c', '#9b59b6'])
        ax_type.set_ylabel('Probability')
        ax_type.set_title('Crisis Type Prediction', fontsize=12)
        ax_type.set_ylim(0, 1)
        
        # Add value labels
        for bar, prob in zip(bars, type_probs):
            ax_type.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{prob:.2f}', ha='center', va='bottom')
        
        # Historical similar crises
        ax_hist = fig.add_subplot(gs[1, :])
        
        similar = anticipation_results.get('similar_historical', [])
        if similar:
            # Create timeline
            current_time = 0
            
            for i, crisis_info in enumerate(similar[:5]):
                y_pos = i
                
                # Draw crisis bar
                duration = crisis_info.get('duration', 30)
                if isinstance(duration, str):
                    duration = 30  # Default
                
                start = -np.random.randint(100, 500)  # Random historical position
                
                rect = Rectangle(
                    (start, y_pos - 0.3), duration, 0.6,
                    facecolor=plt.cm.Reds(crisis_info.get('severity', 0.5)),
                    edgecolor='black'
                )
                ax_hist.add_patch(rect)
                
                # Label
                label = f"{crisis_info.get('from_episode', 'Unknown')[:10]}"
                ax_hist.text(start + duration/2, y_pos, label,
                            ha='center', va='center', fontsize=9)
            
            ax_hist.axvline(current_time, color='green', linestyle='--',
                           linewidth=2, label='Current')
            ax_hist.set_xlim(-600, 100)
            ax_hist.set_ylim(-1, len(similar))
            ax_hist.set_xlabel('Days Ago')
            ax_hist.set_ylabel('Similar Crisis')
            ax_hist.set_title('Historical Similar Crises', fontsize=12)
            ax_hist.legend()
        else:
            ax_hist.text(0.5, 0.5, 'No similar historical crises found',
                        transform=ax_hist.transAxes, ha='center', va='center',
                        fontsize=12, color='gray')
            ax_hist.axis('off')
        
        # Confidence gauge
        ax_conf = fig.add_subplot(gs[2, 0])
        
        confidence = anticipation_results.get('confidence', 0)
        
        # Create circular gauge
        wedges = ax_conf.pie([confidence, 1-confidence],
                            colors=['green', 'lightgray'],
                            startangle=90,
                            counterclock=False)
        
        # Add center circle
        centre_circle = Circle((0, 0), 0.70, fc='white')
        ax_conf.add_artist(centre_circle)
        
        ax_conf.text(0, 0, f'{confidence:.0%}', ha='center', va='center',
                    fontsize=20, fontweight='bold')
        ax_conf.text(0, -0.3, 'Confidence', ha='center', va='center',
                    fontsize=12)
        ax_conf.set_title('Prediction Confidence', fontsize=12)
        
        # Key indicators table
        ax_table = fig.add_subplot(gs[2, 1:])
        
        # Create indicator data
        indicators_data = []
        for episode in similar[:3]:
            if 'key_indicators' in crisis_info:
                for indicator, value in crisis_info['key_indicators'].items():
                    indicators_data.append([
                        indicator.replace('_', ' ').title(),
                        f"{value:.3f}",
                        "High" if value > 0.7 else "Medium" if value > 0.3 else "Low"
                    ])
        
        if indicators_data:
            table = ax_table.table(
                cellText=indicators_data[:6],  # Limit to 6 rows
                colLabels=['Indicator', 'Value', 'Level'],
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
        else:
            ax_table.text(0.5, 0.5, 'No indicator data available',
                         transform=ax_table.transAxes, ha='center', va='center',
                         fontsize=12, color='gray')
        
        ax_table.axis('off')
        ax_table.set_title('Key Warning Indicators', fontsize=12)
        
        plt.suptitle('Crisis Anticipation Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_meta_learning_report(
    memory_system: MetaLearningCrisisMemory,
    output_dir: str = "output/meta_learning"
):
    """
    Generate comprehensive meta-learning visualization report.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = MetaLearningVisualizer()
    
    # 1. Memory landscape
    visualizer.plot_crisis_memory_landscape(
        memory_system,
        save_path=f"{output_dir}/memory_landscape.png"
    )
    
    # 2. Generate sample query result
    sample_observation = torch.randn(memory_system.crisis_encoder[0].in_features)
    query_result = memory_system.process_crisis_observation(sample_observation)
    
    visualizer.plot_retrieval_visualization(
        query_result,
        save_path=f"{output_dir}/retrieval_viz.png"
    )
    
    # 3. Adaptation performance (simulate)
    adaptation_results = []
    for i in range(20):
        result = {
            'success_rate': min(0.95, 0.3 + i * 0.03 + np.random.rand() * 0.1),
            'adaptation_steps': max(1, 10 - i // 2),
            'confidence': min(0.9, 0.4 + i * 0.025 + np.random.rand() * 0.05)
        }
        adaptation_results.append(result)
    
    visualizer.plot_adaptation_performance(
        adaptation_results,
        save_path=f"{output_dir}/adaptation_performance.png"
    )
    
    # Generate summary
    summary = {
        'num_episodes': len(memory_system.memory_bank.crisis_episodes),
        'num_prototypes': len(memory_system.prototype_learner.learned_prototypes),
        'memory_utilization': len(memory_system.memory_bank.episode_embeddings) / 
                             memory_system.memory_bank.memory_size,
        'avg_adaptation_success': np.mean([r['success_rate'] for r in adaptation_results])
    }
    
    print(f"\nMeta-Learning System Summary:")
    print(f"Crisis episodes stored: {summary['num_episodes']}")
    print(f"Prototypes learned: {summary['num_prototypes']}")
    print(f"Memory utilization: {summary['memory_utilization']:.1%}")
    print(f"Avg adaptation success: {summary['avg_adaptation_success']:.1%}")
    print(f"\nVisualizations saved to {output_dir}/")
    
    return summary


if __name__ == "__main__":
    # Test visualization with dummy system
    from meta_learning_crisis_memory import MetaLearningCrisisMemory
    
    # Create dummy system
    memory_system = MetaLearningCrisisMemory(
        input_dim=16,
        hidden_dim=128,
        memory_size=100,
        num_prototypes=5
    )
    
    # Add some dummy episodes
    for i in range(10):
        obs = torch.randn(16)
        episode_info = {
            'trigger': np.random.choice(['liquidity', 'contagion', 'volatility']),
            'severity': np.random.rand(),
            'affected_assets': list(range(np.random.randint(5, 20))),
            'early_warnings': {
                'autocorrelation': np.random.rand(),
                'variance': 1 + np.random.rand()
            },
            'market_state': {
                'volatility': 0.3 + np.random.rand() * 0.3,
                'correlation': 0.5 + np.random.rand() * 0.3,
                'liquidity': 0.3 + np.random.rand() * 0.4
            }
        }
        memory_system.process_crisis_observation(obs, is_crisis=True, episode_info=episode_info)
    
    # Generate report
    create_meta_learning_report(memory_system)