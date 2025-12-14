#!/usr/bin/env python3

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import networkx as nx
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

SEED = 42

ASSET_UNIVERSE = {
    'tech_mega': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
    'finance': ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC', 'BLK'],
    'healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV'],
    'energy': ['XOM', 'CVX', 'COP', 'SLB'],
    'industrials': ['CAT', 'BA', 'HON', 'UPS', 'GE'],
    'consumer': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD'],
    'sector_etfs': ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE'],
    'intl_developed': ['EWJ', 'EWG', 'EWU', 'EWQ', 'EWL', 'EWA', 'EWC'],
    'intl_emerging': ['FXI', 'EWZ', 'EWY', 'EWT', 'EWW', 'EWS', 'INDA', 'VWO'],
    'fixed_income': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'EMB', 'AGG'],
    'commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DBA'],
    'volatility': ['VXX'],
    'broad_market': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'],
}

SYMBOL_CATEGORIES = {}
for category, symbols in ASSET_UNIVERSE.items():
    for sym in symbols:
        SYMBOL_CATEGORIES[sym] = category


class PublicationVisualizations:
    REGIME_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
    REGIME_NAMES = {0: 'Bull/Low Vol', 1: 'Normal', 2: 'Crisis'}
    
    @staticmethod
    def set_publication_style():
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 11,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'figure.titlesize': 18,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
    
    @staticmethod
    def plot_comprehensive_results(
        regime_df: pd.DataFrame,
        cv_results: Dict[str, Any],
        baseline_results: Dict[str, Dict],
        output_dir: str = "output/graphs"
    ):
        PublicationVisualizations.set_publication_style()
        
        fig = plt.figure(figsize=(20, 24))
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        ax1 = fig.add_subplot(gs[0, :])
        PublicationVisualizations._plot_regime_timeline(ax1, regime_df)
        
        ax2 = fig.add_subplot(gs[1, 0])
        if 'all_predictions' in cv_results and 'all_labels' in cv_results:
            PublicationVisualizations._plot_confusion_matrix(
                ax2, cv_results['all_labels'], cv_results['all_predictions']
            )
        
        ax3 = fig.add_subplot(gs[1, 1])
        if 'overall_metrics' in cv_results:
            PublicationVisualizations._plot_model_comparison(
                ax3, cv_results['overall_metrics'], cv_results, baseline_results
            )
        
        ax4 = fig.add_subplot(gs[2, 0])
        if 'overall_metrics' in cv_results:
            PublicationVisualizations._plot_per_class_metrics(ax4, cv_results['overall_metrics'])
        
        ax5 = fig.add_subplot(gs[2, 1])
        if 'all_probabilities' in cv_results and 'all_labels' in cv_results:
            PublicationVisualizations._plot_calibration(
                ax5, cv_results['all_labels'], cv_results['all_probabilities']
            )
        
        ax6 = fig.add_subplot(gs[3, 0])
        PublicationVisualizations._plot_volatility_by_regime(ax6, regime_df)
        
        ax7 = fig.add_subplot(gs[3, 1])
        PublicationVisualizations._plot_regime_statistics(ax7, regime_df)
        
        plt.suptitle('Cross-Market Contagion Networks: Comprehensive Results', 
                     fontsize=20, fontweight='bold', y=1.02)
        
        plt.savefig(f'{output_dir}/comprehensive_results.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved comprehensive results to {output_dir}/comprehensive_results.png")
        plt.close()
    
    @staticmethod
    def _plot_regime_timeline(ax, regime_df: pd.DataFrame):
        dates = pd.to_datetime(regime_df['date'])
        
        for r in [0, 1, 2]:
            mask = regime_df['regime'] == r
            ax.fill_between(dates, 0, 1, where=mask, alpha=0.7, 
                          color=PublicationVisualizations.REGIME_COLORS[r],
                          label=PublicationVisualizations.REGIME_NAMES[r])
        
        events = {
            '2008-09-15': 'Lehman',
            '2020-03-12': 'COVID',
            '2022-02-24': 'Ukraine',
        }
        
        for date_str, label in events.items():
            try:
                event_date = pd.to_datetime(date_str)
                if event_date >= dates.min() and event_date <= dates.max():
                    ax.axvline(event_date, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
                    ax.text(event_date, 1.05, label, rotation=45, fontsize=9, ha='left')
            except:
                pass
        
        ax.set_ylabel('Market Regime')
        ax.set_xlabel('Date')
        ax.set_title('Market Regime Evolution (2005-2025)', fontweight='bold')
        ax.legend(loc='upper right', ncol=3)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
    
    @staticmethod
    def _plot_confusion_matrix(ax, y_true: np.ndarray, y_pred: np.ndarray):
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax,
                   xticklabels=['Bull', 'Normal', 'Crisis'],
                   yticklabels=['Bull', 'Normal', 'Crisis'])
        
        ax.set_ylabel('True Regime')
        ax.set_xlabel('Predicted Regime')
        ax.set_title('Confusion Matrix (Normalized)', fontweight='bold')
    
    @staticmethod
    def _plot_model_comparison(ax, gnn_metrics, cv_results: Dict, baseline_results: Dict[str, Dict]):
        models = ['RALEC-GNN', 'Random Forest', 'Gradient Boosting', 'Logistic Reg']
        metrics_to_plot = ['accuracy', 'balanced_accuracy', 'macro_f1', 'crisis_recall']
        metric_labels = ['Accuracy', 'Balanced Acc', 'Macro F1', 'Crisis Recall']
        
        x = np.arange(len(metrics_to_plot))
        width = 0.2
        
        gnn_values = [gnn_metrics.accuracy, gnn_metrics.balanced_accuracy, gnn_metrics.macro_f1, gnn_metrics.crisis_recall]
        gnn_stds = [cv_results.get('std_val_acc', 0), 0, 0, cv_results.get('std_crisis_recall', 0)]
        
        ax.bar(x, gnn_values, width, label='RALEC-GNN', alpha=0.8, yerr=gnn_stds, capsize=3)
        
        baseline_names = ['random_forest', 'gradient_boosting', 'logistic_regression']
        display_names = ['Random Forest', 'Gradient Boosting', 'Logistic Reg']
        
        for i, (bname, dname) in enumerate(zip(baseline_names, display_names)):
            if bname in baseline_results:
                br = baseline_results[bname]
                values = [
                    br.get('accuracy_mean', 0),
                    br.get('balanced_accuracy_mean', 0),
                    br.get('macro_f1_mean', 0),
                    br.get('crisis_recall_mean', 0)
                ]
                stds = [
                    br.get('accuracy_std', 0),
                    br.get('balanced_accuracy_std', 0),
                    br.get('macro_f1_std', 0),
                    br.get('crisis_recall_std', 0)
                ]
                ax.bar(x + (i + 1) * width, values, width, label=dname, alpha=0.8, yerr=stds, capsize=3)
        
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison (with std)', fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(metric_labels, rotation=15)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(0, 1)
    
    @staticmethod
    def _plot_per_class_metrics(ax, metrics):
        classes = ['Bull/Low Vol', 'Normal', 'Crisis']
        x = np.arange(len(classes))
        width = 0.25
        
        precision = [metrics.precision_per_class.get(i, 0) for i in range(3)]
        recall = [metrics.recall_per_class.get(i, 0) for i in range(3)]
        f1 = [metrics.f1_per_class.get(i, 0) for i in range(3)]
        
        ax.bar(x - width, precision, width, label='Precision', color='#3498db')
        ax.bar(x, recall, width, label='Recall', color='#e74c3c')
        ax.bar(x + width, f1, width, label='F1-Score', color='#2ecc71')
        
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim(0, 1)
    
    @staticmethod
    def _plot_calibration(ax, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                bin_accuracies.append(accuracies[in_bin].mean())
        
        ax.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, label='Observed')
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Plot', fontweight='bold')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    @staticmethod
    def _plot_volatility_by_regime(ax, regime_df: pd.DataFrame):
        for r in [0, 1, 2]:
            vol = regime_df[regime_df['regime'] == r]['volatility']
            ax.hist(vol, bins=30, alpha=0.5, 
                   color=PublicationVisualizations.REGIME_COLORS[r],
                   label=PublicationVisualizations.REGIME_NAMES[r])
        
        ax.set_xlabel('Annualized Volatility')
        ax.set_ylabel('Frequency')
        ax.set_title('Volatility Distribution by Regime', fontweight='bold')
        ax.legend()
    
    @staticmethod
    def _plot_regime_statistics(ax, regime_df: pd.DataFrame):
        stats_data = []
        
        for r in [0, 1, 2]:
            subset = regime_df[regime_df['regime'] == r]
            stats_data.append([
                PublicationVisualizations.REGIME_NAMES[r],
                f"{len(subset)} ({len(subset)/len(regime_df)*100:.1f}%)",
                f"{subset['volatility'].mean():.2f}",
                f"{subset['correlation'].mean():.2f}",
                f"{subset['return'].mean():.1%}"
            ])
        
        table = ax.table(
            cellText=stats_data,
            colLabels=['Regime', 'Days', 'Avg Vol', 'Avg Corr', 'Avg Ret'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        ax.axis('off')
        ax.set_title('Regime Statistics Summary', fontweight='bold', pad=20)
    
    @staticmethod
    def plot_lead_lag_network(
        lead_lag_df: pd.DataFrame,
        top_n: int = 50,
        output_path: str = "output/graphs/lead_lag_network.png"
    ):
        if lead_lag_df.empty:
            return
        
        PublicationVisualizations.set_publication_style()
        
        top_df = lead_lag_df.head(top_n)
        
        G = nx.DiGraph()
        
        for _, row in top_df.iterrows():
            if pd.notna(row.get('corr_leader')):
                G.add_edge(
                    row['corr_leader'],
                    row['corr_follower'],
                    weight=abs(row.get('correlation', 0)),
                    lag=row.get('corr_lag', 0)
                )
        
        if G.number_of_nodes() == 0:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(18, 16))
        
        pos = nx.spring_layout(G, k=3, iterations=50, seed=SEED)
        
        category_colors = {
            'tech_mega': '#3498db', 'finance': '#e74c3c', 'healthcare': '#2ecc71',
            'energy': '#f39c12', 'industrials': '#9b59b6', 'consumer': '#1abc9c',
            'sector_etfs': '#34495e', 'intl_developed': '#16a085', 'intl_emerging': '#d35400',
            'fixed_income': '#7f8c8d', 'commodities': '#f1c40f', 'volatility': '#c0392b',
            'broad_market': '#2980b9', 'unknown': '#bdc3c7'
        }
        
        node_colors = []
        for node in G.nodes():
            sym = node.replace('.US', '')
            cat = SYMBOL_CATEGORIES.get(sym, 'unknown')
            node_colors.append(category_colors.get(cat, '#bdc3c7'))
        
        out_deg = dict(G.out_degree())
        node_sizes = [400 + out_deg.get(n, 0) * 150 for n in G.nodes()]
        
        edges = G.edges(data=True)
        edge_widths = [e[2]['weight'] * 5 for e in edges]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, arrows=True,
                              arrowsize=20, edge_color='gray', ax=ax, connectionstyle='arc3,rad=0.1')
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                              alpha=0.9, edgecolors='black', linewidths=1.5, ax=ax)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        legend_elements = [Patch(facecolor=color, label=cat.replace('_', ' ').title())
                         for cat, color in category_colors.items() 
                         if any(SYMBOL_CATEGORIES.get(n.replace('.US', ''), '') == cat for n in G.nodes())]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        ax.set_title("Cross-Market Lead-Lag Network\n(Node Size = Influence, Color = Asset Category)",
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved network visualization to {output_path}")
        plt.close()
    
    @staticmethod
    def plot_learned_edges_analysis(
        model,
        graph_sequence: List[Data],
        symbols: List[str],
        output_path: str = "output/graphs/learned_edges_analysis.png"
    ):
        PublicationVisualizations.set_publication_style()
        
        model.eval()
        
        with torch.no_grad():
            output = model(graph_sequence, return_analysis=True)
        
        if 'analysis' not in output or not output['analysis']:
            return
        
        analysis = output['analysis']
        
        if 'edge_analyses' not in analysis or not analysis['edge_analyses']:
            return
        
        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        ax1 = fig.add_subplot(gs[0, 0])
        contagion_levels = [ea['contagion_level'].item() for ea in analysis['edge_analyses']]
        ax1.plot(contagion_levels, 'r-', linewidth=2, marker='o', markersize=3)
        ax1.fill_between(range(len(contagion_levels)), 0, contagion_levels, alpha=0.3, color='red')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Contagion Level')
        ax1.set_title('Learned Contagion Detection', fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        ax1.legend()
        
        ax2 = fig.add_subplot(gs[0, 1])
        num_edges = [ea['num_edges_kept'] for ea in analysis['edge_analyses']]
        ax2.bar(range(len(num_edges)), num_edges, color='steelblue', alpha=0.7)
        ax2.axhline(y=np.mean(num_edges), color='red', linestyle='--', label=f'Mean: {np.mean(num_edges):.1f}')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Number of Edges')
        ax2.set_title('Edge Sparsity Over Sequence', fontweight='bold')
        ax2.legend()
        
        ax3 = fig.add_subplot(gs[1, 0])
        regime_probs = np.array([ea['regime_probs'].cpu().numpy() for ea in analysis['edge_analyses']])
        
        ax3.stackplot(range(len(regime_probs)), regime_probs.T,
                     colors=['#2ecc71', '#f39c12', '#e74c3c'],
                     labels=['Bull', 'Normal', 'Crisis'], alpha=0.8)
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Probability')
        ax3.set_title('Learned Regime Detection', fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.set_ylim(0, 1)
        
        ax4 = fig.add_subplot(gs[1, 1])
        if 'attention_weights' in analysis and analysis['attention_weights'] is not None:
            attn = analysis['attention_weights'].cpu().numpy()[0]
            if len(attn.shape) > 2:
                attn = attn.mean(axis=0)
            
            im = ax4.imshow(attn, cmap='Blues', aspect='auto')
            ax4.set_xlabel('Key Timestep')
            ax4.set_ylabel('Query Timestep')
            ax4.set_title('Temporal Attention Weights', fontweight='bold')
            plt.colorbar(im, ax=ax4)
        
        ax5 = fig.add_subplot(gs[2, 0])
        all_weights = []
        for ea in analysis['edge_analyses']:
            if isinstance(ea['avg_edge_weight'], torch.Tensor):
                all_weights.append(ea['avg_edge_weight'].item())
            else:
                all_weights.append(ea['avg_edge_weight'])
        
        ax5.hist(all_weights, bins=20, color='purple', alpha=0.7, edgecolor='black')
        ax5.axvline(np.mean(all_weights), color='red', linestyle='--', label=f'Mean: {np.mean(all_weights):.3f}')
        ax5.set_xlabel('Average Edge Weight')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Edge Weight Distribution', fontweight='bold')
        ax5.legend()
        
        ax6 = fig.add_subplot(gs[2, 1])
        
        final_regime = output['regime_probs'].cpu().numpy().flatten()
        final_contagion = output['contagion_probability'].item()
        
        stats_text = f"""
        FINAL PREDICTIONS:
        
        Regime Probabilities:
          Bull/Low Vol: {final_regime[0]:.1%}
          Normal: {final_regime[1]:.1%}
          Crisis: {final_regime[2]:.1%}
        
        Predicted Regime: {PublicationVisualizations.REGIME_NAMES[np.argmax(final_regime)]}
        
        Contagion Risk: {final_contagion:.1%}
        
        SEQUENCE STATISTICS:
          Avg Contagion: {np.mean(contagion_levels):.1%}
          Avg Edges: {np.mean(num_edges):.1f}
          Edge Variance: {np.std(num_edges):.1f}
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax6.axis('off')
        ax6.set_title('Summary', fontweight='bold')
        
        plt.suptitle('RALEC: Learned Edge Construction Analysis', fontsize=16, fontweight='bold', y=1.02)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved learned edge analysis to {output_path}")
        plt.close()
    
    @staticmethod
    def plot_metrics_dashboard(
        metrics,
        cv_results: Dict[str, Any],
        output_path: str = "output/graphs/metrics_dashboard.png"
    ):
        PublicationVisualizations.set_publication_style()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        ax1 = axes[0, 0]
        overall_metrics = [
            ('Accuracy', metrics.accuracy),
            ('Balanced Acc', metrics.balanced_accuracy),
            ('Macro F1', metrics.macro_f1),
            ('Cohen κ', metrics.cohen_kappa),
            ('MCC', metrics.mcc),
        ]
        
        names, values = zip(*overall_metrics)
        colors = ['#3498db' if v > 0.5 else '#e74c3c' for v in values]
        bars = ax1.barh(names, values, color=colors, alpha=0.8)
        ax1.set_xlim(0, 1)
        ax1.set_title('Overall Metrics', fontweight='bold')
        ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        
        for bar, val in zip(bars, values):
            ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=10)
        
        ax2 = axes[0, 1]
        classes = ['Bull', 'Normal', 'Crisis']
        f1_scores = [metrics.f1_per_class.get(i, 0) for i in range(3)]
        colors = [PublicationVisualizations.REGIME_COLORS[i] for i in range(3)]
        
        ax2.bar(classes, f1_scores, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylim(0, 1)
        ax2.set_title('F1 Score by Regime', fontweight='bold')
        ax2.set_ylabel('F1 Score')
        
        for i, v in enumerate(f1_scores):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        ax3 = axes[0, 2]
        crisis_metrics = [
            ('Recall', metrics.crisis_recall),
            ('Precision', metrics.crisis_precision),
            ('F1', metrics.f1_per_class.get(2, 0))
        ]
        
        names, values = zip(*crisis_metrics)
        ax3.bar(names, values, color='#e74c3c', alpha=0.8, edgecolor='black')
        ax3.set_ylim(0, 1)
        ax3.set_title('Crisis Detection Performance', fontweight='bold')
        ax3.set_ylabel('Score')
        
        for i, v in enumerate(values):
            ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        ax4 = axes[1, 0]
        prob_metrics = [
            ('ROC-AUC', metrics.roc_auc_ovr),
            ('1 - Log Loss', max(0, 1 - metrics.log_loss_value)),
            ('1 - Brier', max(0, 1 - metrics.brier_score)),
            ('1 - ECE', max(0, 1 - metrics.expected_calibration_error))
        ]
        
        names, values = zip(*prob_metrics)
        ax4.bar(names, values, color='#9b59b6', alpha=0.8, edgecolor='black')
        ax4.set_ylim(0, 1)
        ax4.set_title('Probabilistic Metrics', fontweight='bold')
        ax4.set_ylabel('Score (higher = better)')
        
        ax5 = axes[1, 1]
        special_metrics = [
            ('Transition Acc', metrics.regime_transition_accuracy),
            ('Early Warning', metrics.early_warning_score)
        ]
        
        names, values = zip(*special_metrics)
        ax5.bar(names, values, color='#16a085', alpha=0.8, edgecolor='black')
        ax5.set_ylim(0, 1)
        ax5.set_title('Special Metrics', fontweight='bold')
        ax5.set_ylabel('Score')
        
        for i, v in enumerate(values):
            ax5.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        ax6 = axes[1, 2]
        
        summary = f"""
        MODEL PERFORMANCE SUMMARY
        
        Classification:
          Accuracy:        {metrics.accuracy:.1%} +/- {cv_results.get('std_val_acc', 0):.1%}
          Balanced Acc:    {metrics.balanced_accuracy:.1%}
          Macro F1:        {metrics.macro_f1:.3f}
        
        Reliability:
          Cohen's Kappa:   {metrics.cohen_kappa:.3f}
          MCC:             {metrics.mcc:.3f}
        
        Crisis Detection:
          Recall:          {metrics.crisis_recall:.1%} +/- {cv_results.get('std_crisis_recall', 0):.1%}
          Precision:       {metrics.crisis_precision:.1%}
        
        Calibration:
          ROC-AUC (OvR):   {metrics.roc_auc_ovr:.3f}
          ECE:             {metrics.expected_calibration_error:.4f}
        """
        
        ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        ax6.axis('off')
        
        plt.suptitle('Model Performance Dashboard', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metrics dashboard to {output_path}")
        plt.close()