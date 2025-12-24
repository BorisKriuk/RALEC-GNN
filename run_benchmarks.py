#!/usr/bin/env python3
"""
run_benchmarks.py - Run baseline model benchmarks separately
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import torch

from algorithm import ResearchConfig, DEVICE, SEED
from benchmarks import BaselineModels
from metrics import MetricsCalculator, RegimeMetricsReport
from visualizations import PublicationVisualizations
from run_common import (
    setup_logging, CacheManager, get_config_hash, DataPipeline
)

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

warnings.filterwarnings('ignore')

# Create output directories
for dir_name in ['output', 'output/graphs', 'output/baselines', 'output/metrics']:
    Path(dir_name).mkdir(parents=True, exist_ok=True)


def run_benchmarks(fast_mode: bool = False, include_hmm: bool = True, include_lstm: bool = True):
    """
    Run all baseline benchmarks.
    
    Args:
        fast_mode: If True, use faster but less accurate model configurations
        include_hmm: Include Hidden Markov Model baseline
        include_lstm: Include LSTM baseline
    """
    logger = setup_logging('benchmarks')
    
    logger.info("=" * 80)
    logger.info("BENCHMARK MODELS EVALUATION")
    logger.info("Cross-Market Contagion Networks - Baseline Comparison")
    logger.info("=" * 80)
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Fast mode: {fast_mode}")
    logger.info(f"Include HMM: {include_hmm}")
    logger.info(f"Include LSTM: {include_lstm}")
    
    # Initialize configuration
    config = ResearchConfig()
    config.save('output/config_benchmarks.json')
    config_hash = get_config_hash(config)
    
    logger.info(f"\nConfiguration hash: {config_hash}")
    logger.info(f"Regime labeling method: {config.regime_method}")
    logger.info(f"Number of CV splits: {config.n_splits}")
    
    # Prepare data
    pipeline = DataPipeline(config, logger)
    data_bundle = pipeline.prepare_all_data()
    
    if data_bundle is None:
        logger.error("Failed to prepare data. Exiting.")
        return None
    
    graphs = data_bundle['graphs']
    regime_df = data_bundle['regime_df']
    lead_lag_df = data_bundle['lead_lag_df']
    
    # ==========================================================================
    # Run Baseline Models
    # ==========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING BASELINE MODELS")
    logger.info(f"{'='*80}")
    
    baseline_runner = BaselineModels(config)
    X_baseline, y_baseline = baseline_runner.prepare_features(graphs, regime_df)
    
    if len(X_baseline) == 0:
        logger.error("No features extracted from graphs. Exiting.")
        return None
    
    logger.info(f"\nPrepared {len(X_baseline)} samples with {X_baseline.shape[1]} features")
    logger.info(f"Label distribution: {np.bincount(y_baseline.astype(int), minlength=3)}")
    
    # Train and evaluate all baselines
    baseline_results = baseline_runner.train_and_evaluate(
        X_baseline, 
        y_baseline, 
        n_splits=config.n_splits,
        fast_mode=fast_mode,
        include_hmm=include_hmm,
        include_lstm=include_lstm,
        include_gnn=True,  # Enable GNN baselines
        graphs=graphs,     # Pass graphs
        regime_df=regime_df  # Pass regime_df
    )
    
    if not baseline_results:
        logger.error("No baseline results obtained. Exiting.")
        return None
    
    # ==========================================================================
    # Generate Reports and Save Results
    # ==========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"GENERATING REPORTS")
    logger.info(f"{'='*80}")
    
    # Generate comparison report
    comparison_report = baseline_runner.generate_comparison_report(baseline_results)
    logger.info(comparison_report)
    
    # Save detailed results
    serializable_results = {}
    for name, metrics in baseline_results.items():
        serializable_results[name] = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in metrics.items()
            if k != 'fold_metrics'  # Exclude detailed fold metrics for main file
        }
    
    with open('output/baselines/baseline_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2, default=float)
    
    # Save detailed fold metrics separately
    detailed_results = {}
    for name, metrics in baseline_results.items():
        if 'fold_metrics' in metrics:
            detailed_results[name] = metrics['fold_metrics']
    
    with open('output/baselines/baseline_fold_details.json', 'w') as f:
        json.dump(detailed_results, f, indent=2, default=float)
    
    # Save comparison report as text
    with open('output/baselines/comparison_report.txt', 'w') as f:
        f.write(comparison_report)
    
    # ==========================================================================
    # Generate Visualizations
    # ==========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"GENERATING VISUALIZATIONS")
    logger.info(f"{'='*80}")
    
    # Plot lead-lag network
    if not lead_lag_df.empty:
        PublicationVisualizations.plot_lead_lag_network(
            lead_lag_df,
            output_path="output/graphs/lead_lag_network.png"
        )
    
    # Create baseline comparison visualization
    _plot_baseline_comparison(baseline_results, output_path="output/graphs/baseline_comparison.png")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"BENCHMARK EVALUATION COMPLETE")
    logger.info(f"{'='*80}")
    
    logger.info(f"\nOutputs saved to:")
    logger.info(f"   - output/baselines/baseline_results.json")
    logger.info(f"   - output/baselines/baseline_fold_details.json")
    logger.info(f"   - output/baselines/comparison_report.txt")
    logger.info(f"   - output/graphs/baseline_comparison.png")
    logger.info(f"   - output/graphs/lead_lag_network.png")
    
    # Print summary table
    logger.info(f"\n{'='*80}")
    logger.info(f"PERFORMANCE SUMMARY")
    logger.info(f"{'='*80}")
    
    # Sort by balanced accuracy
    sorted_results = sorted(
        baseline_results.items(),
        key=lambda x: x[1].get('balanced_accuracy_mean', 0),
        reverse=True
    )
    
    logger.info(f"\n{'Model':<25} {'Balanced Acc':<15} {'Crisis Recall':<15} {'Macro F1':<12} {'IC':<10}")
    logger.info("-" * 77)
    
    for name, metrics in sorted_results:
        ba = metrics.get('balanced_accuracy_mean', 0)
        ba_std = metrics.get('balanced_accuracy_std', 0)
        cr = metrics.get('crisis_recall_mean', 0)
        cr_std = metrics.get('crisis_recall_std', 0)
        f1 = metrics.get('macro_f1_mean', 0)
        ic = metrics.get('information_coefficient_mean', 0)
        
        logger.info(f"{name:<25} {ba:.2%}±{ba_std:.2%}   {cr:.2%}±{cr_std:.2%}   {f1:.3f}        {ic:.3f}")
    
    return baseline_results


def _plot_baseline_comparison(results: dict, output_path: str):
    """Create visualization comparing baseline models."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    PublicationVisualizations.set_publication_style()
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    models = list(results.keys())
    n_models = len(models)
    
    # Colors for models
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    # Plot 1: Accuracy metrics
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_to_plot = ['accuracy_mean', 'balanced_accuracy_mean']
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / n_models
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = [results[model].get(m, 0) for m in metrics_to_plot]
        stds = [results[model].get(m.replace('_mean', '_std'), 0) for m in metrics_to_plot]
        ax1.bar(x + i * width, values, width, label=model, color=color, yerr=stds, capsize=2)
    
    ax1.set_ylabel('Score')
    ax1.set_title('Accuracy Metrics', fontweight='bold')
    ax1.set_xticks(x + width * (n_models - 1) / 2)
    ax1.set_xticklabels(['Accuracy', 'Balanced Acc'])
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Crisis detection metrics
    ax2 = fig.add_subplot(gs[0, 1])
    metrics_to_plot = ['crisis_recall_mean', 'crisis_precision_mean', 'crisis_f1_mean']
    x = np.arange(len(metrics_to_plot))
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = [results[model].get(m, 0) for m in metrics_to_plot]
        stds = [results[model].get(m.replace('_mean', '_std'), 0) for m in metrics_to_plot]
        ax2.bar(x + i * width, values, width, label=model, color=color, yerr=stds, capsize=2)
    
    ax2.set_ylabel('Score')
    ax2.set_title('Crisis Detection Metrics', fontweight='bold')
    ax2.set_xticks(x + width * (n_models - 1) / 2)
    ax2.set_xticklabels(['Recall', 'Precision', 'F1'])
    ax2.set_ylim(0, 1)
    
    # Plot 3: Quant metrics
    ax3 = fig.add_subplot(gs[1, 0])
    metrics_to_plot = ['information_coefficient_mean', 'early_warning_score_mean', 'hit_rate_mean']
    x = np.arange(len(metrics_to_plot))
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = [results[model].get(m, 0) for m in metrics_to_plot]
        ax3.bar(x + i * width, values, width, label=model, color=color)
    
    ax3.set_ylabel('Score')
    ax3.set_title('Quantitative Finance Metrics', fontweight='bold')
    ax3.set_xticks(x + width * (n_models - 1) / 2)
    ax3.set_xticklabels(['Info Coef', 'Early Warning', 'Hit Rate'])
    ax3.set_ylim(0, 1)
    
    # Plot 4: Ranking by balanced accuracy
    ax4 = fig.add_subplot(gs[1, 1])
    
    sorted_models = sorted(models, key=lambda m: results[m].get('balanced_accuracy_mean', 0), reverse=True)
    y_pos = np.arange(len(sorted_models))
    values = [results[m].get('balanced_accuracy_mean', 0) for m in sorted_models]
    stds = [results[m].get('balanced_accuracy_std', 0) for m in sorted_models]
    
    bars = ax4.barh(y_pos, values, xerr=stds, capsize=3, color=[colors[models.index(m)] for m in sorted_models])
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(sorted_models)
    ax4.set_xlabel('Balanced Accuracy')
    ax4.set_title('Model Ranking', fontweight='bold')
    ax4.set_xlim(0, 1)
    
    # Add value labels
    for i, (v, s) in enumerate(zip(values, stds)):
        ax4.text(v + s + 0.02, i, f'{v:.2%}', va='center', fontsize=9)
    
    plt.suptitle('Baseline Model Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run baseline model benchmarks')
    parser.add_argument('--fast', action='store_true', help='Use fast mode with simpler models')
    parser.add_argument('--no-hmm', action='store_true', help='Exclude HMM baseline')
    parser.add_argument('--no-lstm', action='store_true', help='Exclude LSTM baseline')
    
    args = parser.parse_args()
    
    run_benchmarks(
        fast_mode=args.fast,
        include_hmm=not args.no_hmm,
        include_lstm=not args.no_lstm
    )