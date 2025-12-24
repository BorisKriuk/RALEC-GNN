#!/usr/bin/env python3
"""
run_model.py - Run RALEC-GNN model training and evaluation separately
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import torch

from algorithm import (
    ResearchConfig, TemporalGNNWithLearnedEdges, ResearchTrainer,
    DEVICE, SEED
)
from metrics import MetricsCalculator, RegimeMetricsReport, QuantMetrics
from visualizations import PublicationVisualizations
from run_common import (
    setup_logging, CacheManager, get_config_hash, DataPipeline
)

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

warnings.filterwarnings('ignore')

# Create output directories
for dir_name in ['output', 'output/graphs', 'output/models', 'output/metrics']:
    Path(dir_name).mkdir(parents=True, exist_ok=True)


def run_model(compare_with_baselines: bool = True):
    """
    Run RALEC-GNN model training and evaluation.
    
    Args:
        compare_with_baselines: If True, load and compare with baseline results
    """
    logger = setup_logging('model')
    
    logger.info("=" * 80)
    logger.info("RALEC-GNN MODEL TRAINING AND EVALUATION")
    logger.info("Cross-Market Contagion Networks - Regime-Adaptive Learned Edge Construction")
    logger.info("=" * 80)
    logger.info(f"Using device: {DEVICE}")
    
    # Initialize configuration
    config = ResearchConfig()
    config.save('output/config_model.json')
    config_hash = get_config_hash(config)
    
    logger.info(f"\nConfiguration hash: {config_hash}")
    logger.info(f"Regime labeling method: {config.regime_method}")
    logger.info(f"Using learned edges: {config.use_learned_edges}")
    logger.info(f"Hidden dimension: {config.hidden_dim}")
    logger.info(f"Number of GNN layers: {config.num_layers}")
    logger.info(f"Sequence length: {config.seq_len}")
    logger.info(f"Crisis weight multiplier: {config.crisis_weight_multiplier}")
    logger.info(f"Number of CV splits: {config.n_splits}")
    
    # Prepare data
    pipeline = DataPipeline(config, logger)
    data_bundle = pipeline.prepare_all_data()
    
    if data_bundle is None:
        logger.error("Failed to prepare data. Exiting.")
        return None
    
    graphs = data_bundle['graphs']
    regime_df = data_bundle['regime_df']
    symbols_list = data_bundle['symbols_list']
    lead_lag_df = data_bundle['lead_lag_df']
    
    # ==========================================================================
    # Initialize and Train Model
    # ==========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING RALEC-GNN MODEL")
    logger.info(f"{'='*80}")
    
    model = TemporalGNNWithLearnedEdges(config)
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"\nModel Architecture:")
    logger.info(f"   Total parameters: {num_params:,}")
    logger.info(f"   Trainable parameters: {trainable_params:,}")
    logger.info(f"   Edge constructor params: {sum(p.numel() for p in model.edge_constructor.parameters()):,}")
    logger.info(f"   GNN layers params: {sum(p.numel() for layer in model.gnn_layers for p in layer.parameters()):,}")
    
    # Train with cross-validation
    trainer = ResearchTrainer(model, config)
    cv_results = trainer.train_with_cross_validation(graphs, regime_df)
    
    if not cv_results or 'overall_metrics' not in cv_results:
        logger.error("Training failed or no results obtained. Exiting.")
        return None
    
    overall_metrics = cv_results['overall_metrics']
    
    # ==========================================================================
    # Generate Detailed Metrics Report
    # ==========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"DETAILED METRICS REPORT")
    logger.info(f"{'='*80}")
    
    # Generate comprehensive report
    report = RegimeMetricsReport.generate_report(overall_metrics, "RALEC-GNN")
    logger.info(report)
    
    # Save report to file
    with open('output/metrics/ralec_gnn_report.txt', 'w') as f:
        f.write(report)
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"SAVING RESULTS")
    logger.info(f"{'='*80}")
    
    # Save CV results
    serializable_metrics = {
        'avg_val_acc': float(cv_results['avg_val_acc']),
        'std_val_acc': float(cv_results['std_val_acc']),
        'avg_val_loss': float(cv_results['avg_val_loss']),
        'avg_crisis_recall': float(cv_results['avg_crisis_recall']),
        'std_crisis_recall': float(cv_results['std_crisis_recall']),
        'n_folds': len(cv_results['fold_results']),
        'overall_metrics': overall_metrics.to_dict()
    }
    
    with open('output/metrics/cv_results.json', 'w') as f:
        json.dump(serializable_metrics, f, indent=2, default=float)
    
    # Save fold-level results
    fold_details = []
    for i, fold_result in enumerate(cv_results['fold_results']):
        fold_details.append({
            'fold': i + 1,
            'val_acc': fold_result['best_val_acc'],
            'val_loss': fold_result['best_val_loss'],
            'crisis_recall': fold_result['best_crisis_recall']
        })
    
    with open('output/metrics/fold_details.json', 'w') as f:
        json.dump(fold_details, f, indent=2)
    
    # ==========================================================================
    # Generate Visualizations
    # ==========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"GENERATING VISUALIZATIONS")
    logger.info(f"{'='*80}")
    
    # Load baseline results if available for comparison
    baseline_results = {}
    if compare_with_baselines:
        try:
            with open('output/baselines/baseline_results.json', 'r') as f:
                baseline_results = json.load(f)
            logger.info("Loaded baseline results for comparison")
        except FileNotFoundError:
            logger.warning("Baseline results not found. Run run_benchmarks.py first for comparison.")
    
    # Plot comprehensive results
    PublicationVisualizations.plot_comprehensive_results(
        regime_df, cv_results, baseline_results,
        output_dir="output/graphs"
    )
    
    # Plot metrics dashboard
    PublicationVisualizations.plot_metrics_dashboard(
        overall_metrics, cv_results,
        output_path="output/graphs/metrics_dashboard.png"
    )
    
    # Plot lead-lag network
    if not lead_lag_df.empty:
        PublicationVisualizations.plot_lead_lag_network(
            lead_lag_df,
            output_path="output/graphs/lead_lag_network.png"
        )
    
    # ==========================================================================
    # Current Market State Analysis
    # ==========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"CURRENT MARKET STATE ANALYSIS")
    logger.info(f"{'='*80}")
    
    if len(graphs) >= config.seq_len:
        # Load best model
        try:
            model.load_state_dict(torch.load('output/models/best_crisis_model.pt', map_location=DEVICE))
            logger.info("Loaded best crisis detection model")
        except FileNotFoundError:
            best_fold = len(cv_results.get('fold_results', [])) - 1
            if best_fold >= 0:
                try:
                    model.load_state_dict(torch.load(f'output/models/best_model_fold{best_fold}.pt', map_location=DEVICE))
                    logger.info(f"Loaded best model from fold {best_fold + 1}")
                except FileNotFoundError:
                    logger.warning("Could not load saved model. Using current model state.")
        
        model.eval()
        latest_seq = graphs[-config.seq_len:]
        
        with torch.no_grad():
            output = model(latest_seq, return_analysis=True)
        
        probs = output['regime_probs'].cpu().numpy().flatten()
        regime = np.argmax(probs)
        regime_names = {0: 'Bull/Low Vol', 1: 'Normal/Bear', 2: 'Crisis/Contagion'}
        
        logger.info(f"\nCURRENT MARKET PREDICTION:")
        logger.info(f"   Predicted Regime: {regime_names[regime]}")
        logger.info(f"\n   Regime Probabilities:")
        for i, (name, prob) in enumerate(zip(regime_names.values(), probs)):
            bar = "█" * int(prob * 40)
            logger.info(f"      {name:20s}: {prob:6.1%} {bar}")
        
        contagion_prob = output['contagion_probability'].item()
        vol_forecast = output['volatility_forecast'].item()
        
        logger.info(f"\n   Contagion Risk: {contagion_prob:.1%}")
        logger.info(f"   Volatility Forecast: {vol_forecast:.2f}")
        
        # Plot learned edges analysis
        PublicationVisualizations.plot_learned_edges_analysis(
            model, latest_seq, symbols_list,
            output_path="output/graphs/learned_edges_analysis.png"
        )
        
        # Save current prediction
        current_prediction = {
            'timestamp': str(graphs[-1].timestamp),
            'predicted_regime': int(regime),
            'regime_name': regime_names[regime],
            'probabilities': {regime_names[i]: float(p) for i, p in enumerate(probs)},
            'contagion_risk': float(contagion_prob),
            'volatility_forecast': float(vol_forecast)
        }
        
        with open('output/metrics/current_prediction.json', 'w') as f:
            json.dump(current_prediction, f, indent=2)
    
    # ==========================================================================
    # Comparison with Baselines (if available)
    # ==========================================================================
    if baseline_results:
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPARISON WITH BASELINES")
        logger.info(f"{'='*80}")
        
        # Add RALEC-GNN to comparison
        gnn_result = {
            'accuracy_mean': overall_metrics.accuracy,
            'balanced_accuracy_mean': overall_metrics.balanced_accuracy,
            'macro_f1_mean': overall_metrics.macro_f1,
            'crisis_recall_mean': overall_metrics.crisis_recall,
            'early_warning_score_mean': overall_metrics.early_warning_score,
            'information_coefficient_mean': overall_metrics.information_coefficient,
            'roc_auc_ovr_mean': overall_metrics.roc_auc_ovr,
            'false_alarm_rate_mean': overall_metrics.false_alarm_rate,
        }
        
        all_results = {'RALEC-GNN': gnn_result, **baseline_results}
        
        # Print comparison table
        logger.info(f"\n{'Model':<25} {'Balanced Acc':<15} {'Crisis Recall':<15} {'Early Warning':<15} {'IC':<10}")
        logger.info("-" * 80)
        
        sorted_results = sorted(
            all_results.items(),
            key=lambda x: x[1].get('balanced_accuracy_mean', 0),
            reverse=True
        )
        
        for name, metrics in sorted_results:
            ba = metrics.get('balanced_accuracy_mean', 0)
            cr = metrics.get('crisis_recall_mean', 0)
            ew = metrics.get('early_warning_score_mean', 0)
            ic = metrics.get('information_coefficient_mean', 0)
            
            marker = " ***" if name == 'RALEC-GNN' else ""
            logger.info(f"{name:<25} {ba:.2%}            {cr:.2%}            {ew:.2%}            {ic:.3f}{marker}")
        
        # Calculate improvement over best baseline
        best_baseline_ba = max(
            v.get('balanced_accuracy_mean', 0) 
            for k, v in baseline_results.items()
        )
        improvement = overall_metrics.balanced_accuracy - best_baseline_ba
        
        logger.info(f"\nRALEC-GNN improvement over best baseline: {improvement:+.2%}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"MODEL TRAINING COMPLETE")
    logger.info(f"{'='*80}")
    
    logger.info(f"\nOutputs saved to:")
    logger.info(f"   - output/metrics/cv_results.json")
    logger.info(f"   - output/metrics/fold_details.json")
    logger.info(f"   - output/metrics/ralec_gnn_report.txt")
    logger.info(f"   - output/metrics/current_prediction.json")
    logger.info(f"   - output/graphs/comprehensive_results.png")
    logger.info(f"   - output/graphs/metrics_dashboard.png")
    logger.info(f"   - output/graphs/learned_edges_analysis.png")
    logger.info(f"   - output/models/best_crisis_model.pt")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL PERFORMANCE SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"\n  RALEC-GNN Model:")
    logger.info(f"    Accuracy:            {overall_metrics.accuracy:.2%} ± {cv_results.get('std_val_acc', 0):.2%}")
    logger.info(f"    Balanced Accuracy:   {overall_metrics.balanced_accuracy:.2%}")
    logger.info(f"    Macro F1:            {overall_metrics.macro_f1:.3f}")
    logger.info(f"    Cohen's Kappa:       {overall_metrics.cohen_kappa:.3f}")
    logger.info(f"    MCC:                 {overall_metrics.mcc:.3f}")
    logger.info(f"    Crisis Recall:       {overall_metrics.crisis_recall:.2%} ± {cv_results.get('std_crisis_recall', 0):.2%}")
    logger.info(f"    Crisis Precision:    {overall_metrics.crisis_precision:.2%}")
    logger.info(f"    ROC-AUC:             {overall_metrics.roc_auc_ovr:.3f}")
    logger.info(f"    Early Warning:       {overall_metrics.early_warning_score:.2%}")
    logger.info(f"    Info Coefficient:    {overall_metrics.information_coefficient:.3f}")
    logger.info(f"    Hit Rate:            {overall_metrics.hit_rate:.2%}")
    logger.info(f"    False Alarm Rate:    {overall_metrics.false_alarm_rate:.2%}")
    
    return cv_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RALEC-GNN model training')
    parser.add_argument('--no-compare', action='store_true', 
                       help='Skip comparison with baseline results')
    
    args = parser.parse_args()
    
    run_model(compare_with_baselines=not args.no_compare)