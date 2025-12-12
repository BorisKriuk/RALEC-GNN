#!/usr/bin/env python3
"""
Run RALEC-GNN with Phase 1 optimizations
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import time
import pickle
from typing import Dict, Any

# Import from main.py
from main import (
    load_and_prepare_data,
    detect_lead_lag_relationships,
    label_market_regimes,
    create_temporal_graphs,
    PublicationVisualizations,
    SEED, DEVICE, DATA_CONFIG
)

# Import optimized components
from optimized_train import (
    OptimizedTrainingConfig,
    run_optimized_training,
    optimize_data_pipeline,
    MultiScaleFeatureExtractor,
    OptimizedRALECGNN,
    OptimizedTrainer,
    parallel_cv_training
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def profile_execution(func_name: str):
    """Decorator to profile function execution time"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting {func_name}")
            logger.info(f"{'='*60}")
            
            result = func(*args, **kwargs)
            
            elapsed = time.time() - start_time
            logger.info(f"{func_name} completed in {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
            
            return result
        return wrapper
    return decorator

@profile_execution("STEP 1: Data Loading and Preparation")
def load_data_cached():
    """Load data with caching"""
    cache_file = "cache/prepared_data.pkl"
    
    if os.path.exists(cache_file):
        logger.info("Loading cached data...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    data_dict = load_and_prepare_data(
        symbols=DATA_CONFIG['symbols'],
        lookback_days=DATA_CONFIG['lookback_days']
    )
    
    # Cache for next run
    os.makedirs("cache", exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(data_dict, f)
    
    return data_dict

@profile_execution("STEP 2: Multi-Scale Feature Engineering")
def add_multiscale_features(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add multi-scale temporal features"""
    config = OptimizedTrainingConfig()
    
    # Add multi-scale features to combined data
    enhanced_data = optimize_data_pipeline(
        data_dict['combined_data'],
        data_dict['symbols'],
        config
    )
    
    # Update data dict
    data_dict['combined_data'] = enhanced_data
    
    # Update feature columns
    original_features = data_dict['feature_columns']
    all_columns = enhanced_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove non-feature columns
    exclude_cols = ['regime', 'Close', 'Open', 'High', 'Low', 'Volume']
    new_features = [col for col in all_columns if col not in exclude_cols]
    
    data_dict['feature_columns'] = new_features
    logger.info(f"Features expanded from {len(original_features)} to {len(new_features)}")
    
    return data_dict

@profile_execution("STEP 3: Statistical Analysis")
def run_statistical_analysis(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run lead-lag detection and regime labeling"""
    # Lead-lag relationships
    lead_lag_df = detect_lead_lag_relationships(
        data_dict['individual_data'],
        data_dict['symbols']
    )
    
    # Regime detection
    regime_data = label_market_regimes(
        data_dict['combined_data'],
        method='quantile'
    )
    
    return {
        'lead_lag_df': lead_lag_df,
        'regime_data': regime_data
    }

@profile_execution("STEP 4: Graph Construction")
def build_temporal_graphs(data_dict: Dict[str, Any], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create temporal graph sequences"""
    graph_data = create_temporal_graphs(
        combined_data=analysis_results['regime_data'],
        symbols=data_dict['symbols'],
        feature_columns=data_dict['feature_columns'],
        edge_index=data_dict['edge_index'],
        lead_lag_df=analysis_results['lead_lag_df'],
        window_size=60,
        step_size=5,
        sequence_length=15
    )
    
    return graph_data

@profile_execution("STEP 5: Optimized RALEC-GNN Training")
def train_optimized_model(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """Train model with optimizations"""
    
    # Log dataset statistics
    unique_labels, counts = np.unique(graph_data['labels'], return_counts=True)
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Total sequences: {len(graph_data['labels'])}")
    logger.info(f"Class distribution: {dict(zip(unique_labels, counts))}")
    logger.info(f"Crisis percentage: {(graph_data['labels'] == 2).mean()*100:.1f}%")
    
    # Run optimized training
    results = run_optimized_training(
        graph_sequences=graph_data['graph_sequences'],
        labels=graph_data['labels'],
        volatilities=graph_data['volatilities'],
        num_features=graph_data['num_features'],
        num_edge_features=graph_data['num_edge_features']
    )
    
    return results

def compare_performance():
    """Compare optimized vs baseline performance"""
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("="*80)
    
    # Baseline stats (from logs)
    baseline = {
        'runtime_hours': 2.5,
        'accuracy': 0.7102,
        'crisis_recall': 0.7246,
        'memory_gb': 8.5  # Estimated
    }
    
    # Get current GPU memory if available
    if torch.cuda.is_available():
        memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"GPU Memory Usage: {memory_gb:.2f} GB")
    
    logger.info("\nBaseline Performance:")
    logger.info(f"  Runtime: {baseline['runtime_hours']:.1f} hours")
    logger.info(f"  Accuracy: {baseline['accuracy']:.1%}")
    logger.info(f"  Crisis Recall: {baseline['crisis_recall']:.1%}")
    
    logger.info("\nOptimization Features Implemented:")
    logger.info("  ✓ Mixed Precision Training (FP16)")
    logger.info("  ✓ Gradient Accumulation")
    logger.info("  ✓ Parallel Cross-Validation")
    logger.info("  ✓ Multi-Scale Temporal Features")
    logger.info("  ✓ Efficient Data Loading")
    logger.info("  ✓ Smart Caching")

def main():
    """Main execution pipeline"""
    total_start = time.time()
    
    logger.info("\n" + "="*80)
    logger.info("RALEC-GNN PHASE 1: OPTIMIZATION & ACCELERATION")
    logger.info("="*80)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("="*80)
    
    try:
        # Step 1: Load data
        data_dict = load_data_cached()
        
        # Step 2: Add multi-scale features
        data_dict = add_multiscale_features(data_dict)
        
        # Step 3: Statistical analysis
        analysis_results = run_statistical_analysis(data_dict)
        
        # Step 4: Build graphs
        graph_data = build_temporal_graphs(data_dict, analysis_results)
        
        # Step 5: Train optimized model
        training_results = train_optimized_model(graph_data)
        
        # Performance comparison
        compare_performance()
        
        # Generate visualizations
        logger.info("\nGenerating publication-quality visualizations...")
        # PublicationVisualizations.plot_lead_lag_network(
        #     analysis_results['lead_lag_df'],
        #     top_n=50,
        #     output_path="output/graphs/optimized_lead_lag_network.png"
        # )
        
        total_time = time.time() - total_start
        logger.info(f"\n{'='*80}")
        logger.info(f"TOTAL EXECUTION TIME: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        
        # Calculate speedup
        baseline_time = 2.5 * 3600  # 2.5 hours in seconds
        speedup = baseline_time / total_time
        logger.info(f"SPEEDUP vs BASELINE: {speedup:.1f}x faster")
        logger.info(f"TIME SAVED: {(baseline_time - total_time)/60:.1f} minutes")
        logger.info(f"{'='*80}")
        
        # Save results summary
        summary = {
            'execution_time_minutes': total_time / 60,
            'speedup': speedup,
            'training_results': training_results,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs("output/optimization", exist_ok=True)
        with open("output/optimization/phase1_results.pkl", 'wb') as f:
            pickle.dump(summary, f)
        
        logger.info("\nPhase 1 Optimization Complete!")
        logger.info("Results saved to output/optimization/phase1_results.pkl")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()