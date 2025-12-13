#!/usr/bin/env python3
"""
RALEC-GNN Runner Script
Execute different modes of the enhanced financial crisis prediction system

Usage:
    python run.py --mode train --data path/to/data
    python run.py --mode evaluate --checkpoint path/to/checkpoint
    python run.py --mode predict --data path/to/data
    python run.py --mode benchmark
    python run.py --mode demo
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import RALECGNN
from utils.config import RALECConfig
from benchmarks.benchmark_runner import BenchmarkRunner
from utils.demo import run_demo


def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['output', 'output/checkpoints', 'output/logs', 'output/reports']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def train_model(args):
    """Train RALEC-GNN model."""
    print("\n" + "="*60)
    print("RALEC-GNN TRAINING")
    print("="*60)
    
    # Load config
    config = RALECConfig()
    if args.config:
        config.load_from_file(args.config)
    
    # Override with command line args
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
        
    print(f"\nConfiguration:")
    print(f"  - Epochs: {config.epochs}")
    print(f"  - Batch Size: {config.batch_size}")
    print(f"  - Learning Rate: {config.learning_rate}")
    print(f"  - Hidden Dim: {config.hidden_dim}")
    print(f"  - Optimization: Mixed Precision={config.use_amp}")
    
    # Initialize model
    model = RALECGNN(config)
    
    # Load data
    if args.use_real_data:
        print("\nLoading real financial data from EODHD API...")
        print("⚠️  This may take several minutes on first run")
        
        from core.eodhd_data import get_real_financial_data
        try:
            train_data, val_data, test_data = get_real_financial_data(
                lookback_days=config.lookback_days if hasattr(config, 'lookback_days') else 1000,
                sequence_length=20
            )
            print(f"✅ Loaded real data: {len(train_data)} train, {len(val_data)} val sequences")
        except Exception as e:
            print(f"❌ Failed to load real data: {e}")
            print("Falling back to synthetic data...")
            from utils.synthetic_data import generate_synthetic_data
            train_data, val_data = generate_synthetic_data(
                num_samples=1000,
                num_assets=config.num_assets,
                sequence_length=20
            )
    else:
        print(f"\nLoading data from: {args.data if args.data else 'synthetic generator'}")
        
        # For demo, use synthetic data
        from utils.synthetic_data import generate_synthetic_data
        train_data, val_data = generate_synthetic_data(
            num_samples=1000,
            num_assets=config.num_assets,
            sequence_length=20
        )
        print("ℹ️  Using synthetic data. For real data, use --use-real-data flag")
    
    # Train
    print("\nStarting training...")
    start_time = datetime.now()
    
    results = model.train(train_data, val_data, epochs=config.epochs)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    # Print results
    print(f"\nTraining Complete!")
    print(f"  - Duration: {duration:.1f} minutes")
    print(f"  - Final Train Accuracy: {results['final_train_accuracy']:.4f}")
    print(f"  - Best Val Accuracy: {results['best_val_accuracy']:.4f}")
    
    # Save results
    results_path = f"output/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    return model


def evaluate_model(args):
    """Evaluate RALEC-GNN model."""
    print("\n" + "="*60)
    print("RALEC-GNN EVALUATION")
    print("="*60)
    
    # Load model from checkpoint
    print(f"\nLoading model from: {args.checkpoint}")
    model = RALECGNN.from_checkpoint(args.checkpoint)
    
    # Load test data
    if args.data:
        print(f"Loading test data from: {args.data}")
        # test_data = load_financial_data(args.data, split='test')
    else:
        # Use synthetic data for demo
        from utils.synthetic_data import generate_synthetic_data
        _, test_data = generate_synthetic_data(
            num_samples=200,
            num_assets=model.config.num_assets,
            sequence_length=20
        )
    
    # Evaluate
    print("\nRunning evaluation...")
    metrics = model.evaluate(test_data)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 40)
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Crisis Recall:   {metrics['crisis_recall']:.4f}")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  F1 Score:        {metrics['f1_score']:.4f}")
    print(f"  Lead Time:       {metrics['lead_time']:.1f} days")
    print(f"  Risk Detection:  {metrics['risk_detection_rate']:.4f}")
    
    return metrics


def predict_crisis(args):
    """Make predictions with RALEC-GNN."""
    print("\n" + "="*60)
    print("RALEC-GNN PREDICTION")
    print("="*60)
    
    # Load model
    if args.checkpoint:
        model = RALECGNN.from_checkpoint(args.checkpoint)
    else:
        # Use demo model
        print("No checkpoint provided, using demo model")
        config = RALECConfig()
        model = RALECGNN(config)
        model.is_trained = True  # Skip training check for demo
    
    # Load data for prediction
    if args.data:
        print(f"\nLoading data from: {args.data}")
        # data = load_financial_data(args.data)
    else:
        # Generate synthetic current market data
        from utils.synthetic_data import generate_current_market_state
        data = generate_current_market_state(num_assets=model.config.num_assets)
    
    # Make predictions
    print("\nGenerating predictions...")
    outputs = model.predict(data, return_risk_analysis=True)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    if 'predictions' in outputs:
        pred = outputs['predictions'].argmax().item()
        regimes = ['Normal', 'Bull', 'Bear', 'Crisis']
        confidence = outputs['predictions'].max().item()
        
        print(f"\nRegime Prediction: {regimes[pred]}")
        print(f"Confidence: {confidence:.2%}")
    
    if 'risk_analysis' in outputs:
        risk = outputs['risk_analysis']
        print(f"\nSystemic Risk Analysis:")
        print(f"  Overall Risk:        {risk['overall_risk']:.2%}")
        print(f"  Network Fragility:   {risk['network_fragility']:.2%}")
        print(f"  Cascade Probability: {risk['cascade_probability']:.2%}")
        print(f"  Herding Index:       {risk['herding_index']:.2%}")
        
        if risk['overall_risk'] > 0.7:
            print("\n⚠️  WARNING: HIGH SYSTEMIC RISK DETECTED!")
            print("   Recommended Actions:")
            print("   - Reduce portfolio concentration")
            print("   - Increase defensive positions")
            print("   - Monitor closely for next 48 hours")
    
    if 'alerts' in outputs and outputs['alerts']:
        print(f"\n🚨 ALERTS ({len(outputs['alerts'])}):")
        for alert in outputs['alerts']:
            print(f"   [{alert['level']}] {alert['message']}")
    
    return outputs


def run_benchmarks(args):
    """Run comprehensive benchmarks."""
    print("\n" + "="*60)
    print("RALEC-GNN BENCHMARKS")
    print("="*60)
    
    runner = BenchmarkRunner()
    
    # Run all benchmarks
    print("\nRunning benchmarks against baseline models...")
    results = runner.run_all_benchmarks()
    
    # Display summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    print("\nModel Comparison:")
    print("-" * 60)
    print(f"{'Model':<20} {'Accuracy':<10} {'Recall':<10} {'Lead Time':<10}")
    print("-" * 60)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<10.2%} "
              f"{metrics['recall']:<10.2%} {metrics['lead_time']:<10.1f}")
    
    # Save detailed report
    report_path = "output/benchmark_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed report saved to: {report_path}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RALEC-GNN Runner - Enhanced Financial Crisis Prediction"
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'evaluate', 'predict', 'benchmark', 'demo'],
        help='Execution mode'
    )
    
    # Data arguments
    parser.add_argument('--data', type=str, help='Path to data file/directory')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    
    # Other arguments
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--use-real-data', action='store_true', 
                       help='Use real EODHD data instead of synthetic (requires API key)')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_directories()
    
    # Set device
    if args.gpu and torch.cuda.is_available():
        import torch
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Execute based on mode
    if args.mode == 'train':
        train_model(args)
        
    elif args.mode == 'evaluate':
        if not args.checkpoint:
            print("Error: --checkpoint required for evaluation")
            sys.exit(1)
        evaluate_model(args)
        
    elif args.mode == 'predict':
        predict_crisis(args)
        
    elif args.mode == 'benchmark':
        run_benchmarks(args)
        
    elif args.mode == 'demo':
        print("\nRunning interactive demo...")
        run_demo()
    
    print("\n✓ Execution complete!")


if __name__ == "__main__":
    main()