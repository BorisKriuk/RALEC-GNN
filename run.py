#!/usr/bin/env python3
"""
Optimized run.py with parallel data fetching, caching, and improved training.
"""

import os
import logging
import json
import warnings
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

from algorithm import (
    ResearchConfig, EODHDClient, StatisticalLeadLagDetector,
    MultiMethodGraphBuilder, RegimeLabeler, TemporalGNNWithLearnedEdges,
    ResearchTrainer, ALL_SYMBOLS, SYMBOL_CATEGORIES, DEVICE, API_KEY, SEED
)
from benchmarks import BaselineModels
from visualizations import PublicationVisualizations

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

for dir_name in ['output', 'output/graphs', 'output/models', 'output/metrics', 'output/baselines', 'cache']:
    Path(dir_name).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/research.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

logger.info(f"Using device: {DEVICE}")


# =============================================================================
# OPTIMIZATION 1: Parallel Data Fetching
# =============================================================================

def fetch_single_symbol(client, sym, start_date, end_date, min_observations):
    """Fetch data for a single symbol. Used by thread pool."""
    try:
        df = client.get_eod_data(sym, 'US', start_date, end_date)
        if not df.empty and 'adjusted_close' in df.columns and len(df) >= min_observations:
            return sym, df, True
        return sym, None, False
    except Exception as e:
        return sym, None, False


def fetch_all_data_parallel(client, symbols, start_date, end_date, min_observations, max_workers=10):
    """Parallel data fetching with ThreadPoolExecutor."""
    data = {}
    failed_symbols = []
    completed = 0
    total = len(symbols)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_single_symbol, client, sym, start_date, end_date, min_observations): sym 
            for sym in symbols
        }
        
        for future in as_completed(futures):
            sym, df, success = future.result()
            completed += 1
            
            if success:
                data[f"{sym}.US"] = df
            else:
                failed_symbols.append(sym)
            
            if completed % 20 == 0 or completed == total:
                logger.info(f"  Progress: {completed}/{total} - Loaded {len(data)} symbols")
    
    return data, failed_symbols


# =============================================================================
# OPTIMIZATION 2: Caching Infrastructure
# =============================================================================

class CacheManager:
    """Manages caching for expensive computations."""
    
    def __init__(self, cache_dir='cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_hash(self, *args):
        """Generate hash from arguments."""
        key_str = str(args)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def get_data_cache_path(self, symbols, start_date, end_date):
        """Get cache path for market data."""
        key = self._get_hash(sorted(symbols), start_date, end_date)
        return self.cache_dir / f"data_{key}.pkl"
    
    def get_graphs_cache_path(self, symbols, config_hash):
        """Get cache path for graphs."""
        key = self._get_hash(sorted(symbols), config_hash)
        return self.cache_dir / f"graphs_{key}.pkl"
    
    def get_leadlag_cache_path(self, symbols, config_hash):
        """Get cache path for lead-lag results."""
        key = self._get_hash(sorted(symbols), config_hash)
        return self.cache_dir / f"leadlag_{key}.pkl"
    
    def load(self, path):
        """Load from cache if exists."""
        if path.exists():
            logger.info(f"Loading from cache: {path}")
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return None
        return None
    
    def save(self, path, data):
        """Save to cache."""
        logger.info(f"Saving to cache: {path}")
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


def get_config_hash(config):
    """Get a hash representing the configuration."""
    config_dict = {
        'window_size': config.window_size,
        'step_size': config.step_size,
        'edge_threshold': config.edge_threshold,
        'max_lag': config.max_lag,
        'seq_len': config.seq_len,
        'node_features': config.node_features,
    }
    return hashlib.md5(str(config_dict).encode()).hexdigest()[:12]


# =============================================================================
# OPTIMIZATION 3: Fast Lead-Lag Detection (FFT-based)
# =============================================================================

from scipy import signal

class FastLeadLagDetector:
    """FFT-based lead-lag detection - much faster than loop-based."""
    
    def __init__(self, config):
        self.max_lag = config.max_lag
        self.min_correlation = config.significance_threshold
    
    def detect_relationships_fast(self, returns_dict, top_k=100):
        """Vectorized lead-lag using FFT cross-correlation."""
        symbols = list(returns_dict.keys())
        n = len(symbols)
        
        # Align all series to common index
        df = pd.DataFrame(returns_dict)
        df = df.dropna()
        
        if len(df) < 50:
            logger.warning("Insufficient data for lead-lag detection")
            return pd.DataFrame()
        
        returns_matrix = df.values  # (T, n_assets)
        
        # Standardize
        means = returns_matrix.mean(axis=0, keepdims=True)
        stds = returns_matrix.std(axis=0, keepdims=True) + 1e-8
        returns_matrix = (returns_matrix - means) / stds
        
        results = []
        T = len(returns_matrix)
        
        # Only compute upper triangle
        for i in range(n):
            x = returns_matrix[:, i]
            
            for j in range(i + 1, n):
                y = returns_matrix[:, j]
                
                # FFT-based cross-correlation (much faster than loop)
                corr = signal.correlate(x, y, mode='full', method='fft')
                corr = corr / T  # Normalize
                
                mid = len(corr) // 2
                lag_start = max(0, mid - self.max_lag)
                lag_end = min(len(corr), mid + self.max_lag + 1)
                corr_segment = corr[lag_start:lag_end]
                
                if len(corr_segment) == 0:
                    continue
                
                best_idx = np.argmax(np.abs(corr_segment))
                best_lag = best_idx - (mid - lag_start)
                best_corr = corr_segment[best_idx]
                
                if abs(best_corr) >= self.min_correlation:
                    if best_lag > 0:
                        leader, follower = symbols[i], symbols[j]
                    else:
                        leader, follower = symbols[j], symbols[i]
                    
                    results.append({
                        'corr_leader': leader,
                        'corr_follower': follower,
                        'correlation': abs(best_corr),
                        'corr_lag': abs(best_lag)
                    })
        
        # Sort and return top_k
        results.sort(key=lambda x: -x['correlation'])
        return pd.DataFrame(results[:top_k])


# =============================================================================
# MAIN FUNCTION (OPTIMIZED)
# =============================================================================

def main():
    logger.info("=" * 80)
    logger.info("Cross-Market Contagion Networks - OPTIMIZED Implementation")
    logger.info("NOVEL: Regime-Adaptive Learned Edge Construction (RALEC)")
    logger.info("=" * 80)
    
    if not API_KEY:
        logger.error("API key not found! Please set EODHD_API_KEY in .env file")
        return
    
    config = ResearchConfig()
    config.save('output/config.json')
    config_hash = get_config_hash(config)
    
    logger.info(f"\nConfiguration saved to output/config.json")
    logger.info(f"Config hash: {config_hash}")
    logger.info(f"Regime labeling method: {config.regime_method}")
    logger.info(f"Using learned edges: {config.use_learned_edges}")
    logger.info(f"Step size: {config.step_size}")
    logger.info(f"Node features: {config.node_features}")
    logger.info(f"Crisis weight multiplier: {config.crisis_weight_multiplier}")
    
    cache = CacheManager()
    client = EODHDClient(API_KEY)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=config.lookback_days)).strftime('%Y-%m-%d')
    
    # =========================================================================
    # STEP 1: Data Collection (OPTIMIZED - Parallel + Cached)
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 1: Data Collection ({len(ALL_SYMBOLS)} symbols) [OPTIMIZED]")
    logger.info(f"{'='*80}")
    
    data_cache_path = cache.get_data_cache_path(ALL_SYMBOLS, start_date, end_date)
    cached_data = cache.load(data_cache_path)
    
    if cached_data is not None:
        data, failed_symbols = cached_data
        logger.info(f"Loaded {len(data)} assets from cache")
    else:
        logger.info(f"Fetching data from {start_date} to {end_date} (parallel)...")
        data, failed_symbols = fetch_all_data_parallel(
            client, ALL_SYMBOLS, start_date, end_date, 
            config.min_observations, max_workers=10
        )
        cache.save(data_cache_path, (data, failed_symbols))
    
    logger.info(f"\nSuccessfully loaded {len(data)} assets")
    if failed_symbols:
        logger.info(f"Failed to load: {len(failed_symbols)} symbols")
    
    if len(data) < 10:
        logger.error(f"Insufficient assets: {len(data)} < 10")
        return
    
    # Build returns dict
    returns_dict = {}
    for sym, df in data.items():
        prices = df['adjusted_close']
        returns = np.log(prices / prices.shift(1)).dropna()
        if len(returns) >= config.min_observations:
            returns_dict[sym] = returns
    
    logger.info(f"Working with {len(returns_dict)} assets for analysis")
    
    category_counts = defaultdict(int)
    for sym in returns_dict.keys():
        cat = SYMBOL_CATEGORIES.get(sym.replace('.US', ''), 'unknown')
        category_counts[cat] += 1
    
    logger.info("\nAsset category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        logger.info(f"   {cat}: {count}")
    
    # =========================================================================
    # STEP 2: Lead-Lag Detection (OPTIMIZED - FFT + Cached)
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 2: Statistical Lead-Lag Detection [OPTIMIZED - FFT]")
    logger.info(f"{'='*80}")
    
    leadlag_cache_path = cache.get_leadlag_cache_path(list(returns_dict.keys()), config_hash)
    lead_lag_df = cache.load(leadlag_cache_path)
    
    if lead_lag_df is None:
        # Use fast FFT-based detector
        fast_detector = FastLeadLagDetector(config)
        lead_lag_df = fast_detector.detect_relationships_fast(returns_dict, top_k=100)
        cache.save(leadlag_cache_path, lead_lag_df)
    
    if not lead_lag_df.empty:
        logger.info(f"\nTOP MARKET LEADERS:")
        if 'corr_leader' in lead_lag_df.columns:
            leaders = lead_lag_df['corr_leader'].value_counts().head(10)
            for leader, count in leaders.items():
                logger.info(f"   {leader}: influences {count} assets")
        
        PublicationVisualizations.plot_lead_lag_network(lead_lag_df)
    
    # =========================================================================
    # STEP 3: Regime Detection & Labeling
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 3: Regime Detection & Labeling")
    logger.info(f"{'='*80}")
    
    labeler = RegimeLabeler(config)
    regime_df = labeler.label_regimes(returns_dict, method=config.regime_method)
    
    logger.info(f"\nRegime distribution:")
    regime_counts = regime_df['regime'].value_counts()
    for regime in sorted(regime_counts.index):
        pct = regime_counts[regime] / len(regime_df) * 100
        name = {0: 'Bull/Low Vol', 1: 'Normal/Bear', 2: 'Crisis/Contagion'}[regime]
        logger.info(f"   {name}: {pct:.1f}% ({regime_counts[regime]} days)")
    
    # =========================================================================
    # STEP 4: Temporal Graph Construction (CACHED)
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 4: Temporal Graph Construction [CACHED]")
    logger.info(f"{'='*80}")
    
    graphs_cache_path = cache.get_graphs_cache_path(list(returns_dict.keys()), config_hash)
    cached_graphs = cache.load(graphs_cache_path)
    
    if cached_graphs is not None:
        graphs, symbols_list = cached_graphs
        logger.info(f"Loaded {len(graphs)} graphs from cache")
    else:
        graph_builder = MultiMethodGraphBuilder(config)
        graphs, symbols_list = graph_builder.build_temporal_graphs(
            returns_dict,
            use_full_graph=config.use_learned_edges
        )
        if graphs:
            cache.save(graphs_cache_path, (graphs, symbols_list))
    
    if not graphs:
        logger.error("Failed to build graphs")
        return
    
    avg_nodes = np.mean([g.num_nodes for g in graphs])
    avg_edges = np.mean([g.edge_index.shape[1] for g in graphs])
    logger.info(f"\nGraph statistics:")
    logger.info(f"   Number of graphs: {len(graphs)}")
    logger.info(f"   Average nodes: {avg_nodes:.1f}")
    logger.info(f"   Average edges: {avg_edges:.1f}")
    
    # =========================================================================
    # STEP 5a: Training Baseline Models
    # =========================================================================
    baseline_results = {}
    if config.run_baselines:
        logger.info(f"\n{'='*80}")
        logger.info(f"STEP 5a: Training Baseline Models")
        logger.info(f"{'='*80}")
        
        baseline_runner = BaselineModels(config)
        X_baseline, y_baseline = baseline_runner.prepare_features(graphs, regime_df)
        
        if len(X_baseline) > 0:
            baseline_results = baseline_runner.train_and_evaluate(
                X_baseline, y_baseline, n_splits=config.n_splits
            )
            
            with open('output/baselines/baseline_results.json', 'w') as f:
                json.dump(baseline_results, f, indent=2, default=float)
    
    # =========================================================================
    # STEP 5b: Training RALEC-GNN Model
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 5b: Training RALEC-GNN Model")
    logger.info(f"{'='*80}")
    
    model = TemporalGNNWithLearnedEdges(config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    trainer = ResearchTrainer(model, config)
    cv_results = trainer.train_with_cross_validation(graphs, regime_df)
    
    # =========================================================================
    # STEP 6: Generating Visualizations
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 6: Generating Visualizations")
    logger.info(f"{'='*80}")
    
    if cv_results and 'overall_metrics' in cv_results:
        PublicationVisualizations.plot_comprehensive_results(
            regime_df, cv_results, baseline_results
        )
        
        PublicationVisualizations.plot_metrics_dashboard(
            cv_results['overall_metrics'], cv_results
        )
    
    # =========================================================================
    # STEP 7: Current Market State Analysis
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 7: Current Market State Analysis")
    logger.info(f"{'='*80}")
    
    if len(graphs) >= config.seq_len and cv_results:
        try:
            model.load_state_dict(torch.load('output/models/best_crisis_model.pt', map_location=DEVICE))
            logger.info("Loaded best crisis detection model")
        except:
            best_fold = len(cv_results.get('fold_results', [])) - 1
            if best_fold >= 0:
                try:
                    model.load_state_dict(torch.load(f'output/models/best_model_fold{best_fold}.pt', map_location=DEVICE))
                    logger.info(f"Loaded best model from fold {best_fold + 1}")
                except:
                    pass
        
        model.eval()
        latest_seq = graphs[-config.seq_len:]
        
        with torch.no_grad():
            output = model(latest_seq, return_analysis=True)
        
        probs = output['regime_probs'].cpu().numpy().flatten()
        regime = np.argmax(probs)
        regime_names = {0: 'Bull/Low Vol', 1: 'Normal/Bear', 2: 'Crisis/Contagion'}
        
        logger.info(f"\nPREDICTED REGIME: {regime_names[regime]}")
        logger.info(f"\nProbabilities:")
        for i, (name, prob) in enumerate(zip(regime_names.values(), probs)):
            bar = "█" * int(prob * 40)
            logger.info(f"   {name:20s}: {prob:6.1%} {bar}")
        
        contagion_prob = output['contagion_probability'].item()
        vol_forecast = output['volatility_forecast'].item()
        
        logger.info(f"\nContagion Risk: {contagion_prob:.1%}")
        logger.info(f"Volatility Forecast: {vol_forecast:.2f}")
        
        PublicationVisualizations.plot_learned_edges_analysis(
            model, latest_seq, symbols_list
        )
    
    # =========================================================================
    # COMPLETE
    # =========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"PIPELINE COMPLETE")
    logger.info(f"{'='*80}")
    
    logger.info(f"\nOutputs saved to output/ directory:")
    logger.info(f"   - output/graphs/comprehensive_results.png")
    logger.info(f"   - output/graphs/metrics_dashboard.png")
    logger.info(f"   - output/graphs/lead_lag_network.png")
    logger.info(f"   - output/graphs/learned_edges_analysis.png")
    logger.info(f"   - output/metrics/cv_results.json")
    logger.info(f"   - output/baselines/baseline_results.json")
    
    if cv_results and 'overall_metrics' in cv_results:
        metrics = cv_results['overall_metrics']
        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL PERFORMANCE SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"\n  RALEC-GNN Model:")
        logger.info(f"    Accuracy:        {metrics.accuracy:.2%} +/- {cv_results.get('std_val_acc', 0):.2%}")
        logger.info(f"    Balanced Acc:    {metrics.balanced_accuracy:.2%}")
        logger.info(f"    Macro F1:        {metrics.macro_f1:.3f}")
        logger.info(f"    Cohen's Kappa:   {metrics.cohen_kappa:.3f}")
        logger.info(f"    Crisis Recall:   {metrics.crisis_recall:.2%} +/- {cv_results.get('std_crisis_recall', 0):.2%}")
        logger.info(f"    ROC-AUC:         {metrics.roc_auc_ovr:.3f}")
        
        if baseline_results:
            logger.info(f"\n  Baseline Comparison (mean +/- std):")
            for name, br in baseline_results.items():
                logger.info(f"    {name}: Acc={br['accuracy_mean']:.2%}+/-{br['accuracy_std']:.2%}, "
                           f"Crisis Recall={br['crisis_recall_mean']:.2%}+/-{br['crisis_recall_std']:.2%}, "
                           f"F1={br['macro_f1_mean']:.3f}+/-{br['macro_f1_std']:.3f}")


if __name__ == "__main__":
    main()