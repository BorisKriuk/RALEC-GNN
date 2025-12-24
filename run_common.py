#!/usr/bin/env python3
"""
run_common.py - Shared utilities for benchmark and model runs
"""

import os
import logging
import hashlib
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from scipy import signal

from algorithm import (
    ResearchConfig, EODHDClient, MultiMethodGraphBuilder, 
    RegimeLabeler, ALL_SYMBOLS, SYMBOL_CATEGORIES, DEVICE, API_KEY, SEED
)

# Set seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# Create directories
for dir_name in ['output', 'output/graphs', 'output/models', 'output/metrics', 'output/baselines', 'cache']:
    Path(dir_name).mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')


def setup_logging(log_name: str = 'research'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'output/{log_name}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# =============================================================================
# Parallel Data Fetching
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


def fetch_all_data_parallel(client, symbols, start_date, end_date, min_observations, max_workers=10, logger=None):
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
            
            if logger and (completed % 20 == 0 or completed == total):
                logger.info(f"  Progress: {completed}/{total} - Loaded {len(data)} symbols")
    
    return data, failed_symbols


# =============================================================================
# Caching Infrastructure
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
    
    def get_regime_cache_path(self, symbols, config_hash):
        """Get cache path for regime labels."""
        key = self._get_hash(sorted(symbols), config_hash)
        return self.cache_dir / f"regime_{key}.pkl"
    
    def load(self, path, logger=None):
        """Load from cache if exists."""
        if path.exists():
            if logger:
                logger.info(f"Loading from cache: {path}")
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to load cache: {e}")
                return None
        return None
    
    def save(self, path, data, logger=None):
        """Save to cache."""
        if logger:
            logger.info(f"Saving to cache: {path}")
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            if logger:
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
# Fast Lead-Lag Detection (FFT-based)
# =============================================================================

class FastLeadLagDetector:
    """FFT-based lead-lag detection - much faster than loop-based."""
    
    def __init__(self, config):
        self.max_lag = config.max_lag
        self.min_correlation = config.significance_threshold
    
    def detect_relationships_fast(self, returns_dict, top_k=100, logger=None):
        """Vectorized lead-lag using FFT cross-correlation."""
        symbols = list(returns_dict.keys())
        n = len(symbols)
        
        # Align all series to common index
        df = pd.DataFrame(returns_dict)
        df = df.dropna()
        
        if len(df) < 50:
            if logger:
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
# Data Preparation Pipeline
# =============================================================================

class DataPipeline:
    """Centralized data preparation pipeline."""
    
    def __init__(self, config: ResearchConfig, logger=None):
        self.config = config
        self.logger = logger
        self.cache = CacheManager()
        self.config_hash = get_config_hash(config)
        
    def prepare_all_data(self):
        """
        Prepare all data needed for experiments.
        Returns: data, returns_dict, regime_df, graphs, symbols_list, lead_lag_df
        """
        if not API_KEY:
            if self.logger:
                self.logger.error("API key not found! Please set EODHD_API_KEY in .env file")
            return None
        
        client = EODHDClient(API_KEY)
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=self.config.lookback_days)).strftime('%Y-%m-%d')
        
        # Step 1: Data Collection
        if self.logger:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"STEP 1: Data Collection ({len(ALL_SYMBOLS)} symbols)")
            self.logger.info(f"{'='*80}")
        
        data_cache_path = self.cache.get_data_cache_path(ALL_SYMBOLS, start_date, end_date)
        cached_data = self.cache.load(data_cache_path, self.logger)
        
        if cached_data is not None:
            data, failed_symbols = cached_data
            if self.logger:
                self.logger.info(f"Loaded {len(data)} assets from cache")
        else:
            if self.logger:
                self.logger.info(f"Fetching data from {start_date} to {end_date} (parallel)...")
            data, failed_symbols = fetch_all_data_parallel(
                client, ALL_SYMBOLS, start_date, end_date, 
                self.config.min_observations, max_workers=10, logger=self.logger
            )
            self.cache.save(data_cache_path, (data, failed_symbols), self.logger)
        
        if self.logger:
            self.logger.info(f"\nSuccessfully loaded {len(data)} assets")
            if failed_symbols:
                self.logger.info(f"Failed to load: {len(failed_symbols)} symbols")
        
        if len(data) < 10:
            if self.logger:
                self.logger.error(f"Insufficient assets: {len(data)} < 10")
            return None
        
        # Build returns dict
        returns_dict = {}
        for sym, df in data.items():
            prices = df['adjusted_close']
            returns = np.log(prices / prices.shift(1)).dropna()
            if len(returns) >= self.config.min_observations:
                returns_dict[sym] = returns
        
        if self.logger:
            self.logger.info(f"Working with {len(returns_dict)} assets for analysis")
            
            category_counts = defaultdict(int)
            for sym in returns_dict.keys():
                cat = SYMBOL_CATEGORIES.get(sym.replace('.US', ''), 'unknown')
                category_counts[cat] += 1
            
            self.logger.info("\nAsset category distribution:")
            for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                self.logger.info(f"   {cat}: {count}")
        
        # Step 2: Lead-Lag Detection
        if self.logger:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"STEP 2: Statistical Lead-Lag Detection [FFT]")
            self.logger.info(f"{'='*80}")
        
        leadlag_cache_path = self.cache.get_leadlag_cache_path(list(returns_dict.keys()), self.config_hash)
        lead_lag_df = self.cache.load(leadlag_cache_path, self.logger)
        
        if lead_lag_df is None:
            fast_detector = FastLeadLagDetector(self.config)
            lead_lag_df = fast_detector.detect_relationships_fast(returns_dict, top_k=100, logger=self.logger)
            self.cache.save(leadlag_cache_path, lead_lag_df, self.logger)
        
        if self.logger and not lead_lag_df.empty:
            self.logger.info(f"\nTOP MARKET LEADERS:")
            if 'corr_leader' in lead_lag_df.columns:
                leaders = lead_lag_df['corr_leader'].value_counts().head(10)
                for leader, count in leaders.items():
                    self.logger.info(f"   {leader}: influences {count} assets")
        
        # Step 3: Regime Detection & Labeling
        if self.logger:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"STEP 3: Regime Detection & Labeling")
            self.logger.info(f"{'='*80}")
        
        regime_cache_path = self.cache.get_regime_cache_path(list(returns_dict.keys()), self.config_hash)
        regime_df = self.cache.load(regime_cache_path, self.logger)
        
        if regime_df is None:
            labeler = RegimeLabeler(self.config)
            regime_df = labeler.label_regimes(returns_dict, method=self.config.regime_method)
            self.cache.save(regime_cache_path, regime_df, self.logger)
        
        if self.logger:
            self.logger.info(f"\nRegime distribution:")
            regime_counts = regime_df['regime'].value_counts()
            for regime in sorted(regime_counts.index):
                pct = regime_counts[regime] / len(regime_df) * 100
                name = {0: 'Bull/Low Vol', 1: 'Normal/Bear', 2: 'Crisis/Contagion'}[regime]
                self.logger.info(f"   {name}: {pct:.1f}% ({regime_counts[regime]} days)")
        
        # Step 4: Temporal Graph Construction
        if self.logger:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"STEP 4: Temporal Graph Construction")
            self.logger.info(f"{'='*80}")
        
        graphs_cache_path = self.cache.get_graphs_cache_path(list(returns_dict.keys()), self.config_hash)
        cached_graphs = self.cache.load(graphs_cache_path, self.logger)
        
        if cached_graphs is not None:
            graphs, symbols_list = cached_graphs
            if self.logger:
                self.logger.info(f"Loaded {len(graphs)} graphs from cache")
        else:
            graph_builder = MultiMethodGraphBuilder(self.config)
            graphs, symbols_list = graph_builder.build_temporal_graphs(
                returns_dict,
                use_full_graph=self.config.use_learned_edges
            )
            if graphs:
                self.cache.save(graphs_cache_path, (graphs, symbols_list), self.logger)
        
        if not graphs:
            if self.logger:
                self.logger.error("Failed to build graphs")
            return None
        
        if self.logger:
            avg_nodes = np.mean([g.num_nodes for g in graphs])
            avg_edges = np.mean([g.edge_index.shape[1] for g in graphs])
            self.logger.info(f"\nGraph statistics:")
            self.logger.info(f"   Number of graphs: {len(graphs)}")
            self.logger.info(f"   Average nodes: {avg_nodes:.1f}")
            self.logger.info(f"   Average edges: {avg_edges:.1f}")
        
        return {
            'data': data,
            'returns_dict': returns_dict,
            'regime_df': regime_df,
            'graphs': graphs,
            'symbols_list': symbols_list,
            'lead_lag_df': lead_lag_df
        }