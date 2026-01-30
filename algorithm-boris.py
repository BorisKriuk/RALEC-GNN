#!/usr/bin/env python3
"""
Correlation Graph Eigenvalue Crisis Detector (CGECD)
====================================================

Novel approach: Use spectral properties of dynamic correlation networks
to detect regime shifts and predict crisis events.

Key insight: During crises, normally uncorrelated assets become correlated,
causing measurable changes in the correlation matrix eigenvalue spectrum
BEFORE the full crisis manifests.

What's novel:
1. Dynamic correlation graph construction with multiple timescales
2. Spectral features from eigenvalue decomposition (not just λ₁)
3. Graph topology features (clustering, centrality, community structure)
4. Eigenvalue dynamics (rate of change, acceleration)
5. Cross-asset contagion measures from graph structure

Uses Random Forest for interpretability and speed - no GPU needed.
"""

import os
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy import stats, linalg
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, r2_score,
    average_precision_score, brier_score_loss
)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# SETUP
# ============================================================================
warnings.filterwarnings('ignore')
np.random.seed(42)

load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY") or os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("EODHD_API_KEY not found in environment!")

OUTPUT_DIR = Path("cgecd_results")
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================
class DataLoader:
    """Load market data from EODHD API with caching"""
    
    BASE_URL = "https://eodhd.com/api"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
    
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_file = symbol.replace('.', '_').replace('/', '_')
        cache_path = CACHE_DIR / f"{cache_file}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                    if len(df) > 100:
                        return df
            except Exception:
                pass
        
        try:
            params = {
                'api_token': self.api_key,
                'fmt': 'json',
                'from': start_date,
                'to': end_date
            }
            
            url = f"{self.BASE_URL}/eod/{symbol}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            
            return df
            
        except Exception as e:
            print(f"  Failed to load {symbol}: {e}")
            return pd.DataFrame()


def load_multi_asset_data(years: int = 15) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load diverse set of assets for correlation analysis.
    We need assets that are normally UNCORRELATED but become correlated in crises.
    """
    loader = DataLoader(API_KEY)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    
    # Diverse asset universe - chosen for normally low correlations
    symbols = {
        # US Equity Sectors
        'SPY.US': 'SP500',
        'QQQ.US': 'Nasdaq100',
        'IWM.US': 'Russell2000',
        'XLF.US': 'Financials',
        'XLE.US': 'Energy',
        'XLK.US': 'Technology',
        'XLV.US': 'Healthcare',
        'XLU.US': 'Utilities',
        'XLP.US': 'ConsumerStaples',
        'XLY.US': 'ConsumerDisc',
        'XLI.US': 'Industrials',
        'XLB.US': 'Materials',
        'XLRE.US': 'RealEstate',
        
        # International
        'EFA.US': 'DevIntl',
        'EEM.US': 'EmergingMkts',
        'VGK.US': 'Europe',
        'EWJ.US': 'Japan',
        
        # Fixed Income
        'TLT.US': 'LongTreasury',
        'IEF.US': 'IntermTreasury',
        'LQD.US': 'InvGradeCorp',
        'HYG.US': 'HighYield',
        
        # Alternatives
        'GLD.US': 'Gold',
        'USO.US': 'Oil',
        'UUP.US': 'USDollar',
        'VNQ.US': 'REITs',
    }
    
    print(f"Loading {len(symbols)} assets...")
    
    data_dict = {}
    for symbol, name in symbols.items():
        df = loader.get_data(symbol, start_date, end_date)
        if not df.empty and 'adjusted_close' in df.columns:
            data_dict[name] = df['adjusted_close']
            print(f"  ✓ {name}: {len(df)} days")
        else:
            print(f"  ✗ {name}: failed")
    
    # Combine into single DataFrame
    prices = pd.DataFrame(data_dict)
    prices = prices.dropna(how='all')
    
    # Forward fill small gaps, then drop remaining NaN
    prices = prices.ffill(limit=5)
    prices = prices.dropna()
    
    print(f"\nCombined dataset: {len(prices)} days, {len(prices.columns)} assets")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    return prices, {v: k for k, v in symbols.items()}


# ============================================================================
# CORRELATION GRAPH CONSTRUCTION
# ============================================================================
class CorrelationGraphBuilder:
    """
    Build and analyze dynamic correlation graphs.
    
    Key innovation: Extract features from the STRUCTURE of correlations,
    not just the correlation values themselves.
    """
    
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.n_assets = len(prices.columns)
        self.asset_names = list(prices.columns)
    
    def compute_rolling_correlation(
        self, 
        window: int = 60,
        min_periods: int = 30
    ) -> Dict[pd.Timestamp, np.ndarray]:
        """Compute rolling correlation matrices"""
        
        corr_matrices = {}
        
        for i in range(window, len(self.returns)):
            date = self.returns.index[i]
            window_returns = self.returns.iloc[i-window:i]
            
            if len(window_returns) >= min_periods:
                corr = window_returns.corr().values
                # Handle NaN in correlation matrix
                corr = np.nan_to_num(corr, nan=0)
                np.fill_diagonal(corr, 1.0)
                corr_matrices[date] = corr
        
        return corr_matrices
    
    def compute_exponential_correlation(
        self,
        halflife: int = 20
    ) -> Dict[pd.Timestamp, np.ndarray]:
        """Compute exponentially weighted correlation matrices"""
        
        corr_matrices = {}
        decay = np.exp(-np.log(2) / halflife)
        
        for i in range(60, len(self.returns)):  # Need warmup period
            date = self.returns.index[i]
            window_returns = self.returns.iloc[:i]
            
            # Exponentially weighted covariance
            ewm_cov = window_returns.ewm(halflife=halflife).cov().iloc[-self.n_assets:]
            ewm_cov = ewm_cov.values.reshape(self.n_assets, self.n_assets)
            
            # Convert to correlation
            std = np.sqrt(np.diag(ewm_cov))
            std_outer = np.outer(std, std)
            std_outer[std_outer == 0] = 1  # Avoid division by zero
            corr = ewm_cov / std_outer
            
            corr = np.nan_to_num(corr, nan=0)
            np.fill_diagonal(corr, 1.0)
            corr_matrices[date] = corr
        
        return corr_matrices


# ============================================================================
# SPECTRAL FEATURE EXTRACTION (THE NOVEL PART)
# ============================================================================
class SpectralFeatureExtractor:
    """
    Extract features from eigenvalue decomposition of correlation matrices.
    
    This is where the novelty lies - we're not just looking at correlations,
    but at the STRUCTURE of the correlation matrix through its spectrum.
    """
    
    def __init__(self, n_assets: int):
        self.n_assets = n_assets
    
    def extract_eigenvalue_features(self, corr_matrix: np.ndarray) -> Dict[str, float]:
        """
        Extract features from eigenvalue decomposition.
        
        During crises:
        - λ₁ increases (market factor dominates)
        - Eigenvalue entropy decreases (spectrum concentrates)
        - Effective rank decreases (fewer independent factors)
        - Absorption ratio increases (top eigenvalues explain more)
        """
        
        # Ensure symmetric and valid
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)
        
        try:
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
            eigenvalues = np.maximum(eigenvalues, 1e-10)  # Ensure positive
        except Exception:
            return self._default_features()
        
        n = len(eigenvalues)
        total_var = np.sum(eigenvalues)
        
        features = {}
        
        # === Primary Eigenvalue Features ===
        features['lambda_1'] = eigenvalues[0]
        features['lambda_1_ratio'] = eigenvalues[0] / total_var
        features['lambda_2'] = eigenvalues[1] if n > 1 else 0
        features['lambda_3'] = eigenvalues[2] if n > 2 else 0
        
        # Ratio of first to second (spectral gap)
        features['spectral_gap'] = eigenvalues[0] / (eigenvalues[1] + 1e-10) if n > 1 else n
        
        # === Absorption Ratio (Kritzman et al.) ===
        # Fraction of variance explained by top k eigenvalues
        for k in [1, 3, 5]:
            if k <= n:
                features[f'absorption_ratio_{k}'] = np.sum(eigenvalues[:k]) / total_var
            else:
                features[f'absorption_ratio_{k}'] = 1.0
        
        # === Eigenvalue Entropy (measure of concentration) ===
        # Higher entropy = more uniform distribution = normal times
        # Lower entropy = concentrated in few eigenvalues = crisis
        normalized_eig = eigenvalues / total_var
        entropy = -np.sum(normalized_eig * np.log(normalized_eig + 1e-10))
        max_entropy = np.log(n)
        features['eigenvalue_entropy'] = entropy
        features['normalized_entropy'] = entropy / max_entropy if max_entropy > 0 else 0
        
        # === Effective Rank (Roy, 2007) ===
        # Measures "true" dimensionality of the correlation structure
        features['effective_rank'] = np.exp(entropy)
        features['effective_rank_ratio'] = features['effective_rank'] / n
        
        # === Marchenko-Pastur Analysis ===
        # Compare to random matrix theory expectations
        # For random correlations, largest eigenvalue ~ (1 + sqrt(n/T))²
        # Excess indicates true structure
        mp_upper = (1 + np.sqrt(1))** 2  # Simplified for T >> n
        features['mp_excess'] = max(0, eigenvalues[0] - mp_upper)
        
        # === Higher Order Statistics ===
        features['eigenvalue_std'] = np.std(eigenvalues)
        features['eigenvalue_skew'] = stats.skew(eigenvalues)
        features['eigenvalue_kurt'] = stats.kurtosis(eigenvalues)
        
        # Tail eigenvalues (noise floor)
        features['tail_eigenvalue_mean'] = np.mean(eigenvalues[-5:]) if n >= 5 else np.mean(eigenvalues)
        
        # === Condition Number (numerical stability indicator) ===
        features['condition_number'] = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        features['log_condition_number'] = np.log(features['condition_number'] + 1)
        
        return features
    
    def extract_eigenvector_features(self, corr_matrix: np.ndarray) -> Dict[str, float]:
        """
        Extract features from eigenvectors.
        
        The first eigenvector shows how assets load on the "market factor".
        During crises, this becomes more uniform (all assets move together).
        """
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        except Exception:
            return {}
        
        features = {}
        
        # First eigenvector (market factor loadings)
        v1 = np.abs(eigenvectors[:, 0])
        v1 = v1 / (np.sum(v1) + 1e-10)  # Normalize
        
        # === Market Factor Uniformity ===
        # During crises, all assets load similarly on market factor
        features['v1_entropy'] = -np.sum(v1 * np.log(v1 + 1e-10))
        features['v1_max'] = np.max(v1)
        features['v1_min'] = np.min(v1)
        features['v1_std'] = np.std(v1)
        features['v1_herfindahl'] = np.sum(v1 ** 2)  # Concentration measure
        
        # === Cross-Sector Similarity ===
        # Are different sectors loading similarly? (crisis indicator)
        n = len(v1)
        if n >= 4:
            # Compare loadings of first vs last quartile
            sorted_loadings = np.sort(v1)
            q1_mean = np.mean(sorted_loadings[:n//4])
            q4_mean = np.mean(sorted_loadings[-n//4:])
            features['loading_dispersion'] = q4_mean - q1_mean
        
        return features
    
    def _default_features(self) -> Dict[str, float]:
        """Default features when eigendecomposition fails"""
        return {
            'lambda_1': 1.0,
            'lambda_1_ratio': 1.0 / self.n_assets,
            'lambda_2': 0.0,
            'lambda_3': 0.0,
            'spectral_gap': 1.0,
            'absorption_ratio_1': 1.0 / self.n_assets,
            'absorption_ratio_3': 3.0 / self.n_assets,
            'absorption_ratio_5': 5.0 / self.n_assets,
            'eigenvalue_entropy': np.log(self.n_assets),
            'normalized_entropy': 1.0,
            'effective_rank': self.n_assets,
            'effective_rank_ratio': 1.0,
            'mp_excess': 0.0,
            'eigenvalue_std': 0.0,
            'eigenvalue_skew': 0.0,
            'eigenvalue_kurt': 0.0,
            'tail_eigenvalue_mean': 1.0,
            'condition_number': 1.0,
            'log_condition_number': 0.0,
        }


# ============================================================================
# GRAPH TOPOLOGY FEATURES
# ============================================================================
class GraphTopologyExtractor:
    """
    Extract features from the network topology of the correlation graph.
    
    We threshold correlations to create a graph, then analyze its structure.
    During crises, the graph becomes more connected and centralized.
    """
    
    def __init__(self, n_assets: int, asset_names: List[str]):
        self.n_assets = n_assets
        self.asset_names = asset_names
    
    def extract_topology_features(
        self, 
        corr_matrix: np.ndarray,
        thresholds: List[float] = [0.3, 0.5, 0.7]
    ) -> Dict[str, float]:
        """
        Extract graph topology features at multiple thresholds.
        """
        
        features = {}
        
        for thresh in thresholds:
            # Create adjacency matrix
            adj = (np.abs(corr_matrix) > thresh).astype(float)
            np.fill_diagonal(adj, 0)  # No self-loops
            
            suffix = f"_t{int(thresh*100)}"
            
            # === Basic Connectivity ===
            n_edges = np.sum(adj) / 2
            max_edges = self.n_assets * (self.n_assets - 1) / 2
            features[f'edge_density{suffix}'] = n_edges / max_edges if max_edges > 0 else 0
            features[f'n_edges{suffix}'] = n_edges
            
            # === Degree Statistics ===
            degrees = np.sum(adj, axis=1)
            features[f'degree_mean{suffix}'] = np.mean(degrees)
            features[f'degree_std{suffix}'] = np.std(degrees)
            features[f'degree_max{suffix}'] = np.max(degrees)
            features[f'isolated_nodes{suffix}'] = np.sum(degrees == 0)
            
            # === Centralization ===
            # How much is the graph dominated by a few central nodes?
            if np.max(degrees) > 0:
                centralization = np.sum(np.max(degrees) - degrees) / ((self.n_assets - 1) * (self.n_assets - 2))
                features[f'centralization{suffix}'] = centralization
            else:
                features[f'centralization{suffix}'] = 0
            
            # === Clustering Coefficient ===
            # Local clustering (transitivity)
            clustering = self._compute_clustering_coefficient(adj)
            features[f'clustering_coef{suffix}'] = clustering
            
            # === Connected Components ===
            n_components = self._count_components(adj)
            features[f'n_components{suffix}'] = n_components
            features[f'fragmentation{suffix}'] = (n_components - 1) / (self.n_assets - 1) if self.n_assets > 1 else 0
        
        # === Additional correlation-based features ===
        # Average absolute correlation
        upper_tri = corr_matrix[np.triu_indices(self.n_assets, k=1)]
        features['mean_abs_corr'] = np.mean(np.abs(upper_tri))
        features['median_abs_corr'] = np.median(np.abs(upper_tri))
        features['max_abs_corr'] = np.max(np.abs(upper_tri))
        features['corr_std'] = np.std(upper_tri)
        features['corr_skew'] = stats.skew(upper_tri)
        
        # Fraction of high correlations
        features['frac_corr_above_50'] = np.mean(np.abs(upper_tri) > 0.5)
        features['frac_corr_above_70'] = np.mean(np.abs(upper_tri) > 0.7)
        
        return features
    
    def _compute_clustering_coefficient(self, adj: np.ndarray) -> float:
        """Compute global clustering coefficient"""
        n = len(adj)
        triangles = 0
        triplets = 0
        
        for i in range(n):
            neighbors = np.where(adj[i] > 0)[0]
            k = len(neighbors)
            if k >= 2:
                triplets += k * (k - 1) / 2
                for j in range(len(neighbors)):
                    for k_idx in range(j + 1, len(neighbors)):
                        if adj[neighbors[j], neighbors[k_idx]] > 0:
                            triangles += 1
        
        return triangles / triplets if triplets > 0 else 0
    
    def _count_components(self, adj: np.ndarray) -> int:
        """Count connected components using DFS"""
        n = len(adj)
        visited = [False] * n
        components = 0
        
        def dfs(node):
            visited[node] = True
            for neighbor in range(n):
                if adj[node, neighbor] > 0 and not visited[neighbor]:
                    dfs(neighbor)
        
        for i in range(n):
            if not visited[i]:
                dfs(i)
                components += 1
        
        return components


# ============================================================================
# DYNAMICS FEATURES (RATE OF CHANGE)
# ============================================================================
class DynamicsFeatureExtractor:
    """
    Extract features from the DYNAMICS of spectral/graph features.
    
    Key insight: The RATE OF CHANGE of correlation structure
    often precedes crises. Rapid increases in λ₁ or connectivity
    are early warning signals.
    """
    
    def compute_dynamics(
        self, 
        feature_series: pd.DataFrame,
        lookbacks: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Compute rate of change, acceleration, and regime indicators.
        """
        
        dynamics = pd.DataFrame(index=feature_series.index)
        
        key_features = [
            'lambda_1', 'lambda_1_ratio', 'absorption_ratio_1',
            'eigenvalue_entropy', 'effective_rank', 'mean_abs_corr',
            'edge_density_t50', 'clustering_coef_t50'
        ]
        
        for feat in key_features:
            if feat not in feature_series.columns:
                continue
            
            series = feature_series[feat]
            
            for lb in lookbacks:
                # Rate of change
                dynamics[f'{feat}_roc_{lb}d'] = series.pct_change(lb)
                
                # Absolute change
                dynamics[f'{feat}_diff_{lb}d'] = series.diff(lb)
                
                # Z-score relative to rolling window
                rolling_mean = series.rolling(lb * 2).mean()
                rolling_std = series.rolling(lb * 2).std()
                dynamics[f'{feat}_zscore_{lb}d'] = (series - rolling_mean) / (rolling_std + 1e-10)
            
            # Acceleration (second derivative)
            dynamics[f'{feat}_accel'] = series.diff().diff()
            
            # Percentile rank over last year
            dynamics[f'{feat}_percentile_252d'] = series.rolling(252).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 10 else 0.5
            )
        
        return dynamics


# ============================================================================
# CRISIS TARGET DEFINITION
# ============================================================================
class CrisisTargetBuilder:
    """
    Define what constitutes a "crisis" or "stress event" to predict.
    
    We use multiple definitions to see which is most predictable.
    """
    
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
        # Use SPY or first asset as market proxy
        self.market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
        self.market_returns = self.market.pct_change()
    
    def compute_targets(self) -> pd.DataFrame:
        """Compute various crisis/stress targets"""
        
        targets = pd.DataFrame(index=self.prices.index)
        
        # === Drawdown-Based Targets ===
        for horizon in [5, 10, 20]:
            future_dd = self._compute_future_drawdown(horizon)
            
            for threshold in [0.02, 0.03, 0.05, 0.07, 0.10]:
                col_name = f'drawdown_{int(threshold*100)}pct_{horizon}d'
                targets[col_name] = (future_dd < -threshold).astype(int)
        
        # === Volatility Spike Targets ===
        realized_vol = self.market_returns.rolling(20).std() * np.sqrt(252)
        
        for horizon in [5, 10, 20]:
            future_vol = realized_vol.shift(-horizon)
            
            # Vol doubles
            targets[f'vol_spike_2x_{horizon}d'] = (future_vol > realized_vol * 2).astype(int)
            
            # Vol in top 10% historically
            vol_threshold = realized_vol.rolling(252).quantile(0.9)
            targets[f'vol_extreme_{horizon}d'] = (future_vol > vol_threshold).astype(int)
        
        # === Large Down Move Targets ===
        for horizon in [1, 3, 5, 10]:
            future_ret = self.market.pct_change(horizon).shift(-horizon)
            
            for threshold in [0.02, 0.03, 0.05]:
                targets[f'down_{int(threshold*100)}pct_{horizon}d'] = (future_ret < -threshold).astype(int)
        
        # === VIX-Like Spike (using realized vol as proxy) ===
        vol_percentile = realized_vol.rolling(252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 20 else 0.5
        )
        
        future_vol_pct = vol_percentile.shift(-10)
        targets['vol_regime_shift_10d'] = (future_vol_pct > 0.8).astype(int)
        
        # === Correlation Spike Target (what we're best at predicting!) ===
        # This is somewhat circular but useful for validation
        
        return targets
    
    def _compute_future_drawdown(self, horizon: int) -> pd.Series:
        """Compute maximum drawdown over next N days"""
        
        future_dd = pd.Series(index=self.prices.index, dtype=float)
        
        for i in range(len(self.market) - horizon):
            current_price = self.market.iloc[i]
            future_prices = self.market.iloc[i+1:i+horizon+1]
            min_price = future_prices.min()
            dd = (min_price - current_price) / current_price
            future_dd.iloc[i] = dd
        
        return future_dd


# ============================================================================
# TRADITIONAL FEATURES (FOR COMPARISON)
# ============================================================================
class TraditionalFeatureBuilder:
    """
    Build traditional technical/fundamental features for comparison.
    This helps us isolate the value of spectral features.
    """
    
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
        self.market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
        self.returns = self.market.pct_change()
    
    def compute_features(self) -> pd.DataFrame:
        """Compute standard predictive features"""
        
        features = pd.DataFrame(index=self.prices.index)
        
        # === Return Features ===
        for window in [1, 5, 10, 20, 60]:
            features[f'return_{window}d'] = self.market.pct_change(window)
        
        # === Volatility Features ===
        for window in [5, 10, 20, 60]:
            features[f'volatility_{window}d'] = self.returns.rolling(window).std() * np.sqrt(252)
        
        # Vol ratio
        features['vol_ratio_5_20'] = features['volatility_5d'] / (features['volatility_20d'] + 1e-8)
        
        # === Momentum/Trend ===
        for window in [10, 20, 50]:
            sma = self.market.rolling(window).mean()
            features[f'price_to_sma_{window}'] = self.market / sma - 1
        
        # RSI
        delta = self.market.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # === Drawdown ===
        rolling_max = self.market.rolling(20).max()
        features['drawdown_20d'] = (self.market - rolling_max) / rolling_max
        
        rolling_max_60 = self.market.rolling(60).max()
        features['drawdown_60d'] = (self.market - rolling_max_60) / rolling_max_60
        
        # === Higher Moments ===
        for window in [20, 60]:
            features[f'skewness_{window}d'] = self.returns.rolling(window).skew()
            features[f'kurtosis_{window}d'] = self.returns.rolling(window).kurt()
        
        # === Calendar ===
        features['day_of_week'] = self.prices.index.dayofweek
        features['month'] = self.prices.index.month
        
        return features


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================
class CrisisPredictor:
    """
    Train and evaluate crisis prediction models.
    Uses walk-forward validation with proper temporal separation.
    """
    
    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        feature_groups: Dict[str, List[str]] = None
    ):
        self.features = features
        self.targets = targets
        self.feature_groups = feature_groups or {}
    
    def walk_forward_evaluation(
        self,
        target_col: str,
        feature_cols: List[str] = None,
        n_splits: int = 5,
        train_years: float = 3,
        test_months: int = 6,
        gap_days: int = 10
    ) -> Dict:
        """
        Walk-forward cross-validation with temporal gap.
        """
        
        if feature_cols is None:
            feature_cols = self.features.columns.tolist()
        
        # Align data
        valid_features = [f for f in feature_cols if f in self.features.columns]
        
        common_idx = self.features[valid_features].dropna().index
        common_idx = common_idx.intersection(self.targets[target_col].dropna().index)
        
        X = self.features.loc[common_idx, valid_features]
        y = self.targets.loc[common_idx, target_col]
        
        train_size = int(train_years * 252)
        test_size = int(test_months * 21)
        gap = gap_days
        
        if len(X) < train_size + gap + test_size:
            return {'error': 'Insufficient data'}
        
        # Calculate splits
        available = len(X) - train_size - gap - test_size
        step = max(test_size, available // n_splits)
        
        all_preds = []
        all_probs = []
        all_actuals = []
        all_dates = []
        fold_results = []
        
        for fold in range(n_splits):
            start = fold * step
            train_end = start + train_size
            test_start = train_end + gap
            test_end = min(test_start + test_size, len(X))
            
            if test_end > len(X):
                break
            
            X_train = X.iloc[start:train_end]
            y_train = y.iloc[start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Scale
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Handle inf/nan
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=20,
                min_samples_split=50,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )
            
            try:
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                probs = model.predict_proba(X_test_scaled)[:, 1]
                
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_actuals.extend(y_test.values)
                all_dates.extend(y_test.index.tolist())
                
                # Fold metrics
                fold_metrics = self._compute_metrics(y_test.values, preds, probs)
                fold_metrics['fold'] = fold
                fold_metrics['train_period'] = f"{X.index[start].date()} to {X.index[train_end-1].date()}"
                fold_metrics['test_period'] = f"{X.index[test_start].date()} to {X.index[test_end-1].date()}"
                fold_results.append(fold_metrics)
                
            except Exception as e:
                print(f"  Fold {fold} failed: {e}")
                continue
        
        if len(all_preds) == 0:
            return {'error': 'All folds failed'}
        
        # Overall metrics
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_actuals = np.array(all_actuals)
        
        overall = self._compute_metrics(all_actuals, all_preds, all_probs)
        overall['n_predictions'] = len(all_preds)
        overall['fold_results'] = fold_results
        overall['predictions'] = all_preds
        overall['probabilities'] = all_probs
        overall['actuals'] = all_actuals
        overall['dates'] = all_dates
        
        # Feature importance (from last fold)
        if 'model' in dir():
            importance = pd.DataFrame({
                'feature': valid_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            overall['feature_importance'] = importance
        
        return overall
    
    def _compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray
    ) -> Dict:
        """Compute comprehensive classification metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # AUC-ROC
        if len(np.unique(y_true)) > 1:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                metrics['avg_precision'] = average_precision_score(y_true, y_prob)
                metrics['brier_score'] = brier_score_loss(y_true, y_prob)
            except Exception:
                metrics['auc_roc'] = 0.5
                metrics['avg_precision'] = np.mean(y_true)
                metrics['brier_score'] = 0.25
        
        # Class distribution
        metrics['positive_rate'] = np.mean(y_true)
        metrics['predicted_positive_rate'] = np.mean(y_pred)
        
        # Lift
        baseline_acc = max(np.mean(y_true), 1 - np.mean(y_true))
        metrics['lift_over_baseline'] = metrics['accuracy'] - baseline_acc
        
        return metrics


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_cgecd_experiment():
    """
    Main experiment: Compare spectral features vs traditional features.
    """
    
    print("=" * 80)
    print("CORRELATION GRAPH EIGENVALUE CRISIS DETECTOR (CGECD)")
    print("=" * 80)
    print("\nNovel approach: Use spectral properties of correlation networks")
    print("to detect regime shifts before they fully manifest.\n")
    
    # === Load Data ===
    print("[1/6] Loading multi-asset data...")
    prices, symbol_map = load_multi_asset_data(years=15)
    
    if len(prices) < 1000:
        print("Insufficient data!")
        return
    
    # === Build Correlation Graphs ===
    print("\n[2/6] Building dynamic correlation graphs...")
    
    graph_builder = CorrelationGraphBuilder(prices)
    
    # Multiple timescales
    corr_60d = graph_builder.compute_rolling_correlation(window=60)
    corr_120d = graph_builder.compute_rolling_correlation(window=120)
    corr_ewm = graph_builder.compute_exponential_correlation(halflife=30)
    
    print(f"  Built {len(corr_60d)} correlation matrices (60d window)")
    print(f"  Built {len(corr_120d)} correlation matrices (120d window)")
    print(f"  Built {len(corr_ewm)} correlation matrices (EWM)")
    
    # === Extract Spectral Features ===
    print("\n[3/6] Extracting spectral and graph features...")
    
    n_assets = len(prices.columns)
    spectral_extractor = SpectralFeatureExtractor(n_assets)
    topology_extractor = GraphTopologyExtractor(n_assets, list(prices.columns))
    
    spectral_features = []
    
    for date in corr_60d.keys():
        row = {'date': date}
        
        # Eigenvalue features (60d window)
        eig_feats = spectral_extractor.extract_eigenvalue_features(corr_60d[date])
        for k, v in eig_feats.items():
            row[f'{k}_60d'] = v
        
        # Eigenvector features
        evec_feats = spectral_extractor.extract_eigenvector_features(corr_60d[date])
        for k, v in evec_feats.items():
            row[f'{k}_60d'] = v
        
        # Graph topology features
        topo_feats = topology_extractor.extract_topology_features(corr_60d[date])
        for k, v in topo_feats.items():
            row[f'{k}_60d'] = v
        
        # 120d window features (longer-term structure)
        if date in corr_120d:
            eig_feats_120 = spectral_extractor.extract_eigenvalue_features(corr_120d[date])
            for k, v in eig_feats_120.items():
                row[f'{k}_120d'] = v
        
        # EWM features (more responsive)
        if date in corr_ewm:
            eig_feats_ewm = spectral_extractor.extract_eigenvalue_features(corr_ewm[date])
            for k, v in eig_feats_ewm.items():
                row[f'{k}_ewm'] = v
        
        spectral_features.append(row)
    
    spectral_df = pd.DataFrame(spectral_features)
    spectral_df.set_index('date', inplace=True)
    spectral_df = spectral_df.sort_index()
    
    print(f"  Extracted {len(spectral_df.columns)} spectral/graph features")
    
    # === Add Dynamics Features ===
    print("\n[4/6] Computing dynamics features...")
    
    dynamics_extractor = DynamicsFeatureExtractor()
    dynamics_df = dynamics_extractor.compute_dynamics(spectral_df)
    
    # Combine spectral + dynamics
    all_spectral = pd.concat([spectral_df, dynamics_df], axis=1)
    print(f"  Total spectral features: {len(all_spectral.columns)}")
    
    # === Traditional Features ===
    print("\n[5/6] Building traditional features for comparison...")
    
    trad_builder = TraditionalFeatureBuilder(prices)
    trad_features = trad_builder.compute_features()
    
    print(f"  Traditional features: {len(trad_features.columns)}")
    
    # === Build Targets ===
    print("\n[6/6] Computing crisis targets...")
    
    target_builder = CrisisTargetBuilder(prices)
    targets = target_builder.compute_targets()
    
    print(f"  Target variables: {len(targets.columns)}")
    
    # === Align All Data ===
    common_idx = all_spectral.dropna(thresh=int(len(all_spectral.columns) * 0.5)).index
    common_idx = common_idx.intersection(trad_features.dropna(thresh=int(len(trad_features.columns) * 0.5)).index)
    common_idx = common_idx.intersection(targets.dropna(how='all').index)
    
    all_spectral = all_spectral.loc[common_idx]
    trad_features = trad_features.loc[common_idx]
    targets = targets.loc[common_idx]
    
    # Combined feature set
    combined_features = pd.concat([all_spectral, trad_features], axis=1)
    
    print(f"\nFinal dataset: {len(common_idx)} days")
    print(f"  Spectral features: {len(all_spectral.columns)}")
    print(f"  Traditional features: {len(trad_features.columns)}")
    print(f"  Combined features: {len(combined_features.columns)}")
    
    # === Run Experiments ===
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)
    
    # Select target tasks
    test_targets = [
        ('drawdown_3pct_10d', 'Drawdown >3% in 10 days'),
        ('drawdown_5pct_10d', 'Drawdown >5% in 10 days'),
        ('drawdown_5pct_20d', 'Drawdown >5% in 20 days'),
        ('drawdown_7pct_20d', 'Drawdown >7% in 20 days'),
        ('vol_spike_2x_10d', 'Volatility doubles in 10 days'),
        ('vol_extreme_10d', 'Extreme volatility in 10 days'),
        ('down_3pct_5d', 'Down >3% in 5 days'),
        ('down_5pct_10d', 'Down >5% in 10 days'),
    ]
    
    results = []
    
    predictor_spectral = CrisisPredictor(all_spectral, targets)
    predictor_trad = CrisisPredictor(trad_features, targets)
    predictor_combined = CrisisPredictor(combined_features, targets)
    
    for target_col, target_name in test_targets:
        if target_col not in targets.columns:
            continue
        
        # Check class balance
        pos_rate = targets[target_col].mean()
        if pos_rate < 0.02 or pos_rate > 0.5:
            continue
        
        print(f"\n--- {target_name} (positive rate: {pos_rate:.1%}) ---")
        
        # Test spectral features only
        res_spectral = predictor_spectral.walk_forward_evaluation(
            target_col=target_col,
            n_splits=5,
            train_years=3,
            test_months=6
        )
        
        # Test traditional features only
        res_trad = predictor_trad.walk_forward_evaluation(
            target_col=target_col,
            n_splits=5,
            train_years=3,
            test_months=6
        )
        
        # Test combined features
        res_combined = predictor_combined.walk_forward_evaluation(
            target_col=target_col,
            n_splits=5,
            train_years=3,
            test_months=6
        )
        
        if 'error' not in res_spectral:
            print(f"  SPECTRAL:     AUC={res_spectral['auc_roc']:.3f}, "
                  f"Prec={res_spectral['precision']:.1%}, "
                  f"Recall={res_spectral['recall']:.1%}, "
                  f"F1={res_spectral['f1']:.1%}")
            
            results.append({
                'target': target_name,
                'feature_set': 'Spectral',
                'auc_roc': res_spectral['auc_roc'],
                'precision': res_spectral['precision'],
                'recall': res_spectral['recall'],
                'f1': res_spectral['f1'],
                'avg_precision': res_spectral.get('avg_precision', 0),
                'positive_rate': pos_rate
            })
        
        if 'error' not in res_trad:
            print(f"  TRADITIONAL:  AUC={res_trad['auc_roc']:.3f}, "
                  f"Prec={res_trad['precision']:.1%}, "
                  f"Recall={res_trad['recall']:.1%}, "
                  f"F1={res_trad['f1']:.1%}")
            
            results.append({
                'target': target_name,
                'feature_set': 'Traditional',
                'auc_roc': res_trad['auc_roc'],
                'precision': res_trad['precision'],
                'recall': res_trad['recall'],
                'f1': res_trad['f1'],
                'avg_precision': res_trad.get('avg_precision', 0),
                'positive_rate': pos_rate
            })
        
        if 'error' not in res_combined:
            print(f"  COMBINED:     AUC={res_combined['auc_roc']:.3f}, "
                  f"Prec={res_combined['precision']:.1%}, "
                  f"Recall={res_combined['recall']:.1%}, "
                  f"F1={res_combined['f1']:.1%}")
            
            results.append({
                'target': target_name,
                'feature_set': 'Combined',
                'auc_roc': res_combined['auc_roc'],
                'precision': res_combined['precision'],
                'recall': res_combined['recall'],
                'f1': res_combined['f1'],
                'avg_precision': res_combined.get('avg_precision', 0),
                'positive_rate': pos_rate
            })
            
            # Save feature importance for best model
            if 'feature_importance' in res_combined:
                res_combined['feature_importance'].to_csv(
                    OUTPUT_DIR / f'feature_importance_{target_col}.csv',
                    index=False
                )
    
    # === Summarize Results ===
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'experiment_results.csv', index=False)
    
    print("\n" + "=" * 80)
    print("SUMMARY: SPECTRAL vs TRADITIONAL FEATURES")
    print("=" * 80)
    
    # Compare by feature set
    summary = results_df.groupby('feature_set').agg({
        'auc_roc': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean'
    }).round(3)
    
    print("\nAverage Performance by Feature Set:")
    print(summary.to_string())
    
    # Find where spectral features help most
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    spectral_results = results_df[results_df['feature_set'] == 'Spectral']
    trad_results = results_df[results_df['feature_set'] == 'Traditional']
    combined_results = results_df[results_df['feature_set'] == 'Combined']
    
    if len(spectral_results) > 0 and len(trad_results) > 0:
        for target in spectral_results['target'].unique():
            spec_auc = spectral_results[spectral_results['target'] == target]['auc_roc'].values
            trad_auc = trad_results[trad_results['target'] == target]['auc_roc'].values
            
            if len(spec_auc) > 0 and len(trad_auc) > 0:
                improvement = spec_auc[0] - trad_auc[0]
                if improvement > 0.02:
                    print(f"✓ {target}: Spectral +{improvement:.3f} AUC over Traditional")
                elif improvement < -0.02:
                    print(f"✗ {target}: Traditional +{-improvement:.3f} AUC over Spectral")
                else:
                    print(f"≈ {target}: Similar performance")
    
    # === Create Visualizations ===
    create_cgecd_visualizations(results_df, spectral_df, targets, OUTPUT_DIR)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    
    return results_df, spectral_df, targets


def create_cgecd_visualizations(
    results_df: pd.DataFrame,
    spectral_df: pd.DataFrame,
    targets: pd.DataFrame,
    output_dir: Path
):
    """Create visualizations for the CGECD experiment"""
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. AUC comparison by feature set
        ax = axes[0, 0]
        if len(results_df) > 0:
            pivot = results_df.pivot_table(
                index='target', 
                columns='feature_set', 
                values='auc_roc'
            )
            if len(pivot) > 0:
                pivot.plot(kind='bar', ax=ax, width=0.8)
                ax.axhline(y=0.5, color='red', linestyle='--', label='Random')
                ax.set_ylabel('AUC-ROC')
                ax.set_title('Spectral vs Traditional Features')
                ax.legend(title='Feature Set')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 2. λ₁ over time
        ax = axes[0, 1]
        if 'lambda_1_60d' in spectral_df.columns:
            spectral_df['lambda_1_60d'].plot(ax=ax, label='λ₁ (60d)', alpha=0.7)
            ax.set_ylabel('First Eigenvalue')
            ax.set_title('Market Factor Strength Over Time')
            
            # Shade crisis periods (high λ₁)
            threshold = spectral_df['lambda_1_60d'].quantile(0.9)
            crisis_periods = spectral_df['lambda_1_60d'] > threshold
            ax.fill_between(
                spectral_df.index,
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                where=crisis_periods,
                alpha=0.3,
                color='red',
                label='High Correlation'
            )
            ax.legend()
        
        # 3. Eigenvalue entropy over time
        ax = axes[1, 0]
        if 'eigenvalue_entropy_60d' in spectral_df.columns:
            spectral_df['eigenvalue_entropy_60d'].plot(ax=ax, label='Entropy', alpha=0.7)
            ax.set_ylabel('Eigenvalue Entropy')
            ax.set_title('Correlation Structure Diversity')
            
            # Lower entropy = more concentrated = crisis
            threshold = spectral_df['eigenvalue_entropy_60d'].quantile(0.1)
            crisis_periods = spectral_df['eigenvalue_entropy_60d'] < threshold
            ax.fill_between(
                spectral_df.index,
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                where=crisis_periods,
                alpha=0.3,
                color='red',
                label='Low Diversity (Crisis)'
            )
            ax.legend()
        
        # 4. Absorption ratio
        ax = axes[1, 1]
        if 'absorption_ratio_1_60d' in spectral_df.columns:
            spectral_df['absorption_ratio_1_60d'].plot(ax=ax, label='AR(1)', alpha=0.7)
            if 'absorption_ratio_3_60d' in spectral_df.columns:
                spectral_df['absorption_ratio_3_60d'].plot(ax=ax, label='AR(3)', alpha=0.7)
            ax.set_ylabel('Absorption Ratio')
            ax.set_title('Variance Concentration (Higher = Crisis)')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cgecd_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {output_dir}/cgecd_analysis.png")
        
    except Exception as e:
        print(f"Visualization failed: {e}")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    results, spectral_features, targets = run_cgecd_experiment()