#!/usr/bin/env python3

import os
import logging
import copy
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool
from torch_geometric.utils import softmax

from metrics import MetricsCalculator

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY") or os.getenv("API_KEY")

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)

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

ALL_SYMBOLS = []
SYMBOL_CATEGORIES = {}
for category, symbols in ASSET_UNIVERSE.items():
    ALL_SYMBOLS.extend(symbols)
    for sym in symbols:
        SYMBOL_CATEGORIES[sym] = category


@dataclass
class ResearchConfig:
    lookback_days: int = 7500
    min_observations: int = 252
    max_lag: int = 5
    min_correlation: float = 0.10
    granger_significance: float = 0.05
    correlation_threshold: float = 0.25
    edge_threshold: float = 0.25  
    significance_threshold: float = 0.10  
    window_size: int = 60
    step_size: int = 2
    node_features: int = 20
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    num_regimes: int = 3
    edge_hidden_dim: int = 32
    edge_temperature: float = 0.3
    edge_top_k: int = 15
    use_learned_edges: bool = True
    seq_len: int = 15
    batch_size: int = 16
    learning_rate: float = 0.0005
    weight_decay: float = 0.01
    epochs: int = 150
    early_stopping_patience: int = 30
    dropout: float = 0.3
    edge_sparsity_weight: float = 0.005
    edge_entropy_weight: float = 0.0005
    label_smoothing: float = 0.1
    crisis_weight_multiplier: float = 8.0
    focal_loss_gamma: float = 2.0
    crisis_loss_weight: float = 0.7
    temporal_consistency_weight: float = 0.1
    n_splits: int = 5
    purge_gap: int = 5
    regime_method: str = 'quantile'
    save_models: bool = True
    save_graphs: bool = True
    run_baselines: bool = True
    crisis_augment_factor: int = 2
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = inputs.size(-1)
        
        if self.label_smoothing > 0:
            with torch.no_grad():
                true_dist = torch.zeros_like(inputs)
                true_dist.fill_(self.label_smoothing / (n_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            log_probs = F.log_softmax(inputs, dim=-1)
            if self.alpha is not None:
                log_probs = log_probs * self.alpha.unsqueeze(0)
            
            ce_loss = (-true_dist * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class EODHDClient:
    BASE_URL = "https://eodhd.com/api"
    CACHE_DIR = Path("cache")
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.CACHE_DIR.mkdir(exist_ok=True)
        
    def _get_cache_path(self, symbol: str, exchange: str, start_date: str, end_date: str) -> Path:
        filename = f"{symbol}.{exchange}_{start_date}_{end_date}.pkl"
        return self.CACHE_DIR / filename
    
    def get_eod_data(
        self, 
        symbol: str, 
        exchange: str, 
        start_date: str = None, 
        end_date: str = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        cache_path = self._get_cache_path(symbol, exchange, start_date or '', end_date or '')
        
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                    return df
            except:
                pass
        
        params = {'api_token': self.api_key, 'fmt': 'json'}
        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date
            
        try:
            response = self.session.get(
                f"{self.BASE_URL}/eod/{symbol}.{exchange}", 
                params=params, 
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.sort_index()
            
            if use_cache and not df.empty:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(df, f)
                except:
                    pass
            
            return df
            
        except:
            return pd.DataFrame()


class StatisticalLeadLagDetector:
    def __init__(self, config: ResearchConfig):
        self.config = config
        
    def _compute_granger_causality(
        self, 
        y: np.ndarray, 
        x: np.ndarray, 
        max_lag: int
    ) -> Tuple[int, float, float]:
        data = pd.DataFrame({'y': y, 'x': x})
        
        try:
            results = grangercausalitytests(data[['y', 'x']], maxlag=max_lag, verbose=False)
            
            best_lag = 1
            best_p = 1.0
            best_f = 0.0
            
            for lag in range(1, max_lag + 1):
                f_stat = results[lag][0]['ssr_ftest'][0]
                p_value = results[lag][0]['ssr_ftest'][1]
                
                if p_value < best_p:
                    best_p = p_value
                    best_lag = lag
                    best_f = f_stat
            
            return best_lag, best_p, best_f
            
        except:
            return 0, 1.0, 0.0
    
    def detect_relationships(
        self, 
        returns_dict: Dict[str, pd.Series],
        method: str = 'correlation'
    ) -> pd.DataFrame:
        symbols = list(returns_dict.keys())
        results = []
        
        df = pd.DataFrame(returns_dict).dropna()
        
        if len(df) < self.config.min_observations:
            return pd.DataFrame()
        
        total_pairs = len(symbols) * (len(symbols) - 1) // 2
        logger.info(f"Testing {total_pairs} pairs for lead-lag relationships...")
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i >= j:
                    continue
                
                s1 = df[sym1].values
                s2 = df[sym2].values
                
                relationship = {
                    'asset1': sym1,
                    'asset2': sym2,
                }
                
                best_corr = 0
                best_lag = 0
                best_direction = None
                
                for lag in range(1, self.config.max_lag + 1):
                    if len(s1[:-lag]) > 0:
                        corr_1_leads = np.corrcoef(s1[:-lag], s2[lag:])[0, 1]
                        if not np.isnan(corr_1_leads) and abs(corr_1_leads) > abs(best_corr):
                            best_corr = corr_1_leads
                            best_lag = lag
                            best_direction = (sym1, sym2)
                    
                    if len(s2[:-lag]) > 0:
                        corr_2_leads = np.corrcoef(s2[:-lag], s1[lag:])[0, 1]
                        if not np.isnan(corr_2_leads) and abs(corr_2_leads) > abs(best_corr):
                            best_corr = corr_2_leads
                            best_lag = lag
                            best_direction = (sym2, sym1)
                
                relationship['corr_leader'] = best_direction[0] if best_direction else None
                relationship['corr_follower'] = best_direction[1] if best_direction else None
                relationship['corr_lag'] = best_lag
                relationship['correlation'] = best_corr
                relationship['abs_correlation'] = abs(best_corr)
                
                sym1_base = sym1.replace('.US', '')
                sym2_base = sym2.replace('.US', '')
                relationship['leader_category'] = SYMBOL_CATEGORIES.get(sym1_base, 'unknown')
                relationship['follower_category'] = SYMBOL_CATEGORIES.get(sym2_base, 'unknown')
                
                results.append(relationship)
        
        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            df_results = df_results[df_results['abs_correlation'] >= self.config.min_correlation]
            df_results = df_results.sort_values('abs_correlation', ascending=False)
        
        logger.info(f"Detected {len(df_results)} significant relationships")
        
        if not df_results.empty:
            df_results.to_csv('output/lead_lag_relationships.csv', index=False)
        
        return df_results


class EnhancedFeatureExtractor:
    """Enhanced feature extractor with market-relative features."""
    
    def __init__(self):
        self.market_vol = None
        self.market_ret = None
    
    def set_market_context(self, market_vol: float, market_ret: float):
        """Set market-wide statistics for relative features."""
        self.market_vol = market_vol
        self.market_ret = market_ret
    
    @staticmethod
    def compute_features(
        returns: np.ndarray, 
        window_size: int = 20,
        market_vol: float = None,
        market_ret: float = None
    ) -> np.ndarray:
        """Compute 20 features including 4 market-relative features."""
        if len(returns) < 5:
            return np.zeros(20)
        
        vol_realized = np.std(returns) * np.sqrt(252)
        vol_ewma = pd.Series(returns).ewm(span=10).std().iloc[-1] * np.sqrt(252) if len(returns) > 10 else vol_realized
        vol_parkinson = np.std(returns) * np.sqrt(252) * 1.67 if len(returns) > 5 else vol_realized
        
        mean_return = np.mean(returns) * 252
        cum_return = np.sum(returns)
        
        skewness = stats.skew(returns) if len(returns) > 2 else 0
        kurt = stats.kurtosis(returns) if len(returns) > 3 else 0
        
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / (running_max + 1e-8)
        max_dd = drawdown.min()
        current_dd = drawdown[-1] if len(drawdown) > 0 else 0
        
        acf_1 = pd.Series(returns).autocorr(lag=1) if len(returns) > 1 else 0
        acf_1 = 0 if np.isnan(acf_1) else acf_1
        acf_5 = pd.Series(returns).autocorr(lag=5) if len(returns) > 5 else 0
        acf_5 = 0 if np.isnan(acf_5) else acf_5
        
        if len(returns) > 2:
            x = np.arange(len(returns))
            slope = np.polyfit(x, returns, 1)[0]
        else:
            slope = 0
        
        sharpe = (mean_return / vol_realized) if vol_realized > 0 else 0
        
        if market_vol is not None and market_vol > 0:
            relative_vol = vol_realized / market_vol
            vol_ratio = min(vol_realized / market_vol, 3.0)
        else:
            relative_vol = 1.0
            vol_ratio = 1.0
        
        if market_ret is not None:
            relative_return = mean_return - market_ret
        else:
            relative_return = 0.0
        
        if len(returns) > 5:
            recent_ret = np.mean(returns[-5:])
            older_ret = np.mean(returns[:-5]) if len(returns) > 10 else np.mean(returns)
            momentum_persistence = 1.0 if (recent_ret > 0) == (older_ret > 0) else -1.0
        else:
            momentum_persistence = 0.0
        
        return np.array([
            vol_realized, vol_ewma, vol_parkinson, mean_return, cum_return,
            skewness, kurt, var_95, var_99, cvar_95, max_dd, current_dd,
            acf_1, acf_5, slope, sharpe,
            relative_vol, vol_ratio, relative_return, momentum_persistence
        ])


class MultiMethodGraphBuilder:
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.feature_extractor = EnhancedFeatureExtractor()
        self.scaler = StandardScaler()
        
    def _build_full_graph_for_learning(
        self,
        returns_df: pd.DataFrame,
        symbols: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        corr_matrix = returns_df.corr().values
        n_symbols = len(symbols)
        
        edge_index = []
        edge_attr = []
        
        for i in range(n_symbols):
            for j in range(n_symbols):
                corr = corr_matrix[i, j] if i != j else 1.0
                corr = corr if not np.isnan(corr) else 0.0
                edge_index.append([i, j])
                edge_attr.append([corr, abs(corr), 1.0 if i == j else 0.0])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return edge_index, edge_attr
    
    def build_temporal_graphs(
        self, 
        returns_dict: Dict[str, pd.Series],
        use_full_graph: bool = True
    ) -> Tuple[List[Data], List[str]]:
        symbols = list(returns_dict.keys())
        n_symbols = len(symbols)
        
        df = pd.DataFrame(returns_dict).dropna()
        dates = df.index.tolist()
        
        if len(dates) < self.config.window_size + 20:
            return [], symbols
        
        graphs = []
        
        for end_idx in range(self.config.window_size, len(dates), self.config.step_size):
            window_df = df.iloc[end_idx - self.config.window_size:end_idx]
            
            market_vol = window_df.std().mean() * np.sqrt(252)
            market_ret = window_df.mean().mean() * 252
            
            node_features = []
            for sym in symbols:
                features = self.feature_extractor.compute_features(
                    window_df[sym].values,
                    window_size=self.config.window_size,
                    market_vol=market_vol,
                    market_ret=market_ret
                )
                node_features.append(features)
            
            node_features = np.array(node_features)
            
            edge_index, edge_attr = self._build_full_graph_for_learning(window_df, symbols)
            
            x = torch.tensor(node_features, dtype=torch.float)
            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=n_symbols
            )
            graph.timestamp = dates[end_idx - 1]
            
            corr_matrix = window_df.corr().values
            graph.corr_matrix = torch.tensor(corr_matrix, dtype=torch.float)
            
            graphs.append(graph)
        
        logger.info(f"Built {len(graphs)} temporal graphs")
        
        if graphs:
            graphs = self.scale_graph_features(graphs)
        
        return graphs, symbols
    
    def scale_graph_features(self, graphs: List[Data]) -> List[Data]:
        if not graphs:
            return graphs
        
        all_features = torch.cat([g.x for g in graphs], dim=0).numpy()
        
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=1e6, neginf=-1e6)
        all_features = np.clip(all_features, -1e6, 1e6)
        
        self.scaler.fit(all_features)
        
        scaled_graphs = []
        for graph in graphs:
            features = graph.x.numpy()
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            features = np.clip(features, -1e6, 1e6)
            scaled_x = self.scaler.transform(features)
            
            scaled_graph = Data(
                x=torch.tensor(scaled_x, dtype=torch.float),
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                num_nodes=graph.num_nodes
            )
            scaled_graph.timestamp = graph.timestamp
            scaled_graph.corr_matrix = graph.corr_matrix
            scaled_graphs.append(scaled_graph)
        
        return scaled_graphs


class LearnedEdgeConstructor(nn.Module):
    def __init__(self, config: ResearchConfig):
        super().__init__()
        self.config = config
        self.node_features = config.node_features
        self.edge_hidden_dim = config.edge_hidden_dim
        self.num_regimes = config.num_regimes
        
        edge_input_dim = config.node_features * 2 + 3
        
        self.edge_scorer = nn.Sequential(
            nn.Linear(edge_input_dim, config.edge_hidden_dim),
            nn.LayerNorm(config.edge_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.dropout),
            nn.Linear(config.edge_hidden_dim, 1)
        )
        
        self.edge_proj = nn.Linear(edge_input_dim, config.edge_hidden_dim)
        
        self.regime_encoder = nn.Sequential(
            nn.Linear(config.node_features, config.edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.edge_hidden_dim, config.num_regimes)
        )
        
        self.regime_edge_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.edge_hidden_dim, config.edge_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.edge_hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(config.num_regimes)
        ])
        
        self.contagion_detector = nn.Sequential(
            nn.Linear(config.node_features, config.edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.edge_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.temperature = config.edge_temperature
        self.top_k = config.edge_top_k
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_analysis: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        num_nodes = x.size(0)
        device = x.device
        
        global_state = x.mean(dim=0)
        regime_logits = self.regime_encoder(global_state)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        contagion_scores = self.contagion_detector(x)
        contagion_level = contagion_scores.mean()
        
        src_nodes = edge_index[0]
        tgt_nodes = edge_index[1]
        
        src_features = x[src_nodes]
        tgt_features = x[tgt_nodes]
        
        edge_input = torch.cat([src_features, tgt_features, edge_attr], dim=-1)
        
        edge_scores_raw = self.edge_scorer(edge_input)
        
        edge_hidden = F.relu(self.edge_proj(edge_input))
        
        regime_gates = []
        for regime_idx in range(self.num_regimes):
            gate = self.regime_edge_gates[regime_idx](edge_hidden)
            regime_gates.append(gate)
        
        regime_gates = torch.stack(regime_gates, dim=-1).squeeze(1)
        regime_gate_combined = (regime_gates * regime_probs.unsqueeze(0)).sum(dim=-1, keepdim=True)
        
        contagion_amplifier = 1.0 + contagion_level * 0.5
        
        edge_scores = edge_scores_raw * regime_gate_combined * contagion_amplifier
        
        edge_scores_matrix = torch.full((num_nodes, num_nodes), float('-inf'), device=device)
        edge_scores_flat = edge_scores.squeeze(-1)
        edge_scores_matrix[src_nodes, tgt_nodes] = edge_scores_flat
        
        edge_probs_matrix = F.softmax(edge_scores_matrix / self.temperature, dim=-1)
        
        k = min(self.top_k, num_nodes)
        topk_values, topk_indices = torch.topk(edge_probs_matrix, k, dim=-1)
        
        new_edge_list = []
        new_edge_weights = []
        
        for src in range(num_nodes):
            for kid in range(k):
                tgt = topk_indices[src, kid].item()
                weight = topk_values[src, kid].item()
                if weight > 1e-6:
                    new_edge_list.append([src, tgt])
                    new_edge_weights.append(weight)
        
        if len(new_edge_list) == 0:
            new_edge_index = edge_index
            new_edge_weights_tensor = F.softmax(edge_scores_flat / self.temperature, dim=0)
            new_edge_attr = torch.stack([new_edge_weights_tensor, new_edge_weights_tensor], dim=-1)
            num_edges_kept = new_edge_index.size(1)
            avg_edge_weight = new_edge_weights_tensor.mean()
        else:
            new_edge_index = torch.tensor(new_edge_list, dtype=torch.long, device=device).t()
            new_edge_weights_tensor = torch.tensor(new_edge_weights, dtype=torch.float, device=device)
            new_edge_attr = torch.stack([new_edge_weights_tensor, new_edge_weights_tensor], dim=-1)
            num_edges_kept = len(new_edge_list)
            avg_edge_weight = new_edge_weights_tensor.mean()
        
        analysis = {
            'regime_probs': regime_probs.detach(),
            'contagion_level': contagion_level.detach(),
            'edge_scores': edge_scores.detach(),
            'num_edges_kept': num_edges_kept,
            'avg_edge_weight': avg_edge_weight.detach() if isinstance(avg_edge_weight, torch.Tensor) else torch.tensor(avg_edge_weight)
        }
        
        return new_edge_index, new_edge_attr, analysis


class RegimeAdaptiveGNNLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, num_regimes: int = 3, dropout: float = 0.3):
        super().__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_regimes = num_regimes
        self.dropout = dropout
        
        self.regime_transforms = nn.ModuleList([
            nn.Linear(in_channels, out_channels)
            for _ in range(num_regimes)
        ])
        
        self.att_src = nn.Linear(in_channels, 1)
        self.att_tgt = nn.Linear(in_channels, 1)
        
        self.out_proj = nn.Linear(out_channels * num_regimes, out_channels)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for transform in self.regime_transforms:
            nn.init.xavier_uniform_(transform.weight)
            nn.init.zeros_(transform.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        regime_probs: torch.Tensor
    ) -> torch.Tensor:
        alpha_src = self.att_src(x)
        alpha_tgt = self.att_tgt(x)
        
        out = self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            alpha_src=alpha_src,
            alpha_tgt=alpha_tgt,
            regime_probs=regime_probs
        )
        
        return out
    
    def message(
        self,
        x_j: torch.Tensor,
        edge_weight: torch.Tensor,
        alpha_src_i: torch.Tensor,
        alpha_tgt_j: torch.Tensor,
        regime_probs: torch.Tensor,
        index: torch.Tensor
    ) -> torch.Tensor:
        alpha = alpha_src_i + alpha_tgt_j
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index)
        
        alpha = alpha * edge_weight.unsqueeze(-1)
        
        regime_messages = []
        for r in range(self.num_regimes):
            msg = self.regime_transforms[r](x_j)
            msg = msg * regime_probs[r]
            regime_messages.append(msg)
        
        combined = torch.cat(regime_messages, dim=-1)
        
        return alpha * self.out_proj(combined)


class TemporalGNNWithLearnedEdges(nn.Module):
    def __init__(self, config: ResearchConfig):
        super().__init__()
        self.config = config
        
        self.edge_constructor = LearnedEdgeConstructor(config)
        
        self.input_proj = nn.Linear(config.node_features, config.hidden_dim)
        
        self.gnn_layers = nn.ModuleList([
            RegimeAdaptiveGNNLayer(
                config.hidden_dim,
                config.hidden_dim,
                config.num_regimes,
                config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        self.gcn_layers = nn.ModuleList([
            GCNConv(config.hidden_dim, config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, config.hidden_dim) * 0.02)
        
        self.temporal_gru = nn.GRU(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.regime_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_regimes)
        )
        
        self.contagion_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        self.last_edge_analysis = []
        
    def forward(
        self,
        graph_sequence: List[Data],
        return_analysis: bool = False
    ) -> Dict[str, torch.Tensor]:
        sequence_embeddings = []
        all_edge_analysis = []
        
        for t, graph in enumerate(graph_sequence):
            x = graph.x.to(DEVICE)
            edge_index = graph.edge_index.to(DEVICE)
            edge_attr = graph.edge_attr.to(DEVICE)
            
            if self.config.use_learned_edges:
                learned_edge_index, learned_edge_attr, edge_analysis = self.edge_constructor(
                    x, edge_index, edge_attr, return_analysis=True
                )
                all_edge_analysis.append(edge_analysis)
                
                edge_index = learned_edge_index
                edge_weight = learned_edge_attr[:, 0]
                regime_probs = edge_analysis['regime_probs']
            else:
                edge_weight = edge_attr[:, 1] if edge_attr.size(1) > 1 else torch.ones(edge_attr.size(0), device=DEVICE)
                regime_probs = torch.ones(self.config.num_regimes, device=DEVICE) / self.config.num_regimes
            
            h = F.relu(self.input_proj(x))
            h = F.dropout(h, p=self.config.dropout, training=self.training)
            
            for i, (gnn_layer, gcn_layer, norm) in enumerate(
                zip(self.gnn_layers, self.gcn_layers, self.layer_norms)
            ):
                h_regime = gnn_layer(h, edge_index, edge_weight, regime_probs)
                h_gcn = gcn_layer(h, edge_index, edge_weight)
                
                h_new = h_regime + 0.3 * h_gcn
                h_new = norm(h_new)
                h_new = F.relu(h_new)
                h_new = F.dropout(h_new, p=self.config.dropout, training=self.training)
                
                h = h + h_new
            
            graph_embedding = global_mean_pool(h, torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE))
            sequence_embeddings.append(graph_embedding.squeeze(0))
        
        self.last_edge_analysis = all_edge_analysis
        
        seq_tensor = torch.stack(sequence_embeddings).unsqueeze(0)
        
        seq_len = seq_tensor.size(1)
        seq_tensor = seq_tensor + self.pos_encoding[:, :seq_len, :]
        
        attn_out, attn_weights = self.temporal_attention(seq_tensor, seq_tensor, seq_tensor)
        
        gru_out, _ = self.temporal_gru(seq_tensor)
        
        attn_final = attn_out[:, -1, :]
        gru_final = gru_out[:, -1, :]
        
        fused = torch.cat([attn_final, gru_final], dim=-1)
        fused = self.fusion(fused)
        
        regime_logits = self.regime_head(fused)
        contagion_prob = self.contagion_head(fused)
        volatility_pred = self.volatility_head(fused)
        
        output = {
            'regime_logits': regime_logits,
            'regime_probs': F.softmax(regime_logits, dim=-1),
            'contagion_probability': contagion_prob,
            'volatility_forecast': volatility_pred,
            'embedding': fused
        }
        
        if return_analysis:
            output['analysis'] = {
                'edge_analyses': all_edge_analysis,
                'attention_weights': attn_weights.detach()
            }
        
        return output


class RegimeLabeler:
    def __init__(self, config: ResearchConfig):
        self.config = config

    def label_regimes(
        self, 
        returns_dict: Dict[str, pd.Series],
        method: str = None
    ) -> pd.DataFrame:
        if method is None:
            method = self.config.regime_method
            
        df = pd.DataFrame(returns_dict).dropna()
        
        features_list = []
        dates = []
        window = 30
        
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i]
            
            avg_vol = window_data.std().mean() * np.sqrt(252)
            max_vol = window_data.std().max() * np.sqrt(252)
            corr_matrix = window_data.corr().values
            avg_corr = np.nanmean(corr_matrix[np.triu_indices(len(corr_matrix), k=1)])
            avg_ret = window_data.mean().mean() * 252
            
            pct_negative = (window_data.mean(axis=1) < 0).sum() / len(window_data)
            
            cumsum = window_data.cumsum()
            running_max = cumsum.max()
            with np.errstate(divide='ignore', invalid='ignore'):
                drawdown = (cumsum - running_max) / running_max.replace(0, 1)
                max_drawdown = drawdown.min().min()
                if np.isinf(max_drawdown) or np.isnan(max_drawdown):
                    max_drawdown = 0
            
            features_list.append([
                avg_vol, max_vol,
                avg_corr if not np.isnan(avg_corr) else 0, 
                avg_ret, pct_negative, max_drawdown
            ])
            dates.append(df.index[i])
        
        features = np.array(features_list)
        features = np.clip(features, -1e10, 1e10)
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if method == 'quantile':
            regimes = self._label_by_quantile(features)
        elif method == 'adaptive':
            regimes = self._label_adaptive(features)
        elif method == 'kmeans':
            regimes = self._label_by_kmeans(features)
        else:
            regimes = self._label_by_quantile(features)
        
        unique, counts = np.unique(regimes, return_counts=True)
        logger.info(f"Regime labeling method: {method}")
        for r, c in zip(unique, counts):
            pct = c / len(regimes) * 100
            logger.info(f"   Regime {r}: {c} days ({pct:.1f}%)")
        
        return pd.DataFrame({
            'date': dates,
            'regime': regimes,
            'volatility': features[:, 0],
            'max_volatility': features[:, 1],
            'correlation': features[:, 2],
            'return': features[:, 3],
            'pct_negative': features[:, 4],
            'max_drawdown': features[:, 5]
        })
    
    def _label_by_quantile(self, features: np.ndarray) -> np.ndarray:
        volatility = features[:, 0]
        correlation = features[:, 2]
        
        vol_z = (volatility - np.mean(volatility)) / (np.std(volatility) + 1e-8)
        corr_z = (correlation - np.mean(correlation)) / (np.std(correlation) + 1e-8)
        
        stress_score = vol_z + 0.5 * corr_z
        
        q_low = np.percentile(stress_score, 40)
        q_high = np.percentile(stress_score, 80)
        
        regimes = np.ones(len(stress_score), dtype=int)
        regimes[stress_score <= q_low] = 0
        regimes[stress_score > q_high] = 2
        
        return regimes
    
    def _label_adaptive(self, features: np.ndarray) -> np.ndarray:
        volatility = features[:, 0]
        correlation = features[:, 2]
        
        regimes = np.ones(len(volatility), dtype=int)
        
        lookback = 252
        
        for i in range(lookback, len(volatility)):
            hist_vol = volatility[max(0, i-lookback):i]
            hist_corr = correlation[max(0, i-lookback):i]
            
            current_vol = volatility[i]
            current_corr = correlation[i]
            
            vol_percentile = stats.percentileofscore(hist_vol, current_vol)
            corr_percentile = stats.percentileofscore(hist_corr, current_corr)
            
            if vol_percentile < 40:
                regimes[i] = 0
            elif vol_percentile > 80 and corr_percentile > 70:
                regimes[i] = 2
            elif vol_percentile > 70:
                regimes[i] = 1
            else:
                regimes[i] = 1
        
        if lookback > 0 and lookback < len(volatility):
            early_regimes = self._label_by_quantile(features[:lookback])
            regimes[:lookback] = early_regimes
        
        return regimes
    
    def _label_by_kmeans(self, features: np.ndarray) -> np.ndarray:
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=self.config.num_regimes, random_state=SEED, n_init=20)
        regimes = kmeans.fit_predict(features_scaled)
        
        regime_vols = {r: features[regimes == r, 0].mean() 
                      for r in range(self.config.num_regimes)}
        sorted_regimes = sorted(regime_vols.keys(), key=lambda x: regime_vols[x])
        mapping = {sorted_regimes[i]: i for i in range(self.config.num_regimes)}
        regimes = np.array([mapping[r] for r in regimes])
        
        silhouette = silhouette_score(features_scaled, regimes)
        logger.info(f"K-Means silhouette score: {silhouette:.3f}")
        
        return regimes


class ResearchTrainer:
    def __init__(self, model: nn.Module, config: ResearchConfig):
        self.model = model.to(DEVICE)
        self.config = config
        self.metrics_history = defaultdict(list)
        
    def _augment_crisis_samples(
        self, 
        sequences: List, 
        labels: List, 
        volatilities: List, 
        augment_factor: int = 2
    ) -> Tuple[List, List, List]:
        """Duplicate crisis samples for better learning."""
        augmented_seqs = list(sequences)
        augmented_labels = list(labels)
        augmented_vols = list(volatilities)
        
        for i, label in enumerate(labels):
            if label == 2:
                for _ in range(augment_factor):
                    augmented_seqs.append(sequences[i])
                    augmented_labels.append(label)
                    augmented_vols.append(volatilities[i])
        
        return augmented_seqs, augmented_labels, augmented_vols
        
    def _compute_edge_regularization(self, model: nn.Module) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=DEVICE)
        
        if hasattr(model, 'last_edge_analysis') and model.last_edge_analysis:
            sparsity_sum = 0.0
            entropy_sum = torch.tensor(0.0, device=DEVICE)
            count = 0
            
            for edge_analysis in model.last_edge_analysis:
                num_edges = edge_analysis['num_edges_kept']
                num_nodes = self.config.node_features
                target_edges = self.config.edge_top_k * num_nodes
                
                sparsity_penalty = max(0.0, (num_edges - target_edges) / max(target_edges, 1))
                sparsity_sum += sparsity_penalty
                
                avg_weight = edge_analysis['avg_edge_weight']
                if isinstance(avg_weight, torch.Tensor):
                    entropy_penalty = -avg_weight * torch.log(avg_weight + 1e-8)
                    entropy_sum = entropy_sum + entropy_penalty
                
                count += 1
            
            if count > 0:
                reg_loss = (
                    self.config.edge_sparsity_weight * sparsity_sum / count +
                    self.config.edge_entropy_weight * entropy_sum / count
                )
        
        return reg_loss
    
    def _compute_crisis_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        crisis_mask = (targets == 2)
        if crisis_mask.sum() == 0:
            return torch.tensor(0.0, device=DEVICE)
        
        crisis_logits = logits[crisis_mask]
        crisis_probs = F.softmax(crisis_logits, dim=-1)[:, 2]
        
        crisis_loss = -torch.log(crisis_probs + 1e-8).mean()
        return crisis_loss
    
    def _compute_temporal_consistency_loss(
        self, 
        regime_probs_sequence: List[torch.Tensor]
    ) -> torch.Tensor:
        """Penalize rapid regime probability changes."""
        if len(regime_probs_sequence) < 2:
            return torch.tensor(0.0, device=DEVICE)
        
        consistency_loss = torch.tensor(0.0, device=DEVICE)
        for t in range(1, len(regime_probs_sequence)):
            diff = regime_probs_sequence[t] - regime_probs_sequence[t-1]
            consistency_loss = consistency_loss + (diff ** 2).sum()
        
        return consistency_loss / len(regime_probs_sequence)
        
    def train_with_cross_validation(
        self,
        graphs: List[Data],
        regime_df: pd.DataFrame
    ) -> Dict[str, Any]:
        regime_df['date'] = pd.to_datetime(regime_df['date'])
        
        sequences = []
        labels = []
        volatilities = []
        
        for i in range(len(graphs) - self.config.seq_len):
            seq = graphs[i:i + self.config.seq_len]
            ts = seq[-1].timestamp
            
            match = regime_df[regime_df['date'] <= ts]
            if len(match) > 0:
                sequences.append(seq)
                labels.append(match.iloc[-1]['regime'])
                volatilities.append(match.iloc[-1]['volatility'])
        
        logger.info(f"Created {len(sequences)} training sequences")
        
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"Label distribution in sequences:")
        for u, c in zip(unique, counts):
            logger.info(f"   Regime {u}: {c} ({c/len(labels)*100:.1f}%)")
        
        if len(sequences) < 20:
            return {}
        
        labels_array = np.array(labels)
        
        # Use crisis-aware cross-validation
        from benchmarks import CrisisAwareTimeSeriesSplit
        tscv = CrisisAwareTimeSeriesSplit(
            n_splits=self.config.n_splits, 
            purge_gap=self.config.purge_gap,
            min_crisis_train=20,
            min_crisis_val=5,
            val_ratio=0.15
        )
        
        fold_results = []
        all_preds = []
        all_probs = []
        all_labels_collected = []
        
        best_overall_model_state = None
        best_overall_crisis_recall = 0
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(len(sequences), labels_array)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Fold {fold + 1}/{self.config.n_splits}")
            logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
            logger.info(f"{'='*60}")
            
            train_sequences = [sequences[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            train_vols = [volatilities[i] for i in train_idx]
            
            val_sequences = [sequences[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            val_vols = [volatilities[i] for i in val_idx]
            
            train_regime_dist = np.bincount(train_labels, minlength=3)
            val_regime_dist = np.bincount(val_labels, minlength=3)
            
            logger.info(f"Train regimes (before augment): Bull={train_regime_dist[0]}, Normal={train_regime_dist[1]}, Crisis={train_regime_dist[2]}")
            logger.info(f"Val regimes: Bull={val_regime_dist[0]}, Normal={val_regime_dist[1]}, Crisis={val_regime_dist[2]}")
            
            if train_regime_dist[2] < 10:
                logger.warning(f"Fold {fold+1} has only {train_regime_dist[2]} crisis samples in train, skipping...")
                continue
            
            if val_regime_dist[2] < 3:
                logger.warning(f"Fold {fold+1} has only {val_regime_dist[2]} crisis samples in val, skipping...")
                continue
            
            train_sequences, train_labels, train_vols = self._augment_crisis_samples(
                train_sequences, train_labels, train_vols, 
                augment_factor=self.config.crisis_augment_factor
            )
            
            train_regime_dist_aug = np.bincount(train_labels, minlength=3)
            logger.info(f"Train regimes (after augment): Bull={train_regime_dist_aug[0]}, Normal={train_regime_dist_aug[1]}, Crisis={train_regime_dist_aug[2]}")
            
            fold_metrics, fold_preds, fold_probs = self._train_fold(
                train_sequences, train_labels, train_vols,
                val_sequences, val_labels, val_vols,
                fold
            )
            
            fold_crisis_recall = 0
            crisis_mask = np.array(val_labels) == 2
            if crisis_mask.sum() > 0:
                fold_crisis_recall = (np.array(fold_preds)[crisis_mask] == 2).mean()
            
            if fold_crisis_recall > best_overall_crisis_recall:
                best_overall_crisis_recall = fold_crisis_recall
                best_overall_model_state = copy.deepcopy(self.model.state_dict())
                logger.info(f"New best crisis recall: {fold_crisis_recall:.2%}")
            
            fold_results.append(fold_metrics)
            all_preds.extend(fold_preds)
            all_probs.extend(fold_probs)
            all_labels_collected.extend(val_labels)
        
        if not fold_results:
            logger.error("No valid folds completed!")
            return {}
        
        if best_overall_model_state:
            torch.save(best_overall_model_state, 'output/models/best_crisis_model.pt')
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels_collected = np.array(all_labels_collected)
        
        overall_metrics = MetricsCalculator.calculate_all_metrics(
            all_labels_collected, all_preds, all_probs, self.config.num_regimes
        )
        
        avg_metrics = {
            'avg_val_acc': np.mean([f['best_val_acc'] for f in fold_results]),
            'avg_val_loss': np.mean([f['best_val_loss'] for f in fold_results]),
            'std_val_acc': np.std([f['best_val_acc'] for f in fold_results]),
            'avg_crisis_recall': np.mean([f['best_crisis_recall'] for f in fold_results]),
            'std_crisis_recall': np.std([f['best_crisis_recall'] for f in fold_results]),
            'fold_results': fold_results,
            'overall_metrics': overall_metrics,
            'all_predictions': all_preds,
            'all_probabilities': all_probs,
            'all_labels': all_labels_collected
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("Cross-Validation Results")
        logger.info(f"{'='*60}")
        logger.info(f"Completed Folds: {len(fold_results)}/{self.config.n_splits}")
        logger.info(f"Average Validation Accuracy: {avg_metrics['avg_val_acc']:.2%} ± {avg_metrics['std_val_acc']:.2%}")
        logger.info(f"Average Crisis Recall: {avg_metrics['avg_crisis_recall']:.2%} ± {avg_metrics['std_crisis_recall']:.2%}")
        logger.info(f"Average Validation Loss: {avg_metrics['avg_val_loss']:.4f}")
        logger.info(f"\nComprehensive Metrics:")
        logger.info(f"  {overall_metrics.summary()}")
        logger.info(f"  ROC-AUC (OvR): {overall_metrics.roc_auc_ovr:.3f}")
        logger.info(f"  ECE: {overall_metrics.expected_calibration_error:.4f}")
        logger.info(f"  Early Warning Score: {overall_metrics.early_warning_score:.3f}")
        
        with open('output/metrics/cv_results.json', 'w') as f:
            serializable_metrics = {
                'avg_val_acc': float(avg_metrics['avg_val_acc']),
                'std_val_acc': float(avg_metrics['std_val_acc']),
                'avg_val_loss': float(avg_metrics['avg_val_loss']),
                'avg_crisis_recall': float(avg_metrics['avg_crisis_recall']),
                'std_crisis_recall': float(avg_metrics['std_crisis_recall']),
                'n_folds': len(fold_results),
                'overall_metrics': overall_metrics.to_dict()
            }
            json.dump(serializable_metrics, f, indent=2)
        
        return avg_metrics
    
    def _train_fold(
        self,
        train_sequences, train_labels, train_vols,
        val_sequences, val_labels, val_vols,
        fold: int
    ) -> Tuple[Dict[str, float], List[int], List[np.ndarray]]:
        self.model.apply(self._init_weights)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        class_weights = torch.tensor([1.0, 1.5, self.config.crisis_weight_multiplier], dtype=torch.float, device=DEVICE)
        
        criterion_regime = FocalLoss(
            alpha=class_weights, 
            gamma=self.config.focal_loss_gamma,
            label_smoothing=self.config.label_smoothing
        )
        criterion_vol = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_crisis_recall = 0.0
        patience_counter = 0
        best_preds = []
        best_probs = []
        
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            
            indices = np.random.permutation(len(train_sequences))
            
            for idx in indices:
                seq = train_sequences[idx]
                label = train_labels[idx]
                vol = train_vols[idx]
                
                optimizer.zero_grad()
                output = self.model(seq)
                
                target_regime = torch.tensor([label], device=DEVICE)
                target_vol = torch.tensor([[vol]], device=DEVICE, dtype=torch.float)
                
                loss_regime = criterion_regime(output['regime_logits'], target_regime)
                loss_crisis = self._compute_crisis_loss(output['regime_logits'], target_regime)
                loss_vol = criterion_vol(output['volatility_forecast'], target_vol)
                loss_edge_reg = self._compute_edge_regularization(self.model)
                
                loss_temporal = torch.tensor(0.0, device=DEVICE)
                if hasattr(self.model, 'last_edge_analysis') and self.model.last_edge_analysis:
                    regime_probs_seq = [ea['regime_probs'] for ea in self.model.last_edge_analysis]
                    loss_temporal = self._compute_temporal_consistency_loss(regime_probs_seq)
                
                loss = (
                    loss_regime + 
                    self.config.crisis_loss_weight * loss_crisis + 
                    0.1 * loss_vol + 
                    loss_edge_reg +
                    self.config.temporal_consistency_weight * loss_temporal
                )
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                if output['regime_logits'].argmax().item() == label:
                    train_correct += 1
            
            scheduler.step()
            
            train_loss /= len(train_sequences)
            train_acc = train_correct / len(train_sequences)
            
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_preds = []
            val_probs = []
            
            with torch.no_grad():
                for idx in range(len(val_sequences)):
                    seq = val_sequences[idx]
                    label = val_labels[idx]
                    vol = val_vols[idx]
                    
                    output = self.model(seq)
                    
                    target_regime = torch.tensor([label], device=DEVICE)
                    target_vol = torch.tensor([[vol]], device=DEVICE, dtype=torch.float)
                    
                    loss_regime = criterion_regime(output['regime_logits'], target_regime)
                    loss_vol = criterion_vol(output['volatility_forecast'], target_vol)
                    
                    loss = loss_regime + 0.1 * loss_vol
                    val_loss += loss.item()
                    
                    pred = output['regime_logits'].argmax().item()
                    prob = output['regime_probs'].cpu().numpy().flatten()
                    
                    val_preds.append(pred)
                    val_probs.append(prob)
                    
                    if pred == label:
                        val_correct += 1
            
            val_loss /= len(val_sequences)
            val_acc = val_correct / len(val_sequences)
            
            crisis_mask = np.array(val_labels) == 2
            crisis_recall = 0
            if crisis_mask.sum() > 0:
                crisis_recall = (np.array(val_preds)[crisis_mask] == 2).mean()
            
            combined_metric = 0.5 * val_acc + 0.5 * crisis_recall
            
            if val_loss < best_val_loss or (crisis_recall > best_crisis_recall and combined_metric > 0.5 * best_val_acc + 0.5 * best_crisis_recall):
                best_val_loss = min(best_val_loss, val_loss)
                best_val_acc = val_acc
                best_crisis_recall = crisis_recall
                best_preds = val_preds.copy()
                best_probs = [p.copy() for p in val_probs]
                patience_counter = 0
                
                if self.config.save_models:
                    torch.save(
                        self.model.state_dict(),
                        f'output/models/best_model_fold{fold}.pt'
                    )
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Fold {fold+1}, Epoch {epoch+1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}, "
                    f"Crisis Recall: {crisis_recall:.2%}, LR: {current_lr:.2e}"
                )
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return {
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'best_crisis_recall': best_crisis_recall
        }, best_preds, best_probs
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)