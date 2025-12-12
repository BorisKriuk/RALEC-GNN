#!/usr/bin/env python3

import os
import logging
import copy
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
import json
from pathlib import Path
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, log_loss,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool
from torch_geometric.utils import softmax

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import networkx as nx
import seaborn as sns

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY") or os.getenv("API_KEY")

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

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
    window_size: int = 60
    step_size: int = 5
    node_features: int = 16
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    num_regimes: int = 3
    edge_hidden_dim: int = 32
    edge_temperature: float = 0.5
    edge_top_k: int = 10
    use_learned_edges: bool = True
    seq_len: int = 15
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    epochs: int = 100
    early_stopping_patience: int = 20
    dropout: float = 0.4
    edge_sparsity_weight: float = 0.005
    edge_entropy_weight: float = 0.0005
    label_smoothing: float = 0.1
    crisis_weight_multiplier: float = 5.0
    focal_loss_gamma: float = 2.0
    crisis_loss_weight: float = 0.5
    n_splits: int = 5
    purge_gap: int = 5
    regime_method: str = 'quantile'
    save_models: bool = True
    save_graphs: bool = True
    run_baselines: bool = True
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class QuantMetrics:
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    cohen_kappa: float = 0.0
    mcc: float = 0.0
    precision_per_class: Dict[int, float] = field(default_factory=dict)
    recall_per_class: Dict[int, float] = field(default_factory=dict)
    f1_per_class: Dict[int, float] = field(default_factory=dict)
    log_loss_value: float = 0.0
    roc_auc_ovr: float = 0.0
    brier_score: float = 0.0
    crisis_recall: float = 0.0
    crisis_precision: float = 0.0
    regime_transition_accuracy: float = 0.0
    expected_calibration_error: float = 0.0
    early_warning_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                result[k] = {str(kk): float(vv) for kk, vv in v.items()}
            elif isinstance(v, (np.floating, float)):
                result[k] = float(v)
            else:
                result[k] = v
        return result
    
    def summary(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.2%} | Balanced: {self.balanced_accuracy:.2%} | "
            f"Macro-F1: {self.macro_f1:.3f} | Kappa: {self.cohen_kappa:.3f} | "
            f"Crisis Recall: {self.crisis_recall:.2%}"
        )


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


class MetricsCalculator:
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None,
        num_classes: int = 3
    ) -> QuantMetrics:
        metrics = QuantMetrics()
        
        metrics.accuracy = (y_true == y_pred).mean()
        metrics.balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(num_classes)), zero_division=0
        )
        
        metrics.macro_f1 = f1.mean()
        metrics.weighted_f1 = np.average(f1, weights=support) if support.sum() > 0 else 0.0
        
        for i in range(num_classes):
            metrics.precision_per_class[i] = precision[i]
            metrics.recall_per_class[i] = recall[i]
            metrics.f1_per_class[i] = f1[i]
        
        metrics.cohen_kappa = cohen_kappa_score(y_true, y_pred)
        metrics.mcc = matthews_corrcoef(y_true, y_pred)
        
        crisis_mask = y_true == 2
        if crisis_mask.sum() > 0:
            metrics.crisis_recall = recall[2]
            metrics.crisis_precision = precision[2]
        
        if y_prob is not None:
            try:
                metrics.log_loss_value = log_loss(y_true, y_prob, labels=list(range(num_classes)))
            except:
                metrics.log_loss_value = float('inf')
            
            try:
                y_true_onehot = np.zeros((len(y_true), num_classes))
                for i, label in enumerate(y_true):
                    y_true_onehot[i, int(label)] = 1
                metrics.roc_auc_ovr = roc_auc_score(
                    y_true_onehot, y_prob, 
                    multi_class='ovr', 
                    average='macro'
                )
            except:
                metrics.roc_auc_ovr = 0.5
            
            metrics.brier_score = MetricsCalculator._brier_multi(y_true, y_prob, num_classes)
            metrics.expected_calibration_error = MetricsCalculator._ece(y_true, y_prob)
        
        metrics.regime_transition_accuracy = MetricsCalculator._transition_accuracy(y_true, y_pred)
        metrics.early_warning_score = MetricsCalculator._early_warning_score(y_true, y_pred)
        
        return metrics
    
    @staticmethod
    def _brier_multi(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
        y_true_onehot = np.zeros((len(y_true), num_classes))
        for i, label in enumerate(y_true):
            y_true_onehot[i, int(label)] = 1
        return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))
    
    @staticmethod
    def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
        
        return ece
    
    @staticmethod
    def _transition_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        transitions = np.where(np.diff(y_true) != 0)[0] + 1
        if len(transitions) == 0:
            return 0.0
        return (y_true[transitions] == y_pred[transitions]).mean()
    
    @staticmethod
    def _early_warning_score(y_true: np.ndarray, y_pred: np.ndarray, lookahead: int = 5) -> float:
        score = 0.0
        count = 0
        
        for i in range(1, len(y_true)):
            if y_true[i] == 2 and y_true[i-1] != 2:
                start = max(0, i - lookahead)
                pre_crisis_preds = y_pred[start:i]
                if len(pre_crisis_preds) > 0:
                    elevated_risk = (pre_crisis_preds >= 1).mean()
                    score += elevated_risk
                    count += 1
        
        return score / max(count, 1)


class PurgedTimeSeriesSplit:
    def __init__(self, n_splits: int = 5, purge_gap: int = 5, min_train_size: int = 100):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.min_train_size = min_train_size
    
    def split(self, X):
        n_samples = len(X) if hasattr(X, '__len__') else X
        fold_size = (n_samples - self.min_train_size) // self.n_splits
        
        for i in range(self.n_splits):
            train_end = self.min_train_size + i * fold_size
            val_start = train_end + self.purge_gap
            val_end = min(val_start + fold_size, n_samples)
            
            if val_start >= n_samples:
                break
                
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            
            yield train_idx, val_idx


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
    @staticmethod
    def compute_features(returns: np.ndarray, window_size: int = 20) -> np.ndarray:
        if len(returns) < 5:
            return np.zeros(16)
        
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
        drawdown = (cum_returns - running_max) / running_max
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
        
        return np.array([
            vol_realized, vol_ewma, vol_parkinson, mean_return, cum_return,
            skewness, kurt, var_95, var_99, cvar_95, max_dd, current_dd,
            acf_1, acf_5, slope, sharpe
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
            
            node_features = []
            for sym in symbols:
                features = self.feature_extractor.compute_features(
                    window_df[sym].values,
                    window_size=self.config.window_size
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
        self.scaler.fit(all_features)
        
        scaled_graphs = []
        for graph in graphs:
            scaled_x = self.scaler.transform(graph.x.numpy())
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
    def __init__(self, in_channels: int, out_channels: int, num_regimes: int = 3, dropout: float = 0.4):
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


class BaselineModels:
    def __init__(self, config: ResearchConfig):
        self.config = config
        
    def prepare_features(
        self,
        graphs: List[Data],
        regime_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        regime_df['date'] = pd.to_datetime(regime_df['date'])
        
        features_list = []
        labels = []
        
        for i in range(len(graphs) - self.config.seq_len):
            seq = graphs[i:i + self.config.seq_len]
            ts = seq[-1].timestamp
            
            match = regime_df[regime_df['date'] <= ts]
            if len(match) > 0:
                seq_features = []
                for g in seq:
                    node_mean = g.x.mean(dim=0).numpy()
                    node_std = g.x.std(dim=0).numpy()
                    seq_features.extend(node_mean)
                    seq_features.extend(node_std)
                
                features_list.append(seq_features)
                labels.append(match.iloc[-1]['regime'])
        
        return np.array(features_list), np.array(labels)
    
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, Dict]:
        results = {}
        
        models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                multi_class='multinomial',
                class_weight='balanced',
                random_state=SEED
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=SEED,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=SEED
            )
        }
        
        tscv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_gap=self.config.purge_gap, min_train_size=50)
        
        for name, model in models.items():
            logger.info(f"Training baseline: {name}")
            
            fold_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(len(X))):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                X_train = np.nan_to_num(X_train, nan=0, posinf=1e10, neginf=-1e10)
                X_val = np.nan_to_num(X_val, nan=0, posinf=1e10, neginf=-1e10)
                
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                
                preds = model_clone.predict(X_val)
                probs = model_clone.predict_proba(X_val)
                
                fold_m = MetricsCalculator.calculate_all_metrics(y_val, preds, probs, self.config.num_regimes)
                fold_metrics.append(fold_m)
            
            results[name] = {
                'accuracy_mean': np.mean([m.accuracy for m in fold_metrics]),
                'accuracy_std': np.std([m.accuracy for m in fold_metrics]),
                'balanced_accuracy_mean': np.mean([m.balanced_accuracy for m in fold_metrics]),
                'balanced_accuracy_std': np.std([m.balanced_accuracy for m in fold_metrics]),
                'macro_f1_mean': np.mean([m.macro_f1 for m in fold_metrics]),
                'macro_f1_std': np.std([m.macro_f1 for m in fold_metrics]),
                'crisis_recall_mean': np.mean([m.crisis_recall for m in fold_metrics]),
                'crisis_recall_std': np.std([m.crisis_recall for m in fold_metrics]),
                'cohen_kappa_mean': np.mean([m.cohen_kappa for m in fold_metrics]),
                'cohen_kappa_std': np.std([m.cohen_kappa for m in fold_metrics]),
                'fold_metrics': [m.to_dict() for m in fold_metrics]
            }
            
            logger.info(f"  {name}: Acc={results[name]['accuracy_mean']:.2%}±{results[name]['accuracy_std']:.2%}, "
                       f"Crisis Recall={results[name]['crisis_recall_mean']:.2%}±{results[name]['crisis_recall_std']:.2%}, "
                       f"Macro-F1={results[name]['macro_f1_mean']:.3f}±{results[name]['macro_f1_std']:.3f}")
        
        return results


class ResearchTrainer:
    def __init__(self, model: nn.Module, config: ResearchConfig):
        self.model = model.to(DEVICE)
        self.config = config
        self.metrics_history = defaultdict(list)
        
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
        
        tscv = PurgedTimeSeriesSplit(
            n_splits=self.config.n_splits, 
            purge_gap=self.config.purge_gap, 
            min_train_size=50
        )
        
        fold_results = []
        all_preds = []
        all_probs = []
        all_labels = []
        
        best_overall_model_state = None
        best_overall_crisis_recall = 0
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(len(sequences))):
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
            
            logger.info(f"Train regimes: Bull={train_regime_dist[0]}, Normal={train_regime_dist[1]}, Crisis={train_regime_dist[2]}")
            logger.info(f"Val regimes: Bull={val_regime_dist[0]}, Normal={val_regime_dist[1]}, Crisis={val_regime_dist[2]}")
            
            val_unique = np.unique(val_labels)
            if len(val_unique) < 2:
                logger.warning(f"Fold {fold+1} has only {len(val_unique)} class(es) in validation, skipping...")
                continue
            
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
            all_labels.extend(val_labels)
        
        if not fold_results:
            return {}
        
        if best_overall_model_state:
            torch.save(best_overall_model_state, 'output/models/best_crisis_model.pt')
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        overall_metrics = MetricsCalculator.calculate_all_metrics(
            all_labels, all_preds, all_probs, self.config.num_regimes
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
            'all_labels': all_labels
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("Cross-Validation Results")
        logger.info(f"{'='*60}")
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
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7, verbose=False
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
                
                loss = loss_regime + self.config.crisis_loss_weight * loss_crisis + 0.1 * loss_vol + loss_edge_reg
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                if output['regime_logits'].argmax().item() == label:
                    train_correct += 1
            
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
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
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
                logger.info(
                    f"Fold {fold+1}, Epoch {epoch+1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}, "
                    f"Crisis Recall: {crisis_recall:.2%}"
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
    def _plot_model_comparison(ax, gnn_metrics: QuantMetrics, cv_results: Dict, baseline_results: Dict[str, Dict]):
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
    def _plot_per_class_metrics(ax, metrics: QuantMetrics):
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
        model: nn.Module,
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
        metrics: QuantMetrics,
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


def main():
    logger.info("=" * 80)
    logger.info("Cross-Market Contagion Networks - Extended Research Implementation")
    logger.info("NOVEL: Regime-Adaptive Learned Edge Construction (RALEC)")
    logger.info("=" * 80)
    
    if not API_KEY:
        logger.error("API key not found! Please set EODHD_API_KEY in .env file")
        return
    
    config = ResearchConfig()
    config.save('output/config.json')
    logger.info(f"\nConfiguration saved to output/config.json")
    logger.info(f"Regime labeling method: {config.regime_method}")
    logger.info(f"Using learned edges: {config.use_learned_edges}")
    logger.info(f"Model hidden_dim: {config.hidden_dim}, num_layers: {config.num_layers}")
    logger.info(f"Dropout: {config.dropout}, Crisis weight: {config.crisis_weight_multiplier}x")
    logger.info(f"Run baselines: {config.run_baselines}")
    
    client = EODHDClient(API_KEY)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=config.lookback_days)).strftime('%Y-%m-%d')
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 1: Data Collection ({len(ALL_SYMBOLS)} symbols)")
    logger.info(f"{'='*80}")
    logger.info(f"Fetching data from {start_date} to {end_date}...")
    
    data = {}
    failed_symbols = []
    
    for i, sym in enumerate(ALL_SYMBOLS, 1):
        df = client.get_eod_data(sym, 'US', start_date, end_date)
        if not df.empty and 'adjusted_close' in df.columns and len(df) >= config.min_observations:
            data[f"{sym}.US"] = df
            if i % 10 == 0 or i == len(ALL_SYMBOLS):
                logger.info(f"  Progress: {i}/{len(ALL_SYMBOLS)} - Loaded {len(data)} symbols")
        else:
            failed_symbols.append(sym)
    
    logger.info(f"\nSuccessfully loaded {len(data)} assets")
    if failed_symbols:
        logger.info(f"Failed to load: {len(failed_symbols)} symbols")
    
    if len(data) < 10:
        logger.error(f"Insufficient assets: {len(data)} < 10")
        return
    
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
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 2: Statistical Lead-Lag Detection")
    logger.info(f"{'='*80}")
    
    lead_lag_detector = StatisticalLeadLagDetector(config)
    lead_lag_df = lead_lag_detector.detect_relationships(returns_dict, method='correlation')
    
    if not lead_lag_df.empty:
        logger.info(f"\nTOP MARKET LEADERS:")
        if 'corr_leader' in lead_lag_df.columns:
            leaders = lead_lag_df['corr_leader'].value_counts().head(10)
            for leader, count in leaders.items():
                logger.info(f"   {leader}: influences {count} assets")
        
        PublicationVisualizations.plot_lead_lag_network(lead_lag_df)
    
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
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 4: Temporal Graph Construction")
    logger.info(f"{'='*80}")
    
    graph_builder = MultiMethodGraphBuilder(config)
    graphs, symbols_list = graph_builder.build_temporal_graphs(
        returns_dict,
        use_full_graph=config.use_learned_edges
    )
    
    if not graphs:
        logger.error("Failed to build graphs")
        return
    
    avg_nodes = np.mean([g.num_nodes for g in graphs])
    avg_edges = np.mean([g.edge_index.shape[1] for g in graphs])
    logger.info(f"\nGraph statistics:")
    logger.info(f"   Number of graphs: {len(graphs)}")
    logger.info(f"   Average nodes: {avg_nodes:.1f}")
    logger.info(f"   Average edges: {avg_edges:.1f}")
    
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
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 5b: Training RALEC-GNN Model")
    logger.info(f"{'='*80}")
    
    model = TemporalGNNWithLearnedEdges(config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    trainer = ResearchTrainer(model, config)
    cv_results = trainer.train_with_cross_validation(graphs, regime_df)
    
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
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 7: Current Market State Analysis")
    logger.info(f"{'='*80}")
    
    if len(graphs) >= config.seq_len and cv_results:
        try:
            model.load_state_dict(torch.load('output/models/best_crisis_model.pt'))
            logger.info("Loaded best crisis detection model")
        except:
            best_fold = len(cv_results.get('fold_results', [])) - 1
            if best_fold >= 0:
                try:
                    model.load_state_dict(torch.load(f'output/models/best_model_fold{best_fold}.pt'))
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