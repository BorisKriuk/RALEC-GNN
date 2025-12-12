"""
Data processing and dataset utilities for RALEC-GNN
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data processing"""
    symbols: List[str]
    lookback_days: int = 1260  # 5 years
    window_size: int = 60
    step_size: int = 5
    sequence_length: int = 15
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    multi_scale_windows: List[int] = None
    
    def __post_init__(self):
        if self.multi_scale_windows is None:
            self.multi_scale_windows = [5, 10, 20, 60]  # Days


class GraphSequenceDataset(Dataset):
    """Efficient dataset for graph sequences"""
    
    def __init__(self, sequences: List[List[Data]], labels: List[int], 
                 volatilities: List[float], returns: Optional[List[float]] = None):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.volatilities = torch.tensor(volatilities, dtype=torch.float32)
        self.returns = torch.tensor(returns, dtype=torch.float32) if returns else None
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.returns is not None:
            return (self.sequences[idx], self.labels[idx], 
                   self.volatilities[idx], self.returns[idx])
        return self.sequences[idx], self.labels[idx], self.volatilities[idx]


class FinancialDataProcessor:
    """Process financial data for RALEC-GNN"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.fitted = False
        
    def load_and_prepare_data(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and prepare financial data.
        
        Args:
            data_path: Path to data file or None for synthetic data
            
        Returns:
            Dictionary containing processed data
        """
        if data_path:
            # Load real data
            raw_data = pd.read_csv(data_path)
            logger.info(f"Loaded data from {data_path}")
        else:
            # Generate synthetic data for demo
            raw_data = self._generate_synthetic_data()
            logger.info("Generated synthetic data for demo")
            
        # Process data
        processed_data = self._process_raw_data(raw_data)
        
        return processed_data
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic financial data for testing"""
        dates = pd.date_range(
            end=datetime.now(), 
            periods=self.config.lookback_days, 
            freq='D'
        )
        
        data_list = []
        
        for symbol in self.config.symbols:
            # Generate correlated price series
            np.random.seed(hash(symbol) % 100)
            
            # Base price movement
            returns = np.random.normal(0.0005, 0.02, len(dates))
            
            # Add regime-specific patterns
            crisis_periods = [(800, 850), (1100, 1150)]  # Crisis windows
            for start, end in crisis_periods:
                returns[start:end] *= 2.5  # Higher volatility
                returns[start:end] -= 0.01  # Negative bias
            
            # Create price series
            price = 100 * np.exp(np.cumsum(returns))
            
            # Volume (correlated with volatility)
            volume = 1e6 * (1 + np.abs(returns) * 50)
            
            # Create DataFrame
            symbol_data = pd.DataFrame({
                'Date': dates,
                'symbol': symbol,
                'Close': price,
                'Open': price * (1 + np.random.normal(0, 0.001, len(dates))),
                'High': price * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
                'Low': price * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                'Volume': volume.astype(int),
                'return': returns
            })
            
            data_list.append(symbol_data)
        
        return pd.concat(data_list, ignore_index=True)
    
    def _process_raw_data(self, raw_data: pd.DataFrame) -> Dict[str, Any]:
        """Process raw data into features"""
        # Group by symbol
        individual_data = {}
        for symbol in self.config.symbols:
            symbol_data = raw_data[raw_data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date').reset_index(drop=True)
            
            # Add technical features
            symbol_data = self._add_technical_features(symbol_data)
            
            # Add multi-scale features
            symbol_data = self._add_multi_scale_features(symbol_data)
            
            individual_data[symbol] = symbol_data
        
        # Combine all data
        combined_data = pd.concat(list(individual_data.values()), ignore_index=True)
        combined_data = combined_data.sort_values(['Date', 'symbol']).reset_index(drop=True)
        
        # Extract feature columns
        feature_columns = [col for col in combined_data.columns 
                          if col not in ['Date', 'symbol', 'Close', 'Open', 
                                       'High', 'Low', 'Volume', 'regime']]
        
        # Create edge index (fully connected for now)
        num_assets = len(self.config.symbols)
        edge_index = self._create_edge_index(num_assets)
        
        return {
            'individual_data': individual_data,
            'combined_data': combined_data,
            'symbols': self.config.symbols,
            'feature_columns': feature_columns,
            'edge_index': edge_index,
            'num_features': len(feature_columns),
            'num_assets': num_assets
        }
    
    def _add_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        # Returns
        if 'return' not in data.columns:
            data['return'] = data['Close'].pct_change().fillna(0)
        
        # Volatility
        data['volatility'] = data['return'].rolling(20).std().fillna(0)
        data['volatility_5d'] = data['return'].rolling(5).std().fillna(0)
        
        # Volume ratio
        data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume_ratio'].fillna(1)
        
        # Price ratios
        data['high_low_ratio'] = (data['High'] - data['Low']) / data['Close']
        data['close_open_ratio'] = (data['Close'] - data['Open']) / data['Open']
        
        # Moving averages
        for window in [5, 10, 20]:
            data[f'sma_{window}'] = data['Close'].rolling(window).mean()
            data[f'sma_ratio_{window}'] = data['Close'] / data[f'sma_{window}']
        
        # RSI
        data['rsi'] = self._calculate_rsi(data['Close'])
        
        # Fill NaN values
        data = data.fillna(method='ffill').fillna(0)
        
        return data
    
    def _add_multi_scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add multi-scale temporal features"""
        for window in self.config.multi_scale_windows:
            # Returns
            data[f'return_{window}d'] = data['return'].rolling(window).mean().fillna(0)
            data[f'return_std_{window}d'] = data['return'].rolling(window).std().fillna(0)
            
            # Volume
            data[f'volume_mean_{window}d'] = data['volume_ratio'].rolling(window).mean().fillna(1)
            data[f'volume_std_{window}d'] = data['volume_ratio'].rolling(window).std().fillna(0)
            
            # Volatility
            data[f'volatility_{window}d'] = data['return'].rolling(window).std() * np.sqrt(252)
            data[f'volatility_{window}d'] = data[f'volatility_{window}d'].fillna(0)
            
            # Price levels
            data[f'high_{window}d'] = data['high_low_ratio'].rolling(window).max().fillna(0)
            data[f'low_{window}d'] = data['high_low_ratio'].rolling(window).min().fillna(0)
        
        # Temporal embeddings
        if 'Date' in data.columns:
            data['day_of_week'] = pd.to_datetime(data['Date']).dt.dayofweek / 6.0
            data['day_of_month'] = pd.to_datetime(data['Date']).dt.day / 31.0
            data['month'] = pd.to_datetime(data['Date']).dt.month / 12.0
            data['quarter'] = pd.to_datetime(data['Date']).dt.quarter / 4.0
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Neutral RSI
    
    def _create_edge_index(self, num_nodes: int, connectivity: float = 1.0) -> torch.Tensor:
        """Create edge connectivity matrix"""
        if connectivity >= 1.0:
            # Fully connected
            edges = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edges.append([i, j])
        else:
            # Partial connectivity
            edges = []
            num_edges = int(num_nodes * (num_nodes - 1) * connectivity)
            selected_edges = np.random.choice(
                num_nodes * (num_nodes - 1), 
                num_edges, 
                replace=False
            )
            
            for idx in selected_edges:
                i = idx // (num_nodes - 1)
                j = idx % (num_nodes - 1)
                if j >= i:
                    j += 1
                edges.append([i, j])
        
        return torch.tensor(edges, dtype=torch.long).t()
    
    def label_market_regimes(self, data: pd.DataFrame, 
                           method: str = 'quantile') -> pd.DataFrame:
        """
        Label market regimes based on returns and volatility.
        
        Regimes:
        0: Bull/Low volatility
        1: Normal
        2: Bear/Crisis
        """
        # Calculate rolling metrics
        data['rolling_return'] = data.groupby('symbol')['return'].transform(
            lambda x: x.rolling(20).mean()
        )
        data['rolling_volatility'] = data.groupby('symbol')['return'].transform(
            lambda x: x.rolling(20).std()
        )
        
        if method == 'quantile':
            # Quantile-based labeling
            ret_q33 = data['rolling_return'].quantile(0.33)
            ret_q67 = data['rolling_return'].quantile(0.67)
            vol_q67 = data['rolling_volatility'].quantile(0.67)
            
            conditions = [
                (data['rolling_return'] > ret_q67) & (data['rolling_volatility'] < vol_q67),  # Bull
                (data['rolling_return'] < ret_q33) | (data['rolling_volatility'] > vol_q67),  # Crisis
            ]
            choices = [0, 2]
            
            data['regime'] = np.select(conditions, choices, default=1)  # Normal
        
        elif method == 'threshold':
            # Fixed threshold method
            conditions = [
                (data['rolling_return'] > 0.001) & (data['rolling_volatility'] < 0.15),  # Bull
                (data['rolling_return'] < -0.001) | (data['rolling_volatility'] > 0.25),  # Crisis
            ]
            choices = [0, 2]
            
            data['regime'] = np.select(conditions, choices, default=1)
        
        # Log regime distribution
        regime_counts = data['regime'].value_counts().sort_index()
        logger.info(f"Regime distribution: {dict(regime_counts)}")
        
        return data
    
    def create_temporal_graphs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create temporal graph sequences from processed data.
        
        Returns:
            Dictionary containing graph sequences and labels
        """
        combined_data = data['combined_data']
        feature_columns = data['feature_columns']
        edge_index = data['edge_index']
        
        # Get unique dates
        dates = sorted(combined_data['Date'].unique())
        
        graph_sequences = []
        labels = []
        volatilities = []
        returns = []
        
        # Create sliding windows
        for i in range(0, len(dates) - self.config.window_size - self.config.sequence_length, 
                      self.config.step_size):
            sequence = []
            
            # Create sequence of graphs
            for j in range(self.config.sequence_length):
                date_idx = i + j * 4  # Sample every 4 days in window
                date = dates[date_idx]
                
                # Get data for this date
                date_data = combined_data[combined_data['Date'] == date]
                
                if len(date_data) == len(self.config.symbols):
                    # Extract features
                    features = date_data[feature_columns].values
                    x = torch.tensor(features, dtype=torch.float32)
                    
                    # Create graph
                    graph = Data(x=x, edge_index=edge_index)
                    sequence.append(graph)
            
            if len(sequence) == self.config.sequence_length:
                # Get label from end of window
                end_date = dates[i + self.config.window_size]
                end_data = combined_data[combined_data['Date'] == end_date]
                
                if len(end_data) > 0:
                    # Use majority regime as label
                    regime = int(end_data['regime'].mode()[0])
                    vol = end_data['volatility'].mean()
                    ret = end_data['return'].mean()
                    
                    graph_sequences.append(sequence)
                    labels.append(regime)
                    volatilities.append(vol)
                    returns.append(ret)
        
        logger.info(f"Created {len(graph_sequences)} graph sequences")
        
        # Create indices for train/val/test split
        n_samples = len(labels)
        indices = np.arange(n_samples)
        
        train_end = int(n_samples * self.config.train_split)
        val_end = int(n_samples * (self.config.train_split + self.config.val_split))
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        return {
            'graph_sequences': graph_sequences,
            'labels': np.array(labels),
            'volatilities': np.array(volatilities),
            'returns': np.array(returns),
            'num_features': len(feature_columns),
            'num_edge_features': 1,  # Edge weights only for now
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        }
    
    def create_data_loaders(self, graph_data: Dict[str, Any], 
                          batch_size: int = 32,
                          num_workers: int = 4) -> Dict[str, DataLoader]:
        """Create train/val/test data loaders"""
        # Extract data
        sequences = graph_data['graph_sequences']
        labels = graph_data['labels']
        volatilities = graph_data['volatilities']
        returns = graph_data.get('returns', None)
        
        # Create datasets
        train_dataset = GraphSequenceDataset(
            [sequences[i] for i in graph_data['train_indices']],
            labels[graph_data['train_indices']],
            volatilities[graph_data['train_indices']],
            returns[graph_data['train_indices']] if returns is not None else None
        )
        
        val_dataset = GraphSequenceDataset(
            [sequences[i] for i in graph_data['val_indices']],
            labels[graph_data['val_indices']],
            volatilities[graph_data['val_indices']],
            returns[graph_data['val_indices']] if returns is not None else None
        )
        
        test_dataset = GraphSequenceDataset(
            [sequences[i] for i in graph_data['test_indices']],
            labels[graph_data['test_indices']],
            volatilities[graph_data['test_indices']],
            returns[graph_data['test_indices']] if returns is not None else None
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }


def detect_lead_lag_relationships(individual_data: Dict[str, pd.DataFrame], 
                                symbols: List[str],
                                max_lag: int = 5) -> pd.DataFrame:
    """Detect lead-lag relationships between assets"""
    lead_lag_results = []
    
    for leader in symbols:
        for follower in symbols:
            if leader != follower:
                # Get return series
                leader_returns = individual_data[leader]['return'].values
                follower_returns = individual_data[follower]['return'].values
                
                # Compute cross-correlation
                correlations = []
                for lag in range(0, max_lag + 1):
                    if lag == 0:
                        corr = np.corrcoef(leader_returns, follower_returns)[0, 1]
                    else:
                        if len(leader_returns) > lag:
                            corr = np.corrcoef(
                                leader_returns[:-lag], 
                                follower_returns[lag:]
                            )[0, 1]
                        else:
                            corr = 0
                    
                    correlations.append(corr)
                
                # Find optimal lag
                optimal_lag = np.argmax(np.abs(correlations))
                max_corr = correlations[optimal_lag]
                
                # Statistical significance (simplified)
                n = len(leader_returns)
                t_stat = max_corr * np.sqrt(n - 2) / np.sqrt(1 - max_corr**2)
                p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2)) if n > 2 else 1.0
                
                lead_lag_results.append({
                    'leader': leader,
                    'follower': follower,
                    'optimal_lag': optimal_lag,
                    'correlation': max_corr,
                    'abs_correlation': abs(max_corr),
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
    
    lead_lag_df = pd.DataFrame(lead_lag_results)
    return lead_lag_df.sort_values('abs_correlation', ascending=False)


# Import scipy for stats if available
try:
    from scipy import stats
except ImportError:
    logger.warning("scipy not available, using simplified statistics")
    
    class stats:
        class t:
            @staticmethod
            def cdf(x, df):
                # Simplified t-distribution CDF
                return 0.5 + 0.5 * np.sign(x) * (1 - np.exp(-abs(x)))


if __name__ == "__main__":
    # Example usage
    config = DataConfig(
        symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        lookback_days=252,  # 1 year for demo
        window_size=60,
        sequence_length=10
    )
    
    processor = FinancialDataProcessor(config)
    
    # Load and process data
    data_dict = processor.load_and_prepare_data()
    
    # Label regimes
    data_dict['combined_data'] = processor.label_market_regimes(
        data_dict['combined_data']
    )
    
    # Create graph sequences
    graph_data = processor.create_temporal_graphs(data_dict)
    
    logger.info(f"Dataset statistics:")
    logger.info(f"  Total sequences: {len(graph_data['labels'])}")
    logger.info(f"  Features per node: {graph_data['num_features']}")
    logger.info(f"  Train samples: {len(graph_data['train_indices'])}")
    logger.info(f"  Val samples: {len(graph_data['val_indices'])}")
    logger.info(f"  Test samples: {len(graph_data['test_indices'])}")
    
    # Create data loaders
    loaders = processor.create_data_loaders(graph_data)
    logger.info(f"Created data loaders with {len(loaders['train'])} train batches")