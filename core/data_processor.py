"""Data processing utilities for RALEC-GNN."""

import torch
import numpy as np
from typing import List, Tuple, Any, Optional
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


class FinancialDataProcessor:
    """Process financial data for RALEC-GNN."""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.fitted = False
        
    def process(self, raw_data: Any) -> List[Data]:
        """
        Process raw financial data into graph sequence.
        
        Args:
            raw_data: Raw financial data (prices, volumes, etc.)
            
        Returns:
            List of Data objects representing graph snapshots
        """
        # This is a simplified version
        # In practice, would handle real financial data formats
        
        if isinstance(raw_data, list) and all(isinstance(g, Data) for g in raw_data):
            # Already processed
            return raw_data
            
        # For demo, create synthetic graph sequence
        return self._create_synthetic_graphs(
            num_graphs=20,
            num_assets=self.config.num_assets,
            num_features=self.config.num_features
        )
    
    def _create_synthetic_graphs(
        self,
        num_graphs: int,
        num_assets: int,
        num_features: int
    ) -> List[Data]:
        """Create synthetic graph sequence for testing."""
        graphs = []
        
        for t in range(num_graphs):
            # Node features (assets)
            x = torch.randn(num_assets, num_features)
            
            # Edge connectivity (correlation-based)
            corr_matrix = torch.randn(num_assets, num_assets)
            corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Symmetric
            
            # Threshold to create edges
            threshold = 0.3
            edge_list = []
            edge_weights = []
            
            for i in range(num_assets):
                for j in range(i + 1, num_assets):
                    if abs(corr_matrix[i, j]) > threshold:
                        edge_list.append([i, j])
                        edge_list.append([j, i])
                        edge_weights.append(abs(corr_matrix[i, j]))
                        edge_weights.append(abs(corr_matrix[i, j]))
                        
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 1), dtype=torch.float)
            
            # Create graph
            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr
            )
            
            # Add label for supervised learning
            if t < num_graphs - 1:
                # Simplified: predict next regime
                graph.y = torch.tensor([0])  # Normal regime
            else:
                graph.y = torch.tensor([2])  # Bear regime
                
            graphs.append(graph)
            
        return graphs
    
    def extract_returns(self, price_data: np.ndarray) -> np.ndarray:
        """Extract returns from price data."""
        if len(price_data) < 2:
            return np.zeros_like(price_data)
            
        returns = np.diff(price_data, axis=0) / price_data[:-1]
        return np.nan_to_num(returns, 0)
    
    def create_multi_scale_features(
        self,
        data: np.ndarray,
        windows: Optional[List[int]] = None
    ) -> np.ndarray:
        """Create multi-scale features as per Phase 1."""
        if windows is None:
            windows = self.config.multi_scale_windows
            
        features = []
        
        for window in windows:
            # Rolling statistics
            if len(data) >= window:
                rolling_mean = np.convolve(data, np.ones(window)/window, mode='valid')
                rolling_std = np.array([
                    np.std(data[i:i+window]) for i in range(len(data)-window+1)
                ])
                features.append(rolling_mean)
                features.append(rolling_std)
                
        if features:
            # Pad to same length
            max_len = max(len(f) for f in features)
            padded_features = []
            for f in features:
                pad_len = max_len - len(f)
                if pad_len > 0:
                    f = np.pad(f, (pad_len, 0), mode='edge')
                padded_features.append(f)
                
            return np.column_stack(padded_features)
        
        return data.reshape(-1, 1)