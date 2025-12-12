"""Synthetic data generation for testing RALEC-GNN."""

import torch
import numpy as np
from typing import Tuple, List
from torch_geometric.data import Data


def generate_synthetic_data(
    num_samples: int,
    num_assets: int,
    sequence_length: int = 20,
    crisis_ratio: float = 0.1
) -> Tuple[List[List[Data]], List[List[Data]]]:
    """
    Generate synthetic financial graph data for testing.
    
    Args:
        num_samples: Number of sequences to generate
        num_assets: Number of assets (nodes)
        sequence_length: Length of each sequence
        crisis_ratio: Ratio of crisis samples
        
    Returns:
        Train and validation data
    """
    all_sequences = []
    
    num_crisis = int(num_samples * crisis_ratio)
    
    for i in range(num_samples):
        is_crisis = i < num_crisis
        sequence = generate_graph_sequence(
            num_assets=num_assets,
            sequence_length=sequence_length,
            is_crisis=is_crisis
        )
        all_sequences.append(sequence)
    
    # Shuffle
    np.random.shuffle(all_sequences)
    
    # Split 80/20
    split_idx = int(0.8 * len(all_sequences))
    train_data = all_sequences[:split_idx]
    val_data = all_sequences[split_idx:]
    
    return train_data, val_data


def generate_graph_sequence(
    num_assets: int,
    sequence_length: int,
    is_crisis: bool = False
) -> List[Data]:
    """Generate a single graph sequence."""
    graphs = []
    
    # Base correlation
    base_corr = 0.3 if not is_crisis else 0.6
    
    for t in range(sequence_length):
        # Node features
        if is_crisis and t > sequence_length // 2:
            # Crisis period - higher volatility
            volatility = 0.5 + 0.3 * (t - sequence_length // 2) / (sequence_length // 2)
            correlation = base_corr + 0.3 * (t - sequence_length // 2) / (sequence_length // 2)
        else:
            # Normal period
            volatility = 0.2
            correlation = base_corr
            
        # Generate features
        x = torch.randn(num_assets, 16) * volatility
        
        # Add correlation structure
        common_factor = torch.randn(1, 16) * np.sqrt(correlation)
        x = x * np.sqrt(1 - correlation) + common_factor
        
        # Generate edges based on correlation
        edge_list = []
        edge_weights = []
        
        # Create correlation matrix
        corr_matrix = torch.corrcoef(x)
        
        for i in range(num_assets):
            for j in range(i + 1, num_assets):
                if abs(corr_matrix[i, j]) > 0.3:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
                    weight = abs(corr_matrix[i, j].item())
                    edge_weights.append(weight)
                    edge_weights.append(weight)
        
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
        
        # Add label
        if t == sequence_length - 1:
            # Predict regime at end of sequence
            if is_crisis:
                graph.y = torch.tensor([3])  # Crisis regime
            else:
                graph.y = torch.tensor([0])  # Normal regime
                
        graphs.append(graph)
        
    return graphs


def generate_current_market_state(num_assets: int) -> List[Data]:
    """Generate current market state for prediction."""
    # Generate a short sequence representing current state
    sequence = generate_graph_sequence(
        num_assets=num_assets,
        sequence_length=20,
        is_crisis=False
    )
    
    # Add some randomness to make it interesting
    if np.random.random() > 0.7:
        # Inject some stress
        for graph in sequence[-5:]:
            graph.x *= 1.5  # Increase volatility
            
    return sequence