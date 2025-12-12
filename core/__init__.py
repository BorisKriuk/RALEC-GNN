"""
RALEC-GNN Core Implementation
Consolidated into 3 main modules:
- model.py: Complete RALEC-GNN model with all 6 phases
- data.py: Data processing and dataset utilities  
- training.py: Training and evaluation utilities
"""

from .model import EnhancedRALECGNN
from .data import FinancialDataProcessor, DataConfig, GraphSequenceDataset
from .training import TrainingConfig, Trainer, CrossValidator

__all__ = [
    'EnhancedRALECGNN',
    'FinancialDataProcessor', 
    'DataConfig',
    'GraphSequenceDataset',
    'TrainingConfig',
    'Trainer',
    'CrossValidator'
]