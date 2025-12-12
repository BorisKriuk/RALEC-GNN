"""Configuration management for RALEC-GNN."""

from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class RALECConfig:
    """Configuration for RALEC-GNN system."""
    
    # Model architecture
    num_features: int = 16
    num_assets: int = 77
    hidden_dim: int = 256
    num_regimes: int = 5
    dropout: float = 0.2
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    
    # Optimization settings (Phase 1)
    use_amp: bool = True
    gradient_accumulation_steps: int = 4
    parallel_cv: bool = True
    multi_scale_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    
    # Theory settings (Phase 2)
    volatility_threshold: float = 0.3
    correlation_threshold: float = 0.6
    liquidity_threshold: float = 0.4
    
    # Causal settings (Phase 3)
    max_lag: int = 5
    causality_threshold: float = 0.3
    use_pcmci: bool = True
    
    # Phase detection settings (Phase 4)
    early_warning_window: int = 20
    critical_threshold: float = 0.8
    
    # Meta-learning settings (Phase 5)
    memory_size: int = 1000
    num_prototypes: int = 10
    adaptation_steps: int = 5
    
    # Risk metrics settings (Phase 6)
    risk_threshold: float = 0.7
    cascade_threshold: float = 0.6
    defensive_activation: float = 0.7
    
    # System settings
    use_cuda: bool = True
    seed: int = 42
    log_interval: int = 10
    save_interval: int = 20
    
    # Optimizer settings
    optimizer: str = 'adamw'
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'
    min_lr: float = 1e-6
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'RALECConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def __repr__(self):
        """String representation."""
        return f"RALECConfig(assets={self.num_assets}, hidden={self.hidden_dim}, epochs={self.epochs})"