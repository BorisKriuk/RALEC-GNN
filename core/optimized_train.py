#!/usr/bin/env python3
"""
Optimized RALEC-GNN Training Pipeline
Phase 1: Performance Optimization
- Mixed precision training (AMP)
- Parallel cross-validation
- Multi-scale temporal features
- Efficient data loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import multiprocessing as mp
from joblib import Parallel, delayed
import time
from datetime import datetime
import logging
import os

# Import necessary components from main.py
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import (
    TemporalGNNWithLearnedEdges, FocalLoss, TimeSeriesCrossValidator,
    create_temporal_graphs, SYMBOL_CATEGORIES, DEVICE, SEED, ResearchConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizedTrainingConfig:
    """Enhanced training configuration with optimization parameters"""
    # Original parameters
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 100
    early_stopping_patience: int = 15
    
    # Crisis detection parameters
    crisis_weight_multiplier: float = 5.0
    crisis_loss_weight: float = 0.5
    focal_loss_gamma: float = 2.0
    label_smoothing: float = 0.1
    
    # Optimization parameters
    use_amp: bool = True  # Automatic Mixed Precision
    gradient_accumulation_steps: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Multi-scale parameters
    multi_scale_windows: List[int] = None
    
    # Parallel CV
    parallel_cv: bool = True
    cv_n_jobs: int = -1  # Use all cores
    
    def __post_init__(self):
        if self.multi_scale_windows is None:
            self.multi_scale_windows = [5, 10, 20, 60]  # Days

class GraphSequenceDataset(Dataset):
    """Efficient dataset for graph sequences"""
    def __init__(self, sequences: List[List], labels: List[int], volatilities: List[float]):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.volatilities = torch.tensor(volatilities, dtype=torch.float32)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.volatilities[idx]

class MultiScaleFeatureExtractor:
    """Extract features at multiple temporal scales"""
    
    @staticmethod
    def compute_rolling_features(data: pd.DataFrame, window_sizes: List[int]) -> pd.DataFrame:
        """Compute rolling statistics at multiple scales"""
        features = data.copy()
        
        for window in window_sizes:
            # Returns
            features[f'return_{window}d'] = features['return'].rolling(window).mean()
            features[f'return_std_{window}d'] = features['return'].rolling(window).std()
            
            # Volume
            features[f'volume_mean_{window}d'] = features['volume_ratio'].rolling(window).mean()
            features[f'volume_std_{window}d'] = features['volume_ratio'].rolling(window).std()
            
            # Volatility
            features[f'volatility_{window}d'] = features['return'].rolling(window).std() * np.sqrt(252)
            
            # Price levels
            features[f'high_{window}d'] = features['high_low_ratio'].rolling(window).max()
            features[f'low_{window}d'] = features['high_low_ratio'].rolling(window).min()
            
        return features.fillna(method='ffill').fillna(0)
    
    @staticmethod
    def add_temporal_embeddings(data: pd.DataFrame) -> pd.DataFrame:
        """Add temporal position encodings"""
        if 'Date' in data.columns:
            data['day_of_week'] = pd.to_datetime(data['Date']).dt.dayofweek / 6.0
            data['day_of_month'] = pd.to_datetime(data['Date']).dt.day / 31.0
            data['month'] = pd.to_datetime(data['Date']).dt.month / 12.0
            data['quarter'] = pd.to_datetime(data['Date']).dt.quarter / 4.0
        
        return data

class OptimizedRALECGNN(nn.Module):
    """RALEC-GNN with optimization-friendly modifications"""
    
    def __init__(self, base_model: TemporalGNNWithLearnedEdges):
        super().__init__()
        self.base_model = base_model
        
        # Add gradient checkpointing support
        self.use_checkpoint = False
        
    def forward(self, graph_sequence, return_analysis=False):
        if self.use_checkpoint and self.training:
            # Use gradient checkpointing to save memory
            return torch.utils.checkpoint.checkpoint(
                self.base_model, graph_sequence, return_analysis
            )
        else:
            return self.base_model(graph_sequence, return_analysis)

class OptimizedTrainer:
    """Optimized trainer with AMP and parallel cross-validation"""
    
    def __init__(self, model: nn.Module, config: OptimizedTrainingConfig):
        self.model = model
        self.config = config
        self.scaler = GradScaler() if config.use_amp else None
        
    def train_epoch_amp(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion_regime: nn.Module,
        criterion_vol: nn.Module,
        epoch: int
    ) -> Tuple[float, float]:
        """Train one epoch with mixed precision"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        accumulation_steps = self.config.gradient_accumulation_steps
        
        for batch_idx, (sequences, labels, volatilities) in enumerate(dataloader):
            # Move to device
            labels = labels.to(DEVICE)
            volatilities = volatilities.to(DEVICE)
            
            # Mixed precision forward pass
            with autocast(enabled=self.config.use_amp):
                output = self.model(sequences)
                
                loss_regime = criterion_regime(output['regime_logits'], labels)
                loss_crisis = self._compute_crisis_loss(output['regime_logits'], labels)
                loss_vol = criterion_vol(
                    output['volatility_forecast'].squeeze(), 
                    volatilities
                )
                loss_edge_reg = self._compute_edge_regularization()
                
                # Total loss with gradient accumulation normalization
                loss = (loss_regime + 
                       self.config.crisis_loss_weight * loss_crisis + 
                       0.1 * loss_vol + 
                       loss_edge_reg) / accumulation_steps
            
            # Backward pass with gradient scaling
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item() * accumulation_steps
            predictions = output['regime_logits'].argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.debug(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                           f"Loss: {loss.item() * accumulation_steps:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(
        self,
        dataloader: DataLoader,
        criterion_regime: nn.Module,
        criterion_vol: nn.Module
    ) -> Tuple[float, float, float, List[int], List[np.ndarray]]:
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels, volatilities in dataloader:
                labels = labels.to(DEVICE)
                volatilities = volatilities.to(DEVICE)
                
                with autocast(enabled=self.config.use_amp):
                    output = self.model(sequences)
                    
                    loss_regime = criterion_regime(output['regime_logits'], labels)
                    loss_vol = criterion_vol(
                        output['volatility_forecast'].squeeze(),
                        volatilities
                    )
                    loss = loss_regime + 0.1 * loss_vol
                
                total_loss += loss.item()
                
                preds = output['regime_logits'].argmax(dim=1)
                probs = output['regime_probs'].cpu().numpy()
                
                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().tolist())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        
        # Crisis recall
        crisis_mask = np.array(all_labels) == 2
        crisis_recall = 0
        if crisis_mask.sum() > 0:
            crisis_recall = (np.array(all_preds)[crisis_mask] == 2).mean()
        
        return avg_loss, accuracy, crisis_recall, all_preds, all_probs
    
    def _compute_crisis_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute additional loss for crisis detection"""
        crisis_mask = (targets == 2).float()
        if crisis_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        crisis_probs = F.softmax(logits, dim=1)[:, 2]
        crisis_loss = F.binary_cross_entropy(crisis_probs, crisis_mask)
        
        return crisis_loss
    
    def _compute_edge_regularization(self) -> torch.Tensor:
        """Compute edge sparsity and entropy regularization"""
        reg_loss = torch.tensor(0.0, device=DEVICE)
        
        # Find edge constructor module
        for module in self.model.modules():
            if hasattr(module, 'sparsity_regularization'):
                reg_loss += 0.001 * module.sparsity_regularization
            if hasattr(module, 'entropy_regularization'):
                reg_loss += 0.001 * module.entropy_regularization
        
        return reg_loss
    
    def train_fold(self, train_data, val_data, fold: int) -> Dict[str, Any]:
        """Train a single fold with optimization"""
        train_dataset = GraphSequenceDataset(*train_data)
        val_dataset = GraphSequenceDataset(*val_data)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,  # Adjust based on GPU memory
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,  # Larger batch for validation
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7, verbose=False
        )
        
        # Loss functions
        class_weights = torch.tensor(
            [1.0, 1.5, self.config.crisis_weight_multiplier],
            dtype=torch.float,
            device=DEVICE
        )
        
        criterion_regime = FocalLoss(
            alpha=class_weights,
            gamma=self.config.focal_loss_gamma,
            label_smoothing=self.config.label_smoothing
        )
        criterion_vol = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        best_metrics = {}
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss, train_acc = self.train_epoch_amp(
                train_loader, optimizer, criterion_regime, criterion_vol, epoch
            )
            
            # Validate
            val_loss, val_acc, crisis_recall, val_preds, val_probs = self.validate_epoch(
                val_loader, criterion_regime, criterion_vol
            )
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = {
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'crisis_recall': crisis_recall,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'predictions': val_preds,
                    'probabilities': val_probs
                }
                patience_counter = 0
                
                # Save model checkpoint
                if hasattr(self.config, 'save_models') and self.config.save_models:
                    torch.save(
                        self.model.state_dict(),
                        f'output/models/optimized_model_fold{fold}.pt'
                    )
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 10 == 0 or patience_counter >= self.config.early_stopping_patience:
                elapsed = time.time() - start_time
                logger.info(
                    f"Fold {fold}, Epoch {epoch}/{self.config.epochs} "
                    f"({elapsed:.1f}s) - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}, "
                    f"Crisis Recall: {crisis_recall:.2%}"
                )
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return best_metrics

def parallel_cv_training(
    model_class,
    graph_sequences: List[List],
    labels: np.ndarray,
    volatilities: np.ndarray,
    config: OptimizedTrainingConfig,
    n_splits: int = 5
) -> Dict[str, Any]:
    """Parallel cross-validation training"""
    
    cv = TimeSeriesCrossValidator(n_splits=n_splits, purge_gap=5)
    
    def train_single_fold(fold_idx, train_idx, val_idx):
        """Train a single fold - to be run in parallel"""
        logger.info(f"Starting fold {fold_idx + 1}/{n_splits}")
        
        # Create model instance for this fold
        model = model_class.to(DEVICE)
        trainer = OptimizedTrainer(model, config)
        
        # Prepare data
        train_data = (
            [graph_sequences[i] for i in train_idx],
            labels[train_idx],
            volatilities[train_idx]
        )
        val_data = (
            [graph_sequences[i] for i in val_idx],
            labels[val_idx],
            volatilities[val_idx]
        )
        
        # Train fold
        fold_results = trainer.train_fold(train_data, val_data, fold_idx)
        fold_results['fold'] = fold_idx
        
        return fold_results
    
    # Run folds in parallel or sequential based on config
    if config.parallel_cv and config.cv_n_jobs != 1:
        n_jobs = config.cv_n_jobs if config.cv_n_jobs > 0 else mp.cpu_count()
        logger.info(f"Running {n_splits}-fold CV in parallel with {n_jobs} jobs")
        
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(train_single_fold)(i, train_idx, val_idx)
            for i, (train_idx, val_idx) in enumerate(cv.split(labels))
        )
    else:
        logger.info(f"Running {n_splits}-fold CV sequentially")
        results = []
        for i, (train_idx, val_idx) in enumerate(cv.split(labels)):
            results.append(train_single_fold(i, train_idx, val_idx))
    
    # Aggregate results
    avg_metrics = {
        'val_acc': np.mean([r['val_acc'] for r in results]),
        'val_acc_std': np.std([r['val_acc'] for r in results]),
        'crisis_recall': np.mean([r['crisis_recall'] for r in results]),
        'crisis_recall_std': np.std([r['crisis_recall'] for r in results]),
        'fold_results': results
    }
    
    logger.info(
        f"CV Complete - Avg Val Acc: {avg_metrics['val_acc']:.2%} "
        f"(±{avg_metrics['val_acc_std']:.2%}), "
        f"Avg Crisis Recall: {avg_metrics['crisis_recall']:.2%} "
        f"(±{avg_metrics['crisis_recall_std']:.2%})"
    )
    
    return avg_metrics

def optimize_data_pipeline(
    data: pd.DataFrame,
    symbols: List[str],
    config: OptimizedTrainingConfig
) -> pd.DataFrame:
    """Add multi-scale features to data"""
    logger.info("Adding multi-scale temporal features...")
    
    # Group by symbol and add multi-scale features
    enhanced_data = []
    
    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol].copy()
        
        # Add multi-scale features
        symbol_data = MultiScaleFeatureExtractor.compute_rolling_features(
            symbol_data, 
            config.multi_scale_windows
        )
        
        # Add temporal embeddings
        symbol_data = MultiScaleFeatureExtractor.add_temporal_embeddings(symbol_data)
        
        enhanced_data.append(symbol_data)
    
    enhanced_df = pd.concat(enhanced_data, ignore_index=True)
    
    # Log new features
    new_features = [col for col in enhanced_df.columns if col not in data.columns]
    logger.info(f"Added {len(new_features)} multi-scale features: {new_features[:5]}...")
    
    return enhanced_df

def run_optimized_training(
    graph_sequences: List[List],
    labels: np.ndarray,
    volatilities: np.ndarray,
    num_features: int,
    num_edge_features: int
) -> Dict[str, Any]:
    """Main function to run optimized training"""
    
    config = OptimizedTrainingConfig()
    
    logger.info("=" * 80)
    logger.info("PHASE 1: OPTIMIZED RALEC-GNN TRAINING")
    logger.info("=" * 80)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Mixed Precision: {config.use_amp}")
    logger.info(f"Parallel CV: {config.parallel_cv}")
    logger.info(f"Gradient Accumulation Steps: {config.gradient_accumulation_steps}")
    logger.info("=" * 80)
    
    # Create base model with ResearchConfig
    research_config = ResearchConfig(
        node_features=num_features,
        edge_features=num_edge_features,
        num_regimes=3
    )
    base_model = TemporalGNNWithLearnedEdges(research_config)
    
    # Wrap with optimizations
    model = OptimizedRALECGNN(base_model)
    
    # Run parallel cross-validation
    start_time = time.time()
    
    results = parallel_cv_training(
        model,
        graph_sequences,
        labels,
        volatilities,
        config
    )
    
    total_time = time.time() - start_time
    
    logger.info(f"\nTotal training time: {total_time/60:.1f} minutes")
    logger.info(f"Average time per fold: {total_time/5/60:.1f} minutes")
    
    # Compare with baseline time
    baseline_time = 2.5 * 60 * 60  # 2.5 hours in seconds
    speedup = baseline_time / total_time
    logger.info(f"Speedup vs baseline: {speedup:.1f}x")
    
    return results

if __name__ == "__main__":
    # This would be called from main.py with the prepared data
    logger.info("Optimized training module loaded. Import and use run_optimized_training()")