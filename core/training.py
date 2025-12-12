"""
Training utilities for RALEC-GNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import time
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import multiprocessing as mp
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic parameters
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 100
    early_stopping_patience: int = 15
    batch_size: int = 32
    
    # Optimization parameters
    use_amp: bool = True  # Automatic Mixed Precision
    gradient_accumulation_steps: int = 4
    gradient_clip_value: float = 1.0
    
    # Learning rate scheduling
    scheduler_type: str = 'plateau'  # 'plateau', 'cosine', 'linear'
    scheduler_patience: int = 7
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    
    # Loss weights
    regime_loss_weight: float = 1.0
    volatility_loss_weight: float = 0.1
    risk_loss_weight: float = 0.2
    regularization_weight: float = 0.001
    
    # Crisis detection
    crisis_weight_multiplier: float = 5.0
    focal_loss_gamma: float = 2.0
    label_smoothing: float = 0.1
    
    # Validation
    val_check_interval: int = 1  # Check validation every N epochs
    
    # Parallel training
    parallel_cv: bool = True
    cv_n_jobs: int = -1  # Use all cores
    n_splits: int = 5  # Cross-validation splits
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3  # Save top k models
    
    # Logging
    log_interval: int = 10  # Log every N batches
    tensorboard: bool = True
    wandb: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return self.__dict__.copy()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance, especially for crisis detection.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, 
                 gamma: float = 2.0, 
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Labels of shape (N,)
        """
        n_classes = inputs.shape[1]
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_one_hot = F.one_hot(targets, n_classes).float()
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                            self.label_smoothing / n_classes
            ce_loss = -torch.sum(targets_one_hot * F.log_softmax(inputs, dim=1), dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.view(-1, 1)).squeeze()
        
        # Apply focal term
        focal_term = (1 - p_t) ** self.gamma
        loss = focal_term * ce_loss
        
        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class RALECLoss(nn.Module):
    """Combined loss function for RALEC-GNN"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Component losses
        self.focal_loss = FocalLoss(
            gamma=config.focal_loss_gamma,
            label_smoothing=config.label_smoothing
        )
        self.mse_loss = nn.MSELoss()
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                model: nn.Module) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs
            targets: Target values
            model: Model instance for regularization
            
        Returns:
            Total loss and component losses dict
        """
        losses = {}
        
        # Regime classification loss
        if 'regime_logits' in outputs and 'regimes' in targets:
            # Compute class weights based on batch distribution
            unique_regimes, counts = torch.unique(targets['regimes'], return_counts=True)
            class_weights = torch.zeros(outputs['regime_logits'].shape[1], device=targets['regimes'].device)
            for regime, count in zip(unique_regimes, counts):
                class_weights[regime] = 1.0 / count
            class_weights = class_weights / class_weights.sum()
            
            # Apply crisis weight multiplier
            if len(class_weights) > 2:
                class_weights[2] *= self.config.crisis_weight_multiplier
            
            self.focal_loss.alpha = class_weights
            losses['regime'] = self.focal_loss(outputs['regime_logits'], targets['regimes'])
        
        # Volatility prediction loss
        if 'volatility_forecast' in outputs and 'volatility' in targets:
            losses['volatility'] = self.mse_loss(
                outputs['volatility_forecast'].squeeze(), 
                targets['volatility']
            )
        
        # Risk score loss
        if 'risk_score' in outputs and 'risk_target' in targets:
            losses['risk'] = F.binary_cross_entropy(
                outputs['risk_score'].squeeze(),
                targets['risk_target'].float()
            )
        
        # Regularization losses
        reg_loss = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                reg_loss += torch.sum(param ** 2)
        losses['regularization'] = reg_loss
        
        # Phase-specific losses
        if 'phase_analysis' in outputs:
            # Phase transition loss
            if 'transition_target' in targets:
                losses['transition'] = F.binary_cross_entropy(
                    torch.tensor(outputs['phase_analysis']['transition_probability']),
                    targets['transition_target'].float()
                )
        
        # Combine losses
        total_loss = (
            self.config.regime_loss_weight * losses.get('regime', 0) +
            self.config.volatility_loss_weight * losses.get('volatility', 0) +
            self.config.risk_loss_weight * losses.get('risk', 0) +
            self.config.regularization_weight * losses.get('regularization', 0) +
            0.1 * losses.get('transition', 0)  # Fixed weight for transition
        )
        
        return total_loss, losses


class Trainer:
    """Main trainer class for RALEC-GNN"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Setup optimization
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.scaler = GradScaler() if config.use_amp else None
        
        # Loss function
        self.criterion = RALECLoss(config)
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Setup logging
        self._setup_logging()
        
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler"""
        if self.config.scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.min_lr
            )
        elif self.config.scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr
            )
        else:  # linear
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_lr / self.config.learning_rate,
                total_iters=self.config.epochs
            )
    
    def _setup_logging(self):
        """Setup logging backends"""
        self.log_dir = Path(self.config.checkpoint_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir / "tensorboard")
        
        if self.config.wandb:
            import wandb
            wandb.init(
                project="ralec-gnn",
                config=self.config.to_dict()
            )
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0
        all_losses = defaultdict(float)
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Prepare batch
            if len(batch_data) == 4:
                sequences, regimes, volatility, returns = batch_data
                targets = {
                    'regimes': regimes.to(self.device),
                    'volatility': volatility.to(self.device),
                    'returns': returns.to(self.device)
                }
            else:
                sequences, regimes, volatility = batch_data
                targets = {
                    'regimes': regimes.to(self.device),
                    'volatility': volatility.to(self.device)
                }
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(sequences)
                loss, loss_components = self.criterion(outputs, targets, self.model)
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_value
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_value
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Track losses
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            for k, v in loss_components.items():
                all_losses[k] += v.item() if torch.is_tensor(v) else v
            
            # Track predictions
            if 'regime_logits' in outputs:
                predictions = outputs['regime_logits'].argmax(dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets['regimes'].cpu().numpy())
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                logger.debug(
                    f"Epoch {self.current_epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f}"
                )
        
        # Compute metrics
        avg_loss = total_loss / len(train_loader)
        avg_losses = {k: v / len(train_loader) for k, v in all_losses.items()}
        
        metrics = {}
        if all_predictions:
            metrics['accuracy'] = np.mean(np.array(all_predictions) == np.array(all_targets))
            
            # Crisis recall
            crisis_mask = np.array(all_targets) == 2
            if crisis_mask.sum() > 0:
                metrics['crisis_recall'] = np.mean(
                    np.array(all_predictions)[crisis_mask] == 2
                )
        
        return avg_loss, {**avg_losses, **metrics}
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        all_losses = defaultdict(float)
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Prepare batch
                if len(batch_data) == 4:
                    sequences, regimes, volatility, returns = batch_data
                    targets = {
                        'regimes': regimes.to(self.device),
                        'volatility': volatility.to(self.device),
                        'returns': returns.to(self.device)
                    }
                else:
                    sequences, regimes, volatility = batch_data
                    targets = {
                        'regimes': regimes.to(self.device),
                        'volatility': volatility.to(self.device)
                    }
                
                # Forward pass
                with autocast(enabled=self.config.use_amp):
                    outputs = self.model(sequences)
                    loss, loss_components = self.criterion(outputs, targets, self.model)
                
                # Track losses
                total_loss += loss.item()
                for k, v in loss_components.items():
                    all_losses[k] += v.item() if torch.is_tensor(v) else v
                
                # Track predictions
                if 'regime_logits' in outputs:
                    predictions = outputs['regime_logits'].argmax(dim=1)
                    probs = outputs['regime_probs']
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets['regimes'].cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        avg_losses = {k: v / len(val_loader) for k, v in all_losses.items()}
        
        metrics = self._compute_metrics(
            np.array(all_predictions),
            np.array(all_targets),
            np.array(all_probs) if all_probs else None
        )
        
        return avg_loss, {**avg_losses, **metrics}
    
    def _compute_metrics(self, predictions: np.ndarray, 
                        targets: np.ndarray,
                        probs: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute evaluation metrics"""
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = np.mean(predictions == targets)
        
        # Per-class metrics
        for regime in range(3):  # 0: Bull, 1: Normal, 2: Crisis
            mask = targets == regime
            if mask.sum() > 0:
                metrics[f'regime_{regime}_recall'] = np.mean(predictions[mask] == regime)
                metrics[f'regime_{regime}_precision'] = np.mean(
                    targets[predictions == regime] == regime
                ) if (predictions == regime).sum() > 0 else 0
        
        # Crisis-specific metrics
        crisis_mask = targets == 2
        if crisis_mask.sum() > 0:
            metrics['crisis_recall'] = metrics['regime_2_recall']
            metrics['crisis_precision'] = metrics['regime_2_precision']
            
            # Crisis F1
            if metrics['crisis_precision'] + metrics['crisis_recall'] > 0:
                metrics['crisis_f1'] = 2 * (
                    metrics['crisis_precision'] * metrics['crisis_recall']
                ) / (metrics['crisis_precision'] + metrics['crisis_recall'])
        
        # Confidence metrics
        if probs is not None:
            metrics['avg_confidence'] = np.mean(np.max(probs, axis=1))
            metrics['crisis_confidence'] = np.mean(probs[crisis_mask, 2]) if crisis_mask.sum() > 0 else 0
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            if epoch % self.config.val_check_interval == 0:
                val_loss, val_metrics = self.validate(val_loader)
                
                # Learning rate scheduling
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                
                # Track history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['train_metrics'].append(train_metrics)
                self.training_history['val_metrics'].append(val_metrics)
                
                # Logging
                self._log_epoch(epoch, train_loss, train_metrics, val_loss, val_metrics)
                
                # Checkpointing
                if val_loss < self.best_val_metric:
                    self.best_val_metric = val_loss
                    self.patience_counter = 0
                    if self.config.save_checkpoints:
                        self._save_checkpoint('best')
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        
        # Final evaluation
        test_loss, test_metrics = self.validate(val_loader)
        
        return {
            'best_val_loss': self.best_val_metric,
            'final_metrics': test_metrics,
            'training_history': self.training_history,
            'total_time': total_time
        }
    
    def _log_epoch(self, epoch: int, train_loss: float, train_metrics: Dict[str, float],
                  val_loss: float, val_metrics: Dict[str, float]):
        """Log epoch results"""
        lr = self.optimizer.param_groups[0]['lr']
        
        logger.info(
            f"Epoch {epoch}/{self.config.epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train Acc: {train_metrics.get('accuracy', 0):.2%}, "
            f"Val Acc: {val_metrics.get('accuracy', 0):.2%}, "
            f"Crisis Recall: {val_metrics.get('crisis_recall', 0):.2%}, "
            f"LR: {lr:.2e}"
        )
        
        # TensorBoard logging
        if hasattr(self, 'writer'):
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Metrics/train_acc', train_metrics.get('accuracy', 0), epoch)
            self.writer.add_scalar('Metrics/val_acc', val_metrics.get('accuracy', 0), epoch)
            self.writer.add_scalar('Metrics/crisis_recall', val_metrics.get('crisis_recall', 0), epoch)
            self.writer.add_scalar('LR', lr, epoch)
        
        # Weights & Biases logging
        if self.config.wandb:
            import wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_metrics.get('accuracy', 0),
                'val_acc': val_metrics.get('accuracy', 0),
                'crisis_recall': val_metrics.get('crisis_recall', 0),
                'lr': lr
            })
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config.to_dict(),
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, checkpoint_dir / f"{name}_model.pt")
        logger.info(f"Saved checkpoint: {name}_model.pt")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint['best_val_metric']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


class CrossValidator:
    """Cross-validation for RALEC-GNN"""
    
    def __init__(self, model_class: type, config: TrainingConfig):
        self.model_class = model_class
        self.config = config
        
    def cross_validate(self, data_dict: Dict[str, Any], n_splits: int = 5) -> Dict[str, Any]:
        """Perform time series cross-validation"""
        # Extract data
        sequences = data_dict['graph_sequences']
        labels = data_dict['labels']
        volatilities = data_dict['volatilities']
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=len(labels) // (n_splits + 1))
        
        if self.config.parallel_cv:
            # Parallel CV
            results = self._parallel_cv(sequences, labels, volatilities, tscv)
        else:
            # Sequential CV
            results = self._sequential_cv(sequences, labels, volatilities, tscv)
        
        # Aggregate results
        aggregated = self._aggregate_cv_results(results)
        
        return aggregated
    
    def _train_fold(self, fold_idx: int, train_data: Dict, val_data: Dict) -> Dict[str, Any]:
        """Train a single fold"""
        logger.info(f"Training fold {fold_idx + 1}/{self.config.n_splits}")
        
        # Create model instance
        model = self.model_class(
            num_features=train_data['num_features'],
            num_assets=train_data['num_assets'],
            hidden_dim=256
        )
        
        # Create data loaders
        from .data import GraphSequenceDataset
        
        train_dataset = GraphSequenceDataset(
            train_data['sequences'],
            train_data['labels'],
            train_data['volatilities']
        )
        
        val_dataset = GraphSequenceDataset(
            val_data['sequences'],
            val_data['labels'],
            val_data['volatilities']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=4
        )
        
        # Train
        trainer = Trainer(model, self.config)
        results = trainer.train(train_loader, val_loader)
        results['fold'] = fold_idx
        
        return results
    
    def _parallel_cv(self, sequences, labels, volatilities, tscv):
        """Run CV in parallel"""
        from joblib import Parallel, delayed
        
        fold_data = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(sequences)):
            train_data = {
                'sequences': [sequences[i] for i in train_idx],
                'labels': labels[train_idx],
                'volatilities': volatilities[train_idx],
                'num_features': sequences[0][0].x.shape[1],
                'num_assets': sequences[0][0].x.shape[0]
            }
            
            val_data = {
                'sequences': [sequences[i] for i in val_idx],
                'labels': labels[val_idx],
                'volatilities': volatilities[val_idx],
                'num_features': sequences[0][0].x.shape[1],
                'num_assets': sequences[0][0].x.shape[0]
            }
            
            fold_data.append((fold_idx, train_data, val_data))
        
        # Run folds in parallel
        n_jobs = self.config.cv_n_jobs if self.config.cv_n_jobs > 0 else mp.cpu_count()
        
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(self._train_fold)(fold_idx, train_data, val_data)
            for fold_idx, train_data, val_data in fold_data
        )
        
        return results
    
    def _sequential_cv(self, sequences, labels, volatilities, tscv):
        """Run CV sequentially"""
        results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(sequences)):
            train_data = {
                'sequences': [sequences[i] for i in train_idx],
                'labels': labels[train_idx],
                'volatilities': volatilities[train_idx],
                'num_features': sequences[0][0].x.shape[1],
                'num_assets': sequences[0][0].x.shape[0]
            }
            
            val_data = {
                'sequences': [sequences[i] for i in val_idx],
                'labels': labels[val_idx],
                'volatilities': volatilities[val_idx],
                'num_features': sequences[0][0].x.shape[1],
                'num_assets': sequences[0][0].x.shape[0]
            }
            
            result = self._train_fold(fold_idx, train_data, val_data)
            results.append(result)
        
        return results
    
    def _aggregate_cv_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate cross-validation results"""
        metrics_keys = ['accuracy', 'crisis_recall', 'crisis_f1']
        
        aggregated = {
            'mean_metrics': {},
            'std_metrics': {},
            'fold_results': results
        }
        
        for key in metrics_keys:
            values = []
            for result in results:
                if key in result['final_metrics']:
                    values.append(result['final_metrics'][key])
            
            if values:
                aggregated['mean_metrics'][key] = np.mean(values)
                aggregated['std_metrics'][key] = np.std(values)
        
        # Best fold
        best_fold_idx = np.argmin([r['best_val_loss'] for r in results])
        aggregated['best_fold'] = results[best_fold_idx]
        
        logger.info("Cross-validation results:")
        for metric, value in aggregated['mean_metrics'].items():
            std = aggregated['std_metrics'][metric]
            logger.info(f"  {metric}: {value:.3f} (+/- {std:.3f})")
        
        return aggregated


def visualize_training_history(history: Dict[str, List], save_path: Optional[str] = None):
    """Visualize training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    train_acc = [m.get('accuracy', 0) for m in history['train_metrics']]
    val_acc = [m.get('accuracy', 0) for m in history['val_metrics']]
    
    axes[0, 1].plot(train_acc, label='Train')
    axes[0, 1].plot(val_acc, label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Crisis recall
    crisis_recall = [m.get('crisis_recall', 0) for m in history['val_metrics']]
    
    axes[1, 0].plot(crisis_recall, color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Crisis Recall')
    axes[1, 0].set_title('Crisis Detection Performance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                  device: str = 'cuda') -> Dict[str, Any]:
    """Comprehensive model evaluation"""
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 4:
                sequences, regimes, _, _ = batch_data
            else:
                sequences, regimes, _ = batch_data
            
            regimes = regimes.to(device)
            
            # Forward pass
            outputs = model(sequences)
            
            if 'regime_logits' in outputs:
                predictions = outputs['regime_logits'].argmax(dim=1)
                probs = outputs['regime_probs']
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(regimes.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
    
    # Convert to arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    probs = np.array(all_probs) if all_probs else None
    
    # Classification report
    report = classification_report(
        targets, predictions,
        target_names=['Bull', 'Normal', 'Crisis'],
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Bull', 'Normal', 'Crisis'],
        yticklabels=['Bull', 'Normal', 'Crisis']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    return {
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': predictions,
        'targets': targets,
        'probabilities': probs
    }


# Default export
from collections import defaultdict

__all__ = [
    'TrainingConfig',
    'FocalLoss',
    'RALECLoss',
    'Trainer',
    'CrossValidator',
    'visualize_training_history',
    'evaluate_model'
]