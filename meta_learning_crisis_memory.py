#!/usr/bin/env python3
"""
RALEC-GNN Phase 5: Meta-Learning Architecture for Crisis Memory
Implements advanced meta-learning to remember and generalize from past crises,
enabling rapid adaptation to new crisis patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import copy
import logging
from datetime import datetime
from torch.nn.utils import clip_grad_norm_

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrisisEpisode:
    """Represents a single crisis episode with its characteristics"""
    episode_id: str
    start_time: datetime
    end_time: Optional[datetime]
    trigger: str  # What triggered the crisis
    affected_assets: List[int]
    contagion_path: List[Tuple[int, int]]  # Sequence of (source, target) spreads
    severity: float  # 0-1 scale
    regime_sequence: List[int]  # Regime evolution during crisis
    causal_structure: torch.Tensor  # Causal adjacency at crisis peak
    early_warning_signals: Dict[str, float]
    market_state: Dict[str, float]  # Volatility, correlation, liquidity
    learned_parameters: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class CrisisPrototype:
    """Learned prototype representing a class of similar crises"""
    prototype_id: str
    crisis_type: str  # e.g., "liquidity", "contagion", "systemic"
    episodes: List[str]  # Episode IDs in this cluster
    centroid_features: torch.Tensor
    signature_pattern: torch.Tensor  # Characteristic evolution pattern
    avg_lead_time: float
    avg_duration: float
    key_indicators: List[str]  # Most predictive indicators


class CrisisMemoryBank(nn.Module):
    """
    Episodic memory system for storing and retrieving crisis experiences.
    Uses neural episodic memory with attention-based retrieval.
    """
    
    def __init__(
        self,
        memory_size: int = 1000,
        key_dim: int = 128,
        value_dim: int = 256,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Memory components
        self.memory_keys = nn.Parameter(torch.randn(memory_size, key_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_size, value_dim))
        self.memory_ages = torch.zeros(memory_size)  # Track memory age
        
        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(value_dim, key_dim * 2),
            nn.ReLU(),
            nn.Linear(key_dim * 2, key_dim)
        )
        
        # Memory attention
        self.memory_attention = nn.MultiheadAttention(
            key_dim,
            num_heads,
            batch_first=True
        )
        
        # Memory writer
        self.memory_writer = nn.Sequential(
            nn.Linear(value_dim * 2, value_dim),
            nn.ReLU(),
            nn.Linear(value_dim, value_dim)
        )
        
        # Crisis episodes storage
        self.crisis_episodes: Dict[str, CrisisEpisode] = {}
        self.episode_embeddings: Dict[str, int] = {}  # Episode ID to memory index
        
    def store_episode(self, episode: CrisisEpisode, embedding: torch.Tensor):
        """Store a crisis episode in memory"""
        # Find least recently used slot
        oldest_idx = torch.argmax(self.memory_ages).item()
        
        # Store episode
        self.crisis_episodes[episode.episode_id] = episode
        self.episode_embeddings[episode.episode_id] = oldest_idx
        
        # Update memory
        with torch.no_grad():
            self.memory_keys.data[oldest_idx] = self.query_encoder(embedding)
            self.memory_values.data[oldest_idx] = embedding
            self.memory_ages[oldest_idx] = 0
        
        # Age all memories
        self.memory_ages += 1
        
    def retrieve_similar_episodes(
        self,
        query: torch.Tensor,
        k: int = 5
    ) -> Tuple[List[CrisisEpisode], torch.Tensor]:
        """Retrieve k most similar crisis episodes"""
        # Encode query
        query_key = self.query_encoder(query).unsqueeze(0)
        
        # Attention-based retrieval
        attended_memory, attention_weights = self.memory_attention(
            query_key,
            self.memory_keys.unsqueeze(0),
            self.memory_values.unsqueeze(0)
        )
        
        # Get top-k attention weights
        top_k_weights, top_k_indices = torch.topk(
            attention_weights.squeeze(), k
        )
        
        # Retrieve corresponding episodes
        retrieved_episodes = []
        for idx in top_k_indices:
            idx_val = idx.item()
            # Find episode with this memory index
            for episode_id, mem_idx in self.episode_embeddings.items():
                if mem_idx == idx_val and episode_id in self.crisis_episodes:
                    retrieved_episodes.append(self.crisis_episodes[episode_id])
                    break
        
        return retrieved_episodes, top_k_weights
    
    def consolidate_memories(self):
        """Consolidate similar memories to free space"""
        # Cluster similar memories
        similarity_matrix = F.cosine_similarity(
            self.memory_keys.unsqueeze(0),
            self.memory_keys.unsqueeze(1),
            dim=2
        )
        
        # Find highly similar pairs
        similar_pairs = torch.where(similarity_matrix > 0.9)
        
        # Merge similar memories (simplified)
        merged = set()
        for i, j in zip(similar_pairs[0], similar_pairs[1]):
            if i != j and i not in merged and j not in merged:
                # Average the memories
                self.memory_keys.data[i] = (
                    self.memory_keys[i] + self.memory_keys[j]
                ) / 2
                self.memory_values.data[i] = (
                    self.memory_values[i] + self.memory_values[j]
                ) / 2
                
                # Mark j as merged
                merged.add(j.item())
                self.memory_ages[j] = float('inf')  # Mark for overwriting


class PrototypeMetaLearner(nn.Module):
    """
    Meta-learning system that learns crisis prototypes and enables
    few-shot adaptation to new crisis types.
    
    Based on Prototypical Networks and MAML.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        num_prototypes: int = 10,
        adaptation_steps: int = 5
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.adaptation_steps = adaptation_steps
        
        # Feature extractor (shared across all crises)
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Prototype embeddings
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dim))
        
        # Adaptation network (for MAML-style quick adaptation)
        self.adaptation_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Task-specific heads
        self.prediction_head = nn.Linear(hidden_dim, 3)  # Regime prediction
        self.severity_head = nn.Linear(hidden_dim, 1)   # Severity estimation
        self.duration_head = nn.Linear(hidden_dim, 1)   # Duration prediction
        
        # Learned crisis prototypes
        self.learned_prototypes: Dict[str, CrisisPrototype] = {}
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract crisis-invariant features"""
        return self.feature_extractor(x)
    
    def compute_prototype_distances(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Compute distances to all prototypes"""
        # Expand for broadcasting
        features_exp = features.unsqueeze(1)  # (batch, 1, hidden)
        prototypes_exp = self.prototypes.unsqueeze(0)  # (1, num_proto, hidden)
        
        # Euclidean distance
        distances = torch.norm(features_exp - prototypes_exp, dim=2)
        
        return distances
    
    def classify_crisis_type(
        self,
        features: torch.Tensor
    ) -> Tuple[int, torch.Tensor]:
        """Classify crisis into prototype categories"""
        distances = self.compute_prototype_distances(features)
        
        # Convert distances to similarities
        similarities = 1 / (1 + distances)
        
        # Softmax over prototypes
        prototype_probs = F.softmax(similarities, dim=1)
        
        # Get most likely prototype
        best_prototype = torch.argmax(prototype_probs, dim=1)
        
        return best_prototype, prototype_probs
    
    def adapt_to_new_crisis(
        self,
        support_set: Tuple[torch.Tensor, torch.Tensor],
        query_set: torch.Tensor,
        num_adaptation_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        MAML-style few-shot adaptation to new crisis pattern.
        
        Args:
            support_set: (features, labels) for few examples of new crisis
            query_set: Features to make predictions on
            num_adaptation_steps: Override default adaptation steps
        """
        if num_adaptation_steps is None:
            num_adaptation_steps = self.adaptation_steps
        
        support_features, support_labels = support_set
        
        # Clone model for adaptation
        adapted_model = copy.deepcopy(self)
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=0.01
        )
        
        # Inner loop adaptation
        for _ in range(num_adaptation_steps):
            # Forward pass on support set
            support_embeddings = adapted_model.extract_features(support_features)
            support_logits = adapted_model.prediction_head(support_embeddings)
            
            # Compute loss
            inner_loss = F.cross_entropy(support_logits, support_labels)
            
            # Update adapted model
            inner_optimizer.zero_grad()
            inner_loss.backward()
            inner_optimizer.step()
        
        # Make predictions on query set with adapted model
        with torch.no_grad():
            query_embeddings = adapted_model.extract_features(query_set)
            query_predictions = adapted_model.prediction_head(query_embeddings)
        
        return query_predictions
    
    def learn_prototype(
        self,
        episodes: List[CrisisEpisode],
        prototype_id: str
    ) -> CrisisPrototype:
        """Learn a prototype from a cluster of similar crisis episodes"""
        # Extract features from all episodes
        all_features = []
        all_patterns = []
        
        for episode in episodes:
            # Convert episode to features
            features = self._episode_to_features(episode)
            all_features.append(features)
            
            # Extract temporal pattern
            pattern = torch.tensor(episode.regime_sequence)
            all_patterns.append(pattern)
        
        # Compute centroid
        features_tensor = torch.stack(all_features)
        centroid = features_tensor.mean(dim=0)
        
        # Find most representative pattern
        patterns_tensor = torch.nn.utils.rnn.pad_sequence(
            all_patterns, batch_first=True
        )
        signature_pattern = patterns_tensor[0]  # Simplified: take first
        
        # Identify key indicators
        indicator_importance = defaultdict(float)
        for episode in episodes:
            for indicator, value in episode.early_warning_signals.items():
                indicator_importance[indicator] += value
        
        key_indicators = sorted(
            indicator_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        prototype = CrisisPrototype(
            prototype_id=prototype_id,
            crisis_type=self._infer_crisis_type(episodes),
            episodes=[ep.episode_id for ep in episodes],
            centroid_features=centroid,
            signature_pattern=signature_pattern,
            avg_lead_time=np.mean([
                self._compute_lead_time(ep) for ep in episodes
            ]),
            avg_duration=np.mean([
                (ep.end_time - ep.start_time).total_seconds() / 86400
                for ep in episodes if ep.end_time
            ]),
            key_indicators=[ind[0] for ind in key_indicators]
        )
        
        self.learned_prototypes[prototype_id] = prototype
        return prototype
    
    def _episode_to_features(self, episode: CrisisEpisode) -> torch.Tensor:
        """Convert crisis episode to feature vector"""
        features = []
        
        # Market state features
        features.extend([
            episode.market_state.get('volatility', 0),
            episode.market_state.get('correlation', 0),
            episode.market_state.get('liquidity', 1)
        ])
        
        # Severity and spread
        features.extend([
            episode.severity,
            len(episode.affected_assets) / 100,  # Normalize
            len(episode.contagion_path) / 100   # Normalize
        ])
        
        # Early warning summary
        warning_values = list(episode.early_warning_signals.values())
        if warning_values:
            features.extend([
                np.mean(warning_values),
                np.max(warning_values),
                np.std(warning_values)
            ])
        else:
            features.extend([0, 0, 0])
        
        # Causal structure summary
        if episode.causal_structure is not None:
            causal_density = (
                episode.causal_structure > 0
            ).float().mean().item()
            causal_strength = episode.causal_structure[
                episode.causal_structure > 0
            ].mean().item() if (episode.causal_structure > 0).any() else 0
        else:
            causal_density = 0
            causal_strength = 0
        
        features.extend([causal_density, causal_strength])
        
        # Pad to feature dimension
        while len(features) < self.feature_dim:
            features.append(0)
        
        return torch.tensor(features[:self.feature_dim], dtype=torch.float32)
    
    def _infer_crisis_type(self, episodes: List[CrisisEpisode]) -> str:
        """Infer crisis type from episode characteristics"""
        # Simple heuristic based on triggers
        trigger_counts = defaultdict(int)
        for ep in episodes:
            trigger_counts[ep.trigger] += 1
        
        most_common_trigger = max(trigger_counts, key=trigger_counts.get)
        
        # Map triggers to crisis types
        trigger_to_type = {
            'liquidity_shock': 'liquidity',
            'credit_event': 'credit',
            'contagion': 'contagion',
            'systemic': 'systemic',
            'volatility_spike': 'volatility'
        }
        
        return trigger_to_type.get(most_common_trigger, 'unknown')
    
    def _compute_lead_time(self, episode: CrisisEpisode) -> float:
        """Compute how much advance warning was available"""
        # Find when warnings exceeded threshold
        warning_values = list(episode.early_warning_signals.values())
        if warning_values:
            max_warning = max(warning_values)
            # Simple estimate: higher warning = more lead time
            return max_warning * 20  # Scale to days
        return 0


class CrisisPatternMatcher(nn.Module):
    """
    Matches current market conditions to historical crisis patterns
    using neural pattern matching and similarity learning.
    """
    
    def __init__(
        self,
        pattern_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8
    ):
        super().__init__()
        
        # Pattern encoder
        self.pattern_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=pattern_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(pattern_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Pattern memory
        self.pattern_memory = deque(maxlen=100)
        
    def encode_pattern(self, sequence: torch.Tensor) -> torch.Tensor:
        """Encode temporal pattern into fixed representation"""
        # Add positional encoding
        seq_len = sequence.shape[1]
        pos_encoding = self._get_positional_encoding(seq_len, sequence.shape[2])
        sequence_with_pos = sequence + pos_encoding.to(sequence.device)
        
        # Encode through transformer
        encoded = self.pattern_encoder(sequence_with_pos)
        
        # Pool over time dimension
        pattern_embedding = encoded.mean(dim=1)
        
        return pattern_embedding
    
    def compute_similarity(
        self,
        pattern1: torch.Tensor,
        pattern2: torch.Tensor
    ) -> torch.Tensor:
        """Compute learned similarity between two patterns"""
        combined = torch.cat([pattern1, pattern2], dim=-1)
        similarity = self.similarity_net(combined)
        return similarity
    
    def match_to_historical(
        self,
        current_pattern: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[float, Any]]:
        """Find most similar historical patterns"""
        current_embedding = self.encode_pattern(current_pattern)
        
        similarities = []
        for historical in self.pattern_memory:
            hist_embedding = self.encode_pattern(historical['pattern'])
            sim = self.compute_similarity(
                current_embedding,
                hist_embedding
            )
            similarities.append((sim.item(), historical))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return similarities[:top_k]
    
    def _get_positional_encoding(
        self,
        seq_len: int,
        d_model: int
    ) -> torch.Tensor:
        """Generate positional encoding"""
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(np.log(10000.0) / d_model)
        )
        
        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)


class MetaLearningCrisisMemory(nn.Module):
    """
    Complete meta-learning system for crisis memory and adaptation.
    Integrates memory bank, prototype learning, and pattern matching.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        memory_size: int = 1000,
        num_prototypes: int = 10
    ):
        super().__init__()
        
        # Components
        self.memory_bank = CrisisMemoryBank(
            memory_size=memory_size,
            key_dim=hidden_dim // 2,
            value_dim=hidden_dim
        )
        
        self.prototype_learner = PrototypeMetaLearner(
            feature_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            num_prototypes=num_prototypes
        )
        
        self.pattern_matcher = CrisisPatternMatcher(
            pattern_dim=hidden_dim // 2,
            hidden_dim=hidden_dim
        )
        
        # Crisis encoder
        self.crisis_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Meta-prediction network
        self.meta_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # Current + retrieved + prototype
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # Predictions
        )
        
        # Crisis statistics
        self.crisis_counter = 0
        self.adaptation_success_rate = deque(maxlen=100)
        
    def process_crisis_observation(
        self,
        observation: torch.Tensor,
        is_crisis: bool = False,
        episode_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process current observation with crisis memory.
        
        Args:
            observation: Current market features
            is_crisis: Whether currently in crisis
            episode_info: Additional crisis information if available
        """
        # Encode current observation
        current_encoding = self.crisis_encoder(observation)
        
        # Retrieve similar historical episodes
        similar_episodes, retrieval_scores = self.memory_bank.retrieve_similar_episodes(
            current_encoding,
            k=5
        )
        
        # Extract prototype features
        prototype_features = self.prototype_learner.extract_features(
            current_encoding
        )
        prototype_idx, prototype_probs = self.prototype_learner.classify_crisis_type(
            prototype_features
        )
        
        # Pattern matching (if we have temporal context)
        pattern_matches = []
        if hasattr(self, 'recent_observations'):
            recent_pattern = torch.stack(self.recent_observations[-20:])
            pattern_matches = self.pattern_matcher.match_to_historical(
                recent_pattern.unsqueeze(0)
            )
        
        # Combine information for meta-prediction
        if similar_episodes:
            # Average retrieved episode features
            retrieved_features = torch.stack([
                self.crisis_encoder(
                    self.prototype_learner._episode_to_features(ep)
                )
                for ep in similar_episodes[:3]  # Top 3
            ]).mean(dim=0)
        else:
            retrieved_features = torch.zeros_like(current_encoding)
        
        # Get prototype embedding
        prototype_embedding = self.prototype_learner.prototypes[prototype_idx]
        
        # Meta-prediction
        combined_features = torch.cat([
            current_encoding,
            retrieved_features,
            prototype_embedding
        ])
        
        meta_predictions = self.meta_predictor(combined_features)
        
        # Parse predictions
        regime_logits = meta_predictions[:3]
        crisis_score = torch.sigmoid(meta_predictions[3])
        
        # Store episode if in crisis
        if is_crisis and episode_info:
            self._store_crisis_episode(current_encoding, episode_info)
        
        # Update recent observations
        if not hasattr(self, 'recent_observations'):
            self.recent_observations = deque(maxlen=100)
        self.recent_observations.append(observation)
        
        return {
            'regime_predictions': F.softmax(regime_logits, dim=0),
            'crisis_probability': crisis_score,
            'prototype_match': prototype_idx,
            'prototype_confidence': prototype_probs.max(),
            'similar_episodes': similar_episodes,
            'retrieval_scores': retrieval_scores,
            'pattern_matches': pattern_matches[:3] if pattern_matches else [],
            'meta_features': combined_features
        }
    
    def adapt_to_new_crisis(
        self,
        crisis_observations: List[torch.Tensor],
        crisis_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Rapidly adapt to new crisis using meta-learning.
        """
        if len(crisis_observations) < 3:
            return {'adapted': False, 'reason': 'Insufficient observations'}
        
        # Prepare support set
        support_features = torch.stack(crisis_observations[:5])
        
        if crisis_labels:
            support_labels = torch.tensor(crisis_labels[:5])
        else:
            # Assume crisis regime (2) for all
            support_labels = torch.full((5,), 2)
        
        # Prepare query set
        query_features = torch.stack(crisis_observations[5:10]) \
            if len(crisis_observations) >= 10 else support_features
        
        # Adapt prototype learner
        adapted_predictions = self.prototype_learner.adapt_to_new_crisis(
            (support_features, support_labels),
            query_features
        )
        
        # Evaluate adaptation
        if crisis_labels and len(crisis_labels) >= 10:
            query_labels = torch.tensor(crisis_labels[5:10])
            accuracy = (
                adapted_predictions.argmax(dim=1) == query_labels
            ).float().mean()
            self.adaptation_success_rate.append(accuracy.item())
        
        return {
            'adapted': True,
            'adapted_predictions': adapted_predictions,
            'adaptation_steps': self.prototype_learner.adaptation_steps,
            'success_rate': np.mean(self.adaptation_success_rate) 
                if self.adaptation_success_rate else 0
        }
    
    def _store_crisis_episode(
        self,
        encoding: torch.Tensor,
        episode_info: Dict[str, Any]
    ):
        """Store new crisis episode in memory"""
        self.crisis_counter += 1
        
        # Create episode
        episode = CrisisEpisode(
            episode_id=f"crisis_{self.crisis_counter}",
            start_time=datetime.now(),
            end_time=None,
            trigger=episode_info.get('trigger', 'unknown'),
            affected_assets=episode_info.get('affected_assets', []),
            contagion_path=episode_info.get('contagion_path', []),
            severity=episode_info.get('severity', 0.5),
            regime_sequence=episode_info.get('regime_sequence', []),
            causal_structure=episode_info.get('causal_structure'),
            early_warning_signals=episode_info.get('early_warnings', {}),
            market_state=episode_info.get('market_state', {})
        )
        
        # Store in memory bank
        self.memory_bank.store_episode(episode, encoding)
        
        # Update pattern memory
        if 'pattern' in episode_info:
            self.pattern_matcher.pattern_memory.append({
                'pattern': episode_info['pattern'],
                'episode_id': episode.episode_id,
                'timestamp': datetime.now()
            })
    
    def consolidate_learning(self):
        """
        Consolidate learned experiences into prototypes.
        Called periodically or after major events.
        """
        # Consolidate memory bank
        self.memory_bank.consolidate_memories()
        
        # Cluster episodes and learn prototypes
        if len(self.memory_bank.crisis_episodes) >= 5:
            # Simple clustering based on triggers
            clusters = defaultdict(list)
            for episode in self.memory_bank.crisis_episodes.values():
                clusters[episode.trigger].append(episode)
            
            # Learn prototype for each cluster
            for trigger, episodes in clusters.items():
                if len(episodes) >= 3:
                    prototype_id = f"proto_{trigger}_{len(self.prototype_learner.learned_prototypes)}"
                    self.prototype_learner.learn_prototype(
                        episodes,
                        prototype_id
                    )
        
        logger.info(f"Consolidation complete. "
                   f"Episodes: {len(self.memory_bank.crisis_episodes)}, "
                   f"Prototypes: {len(self.prototype_learner.learned_prototypes)}")


class CrisisAnticipationModule(nn.Module):
    """
    Uses meta-learned knowledge to anticipate and prepare for crises.
    """
    
    def __init__(self, memory_system: MetaLearningCrisisMemory):
        super().__init__()
        self.memory_system = memory_system
        
        # Anticipation network
        self.anticipation_net = nn.Sequential(
            nn.Linear(256 * 2, 256),  # Current + historical
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # Multiple anticipation outputs
        )
        
    def anticipate_crisis(
        self,
        current_state: torch.Tensor,
        historical_context: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Anticipate potential crisis based on meta-learned patterns.
        """
        # Process through memory system
        memory_output = self.memory_system.process_crisis_observation(
            current_state
        )
        
        # Get historical encoding
        if historical_context:
            hist_encoding = torch.stack(historical_context[-10:]).mean(dim=0)
        else:
            hist_encoding = torch.zeros_like(current_state)
        
        # Anticipation predictions
        combined = torch.cat([
            memory_output['meta_features'][:256],  # First 256 dims
            self.memory_system.crisis_encoder(hist_encoding)
        ])
        
        anticipation_output = self.anticipation_net(combined)
        
        # Parse outputs
        time_to_crisis = torch.sigmoid(anticipation_output[0]) * 30  # Days
        crisis_type_logits = anticipation_output[1:4]
        preparation_urgency = torch.sigmoid(anticipation_output[4])
        
        # Get recommendations from similar episodes
        recommendations = []
        for episode in memory_output['similar_episodes'][:3]:
            if hasattr(episode, 'learned_parameters'):
                recommendations.append({
                    'from_episode': episode.episode_id,
                    'severity': episode.severity,
                    'duration': (episode.end_time - episode.start_time).days
                        if episode.end_time else 'ongoing',
                    'key_indicators': episode.early_warning_signals
                })
        
        return {
            'time_to_crisis': time_to_crisis.item(),
            'crisis_type_probs': F.softmax(crisis_type_logits, dim=0),
            'preparation_urgency': preparation_urgency.item(),
            'similar_historical': recommendations,
            'confidence': memory_output['prototype_confidence'].item()
        }


if __name__ == "__main__":
    # Test meta-learning system
    logger.info("Testing Meta-Learning Crisis Memory System...")
    
    # Initialize
    memory_system = MetaLearningCrisisMemory(
        input_dim=16,
        hidden_dim=256,
        memory_size=100,
        num_prototypes=5
    )
    
    # Simulate crisis observations
    n_observations = 50
    observations = []
    
    # Normal period
    for t in range(20):
        obs = torch.randn(16) * 0.1
        observations.append(obs)
        result = memory_system.process_crisis_observation(obs, is_crisis=False)
    
    # Crisis period
    crisis_start = 20
    for t in range(crisis_start, 40):
        obs = torch.randn(16) * 0.5 + torch.sin(torch.tensor(t * 0.1))
        observations.append(obs)
        
        episode_info = {
            'trigger': 'volatility_spike',
            'affected_assets': list(range(10)),
            'severity': 0.7,
            'early_warnings': {
                'autocorrelation': 0.85,
                'variance': 2.1
            },
            'market_state': {
                'volatility': 0.45,
                'correlation': 0.72,
                'liquidity': 0.3
            }
        }
        
        result = memory_system.process_crisis_observation(
            obs, is_crisis=True, episode_info=episode_info
        )
        
        if t == crisis_start + 5:
            logger.info(f"Crisis detection: {result['crisis_probability']:.3f}")
            logger.info(f"Prototype match: {result['prototype_match']}")
    
    # Test adaptation
    logger.info("\nTesting rapid adaptation...")
    adaptation_result = memory_system.adapt_to_new_crisis(
        observations[crisis_start:crisis_start+10]
    )
    logger.info(f"Adaptation successful: {adaptation_result['adapted']}")
    
    # Consolidate learning
    memory_system.consolidate_learning()
    
    # Test anticipation
    anticipation = CrisisAnticipationModule(memory_system)
    future_crisis = anticipation.anticipate_crisis(
        observations[-1],
        observations[-20:]
    )
    
    logger.info(f"\nCrisis Anticipation:")
    logger.info(f"Time to crisis: {future_crisis['time_to_crisis']:.1f} days")
    logger.info(f"Preparation urgency: {future_crisis['preparation_urgency']:.3f}")
    logger.info(f"Confidence: {future_crisis['confidence']:.3f}")