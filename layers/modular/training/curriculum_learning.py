"""
Curriculum Learning Strategies

Modular curriculum learning components for progressive training of time series models.
Different strategies for gradually increasing training complexity.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Callable
from abc import ABC, abstractmethod
import numpy as np
import math


class BaseCurriculumStrategy(ABC):
    """Abstract base class for curriculum learning strategies."""
    
    def __init__(self, name: str, total_epochs: int, warmup_epochs: int = 10):
        self.name = name
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
    @abstractmethod
    def get_curriculum_params(self, epoch: int) -> Dict[str, Any]:
        """Get curriculum parameters for the given epoch."""
        pass
    
    def update_epoch(self, epoch: int):
        """Update current epoch."""
        self.current_epoch = epoch
    
    def get_progress(self) -> float:
        """Get training progress [0, 1]."""
        return min(1.0, max(0.0, (self.current_epoch - self.warmup_epochs) / 
                           (self.total_epochs - self.warmup_epochs)))


class SequenceLengthCurriculum(BaseCurriculumStrategy):
    """Curriculum that gradually increases sequence length."""
    
    def __init__(self, total_epochs: int, min_seq_len: int = 24, max_seq_len: int = 96, 
                 growth_strategy: str = 'linear'):
        super().__init__('sequence_length', total_epochs)
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.growth_strategy = growth_strategy
        
    def get_curriculum_params(self, epoch: int) -> Dict[str, Any]:
        """Get sequence length for current epoch."""
        progress = self.get_progress()
        
        if self.growth_strategy == 'linear':
            seq_len = self.min_seq_len + progress * (self.max_seq_len - self.min_seq_len)
        elif self.growth_strategy == 'exponential':
            seq_len = self.min_seq_len * (self.max_seq_len / self.min_seq_len) ** progress
        elif self.growth_strategy == 'step':
            # Step-wise increase every 25% of training
            step = int(progress * 4)
            seq_len = self.min_seq_len + step * (self.max_seq_len - self.min_seq_len) / 4
        else:
            seq_len = self.max_seq_len
        
        return {
            'seq_len': int(seq_len),
            'progress': progress,
            'strategy': self.growth_strategy
        }


class ComplexityCurriculum(BaseCurriculumStrategy):
    """Curriculum that gradually increases model complexity."""
    
    def __init__(self, total_epochs: int, min_experts: int = 1, max_experts: int = 4,
                 min_layers: int = 1, max_layers: int = 3):
        super().__init__('complexity', total_epochs)
        self.min_experts = min_experts
        self.max_experts = max_experts
        self.min_layers = min_layers
        self.max_layers = max_layers
        
    def get_curriculum_params(self, epoch: int) -> Dict[str, Any]:
        """Get model complexity parameters for current epoch."""
        progress = self.get_progress()
        
        # Gradually increase number of active experts
        num_experts = self.min_experts + int(progress * (self.max_experts - self.min_experts))
        
        # Gradually increase number of layers
        num_layers = self.min_layers + int(progress * (self.max_layers - self.min_layers))
        
        # Gradually increase top-k for MoE
        top_k = max(1, min(num_experts, 1 + int(progress * 2)))
        
        return {
            'num_active_experts': num_experts,
            'num_layers': num_layers,
            'top_k': top_k,
            'progress': progress
        }


class UncertaintyCurriculum(BaseCurriculumStrategy):
    """Curriculum based on prediction uncertainty."""
    
    def __init__(self, total_epochs: int, uncertainty_threshold: float = 0.5,
                 adaptation_rate: float = 0.1):
        super().__init__('uncertainty', total_epochs)
        self.uncertainty_threshold = uncertainty_threshold
        self.adaptation_rate = adaptation_rate
        self.uncertainty_history = []
        
    def get_curriculum_params(self, epoch: int) -> Dict[str, Any]:
        """Get curriculum parameters based on uncertainty."""
        progress = self.get_progress()
        
        # Adaptive threshold based on uncertainty history
        if self.uncertainty_history:
            avg_uncertainty = np.mean(self.uncertainty_history[-10:])  # Last 10 epochs
            adaptive_threshold = self.uncertainty_threshold * (1 + self.adaptation_rate * avg_uncertainty)
        else:
            adaptive_threshold = self.uncertainty_threshold
        
        # Difficulty increases as uncertainty decreases
        difficulty_multiplier = max(0.5, 1.0 - adaptive_threshold)
        
        return {
            'uncertainty_threshold': adaptive_threshold,
            'difficulty_multiplier': difficulty_multiplier,
            'progress': progress,
            'avg_uncertainty': np.mean(self.uncertainty_history[-5:]) if self.uncertainty_history else 0.0
        }
    
    def update_uncertainty(self, uncertainty: float):
        """Update uncertainty history."""
        self.uncertainty_history.append(uncertainty)
        # Keep only recent history
        if len(self.uncertainty_history) > 100:
            self.uncertainty_history = self.uncertainty_history[-100:]


class MultiModalCurriculum(BaseCurriculumStrategy):
    """Curriculum that combines multiple strategies."""
    
    def __init__(self, total_epochs: int, strategies: List[BaseCurriculumStrategy],
                 weights: Optional[List[float]] = None):
        super().__init__('multimodal', total_epochs)
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
    def get_curriculum_params(self, epoch: int) -> Dict[str, Any]:
        """Get combined curriculum parameters."""
        combined_params = {'progress': self.get_progress()}
        
        for strategy, weight in zip(self.strategies, self.weights):
            strategy_params = strategy.get_curriculum_params(epoch)
            
            # Weight and combine parameters
            for key, value in strategy_params.items():
                if key == 'progress':
                    continue
                
                weighted_key = f"{strategy.name}_{key}"
                combined_params[weighted_key] = value
                
                # Also create weighted average for numeric values
                if isinstance(value, (int, float)):
                    avg_key = f"avg_{key}"
                    if avg_key not in combined_params:
                        combined_params[avg_key] = 0.0
                    combined_params[avg_key] += weight * value
        
        return combined_params


class DataDifficultyRanker:
    """Ranks data samples by difficulty for curriculum learning."""
    
    def __init__(self, ranking_strategy: str = 'loss_based'):
        self.ranking_strategy = ranking_strategy
        self.sample_difficulties = {}
        
    def update_difficulties(self, sample_ids: List[str], losses: List[float]):
        """Update difficulty scores based on losses."""
        for sample_id, loss in zip(sample_ids, losses):
            if sample_id not in self.sample_difficulties:
                self.sample_difficulties[sample_id] = []
            
            self.sample_difficulties[sample_id].append(loss)
            
            # Keep only recent history
            if len(self.sample_difficulties[sample_id]) > 10:
                self.sample_difficulties[sample_id] = self.sample_difficulties[sample_id][-10:]
    
    def get_difficulty_score(self, sample_id: str) -> float:
        """Get difficulty score for a sample."""
        if sample_id not in self.sample_difficulties:
            return 0.5  # Default medium difficulty
        
        losses = self.sample_difficulties[sample_id]
        
        if self.ranking_strategy == 'loss_based':
            return np.mean(losses)
        elif self.ranking_strategy == 'variance_based':
            return np.var(losses)
        elif self.ranking_strategy == 'trend_based':
            if len(losses) < 2:
                return np.mean(losses)
            # Increasing loss trend indicates difficulty
            trend = np.polyfit(range(len(losses)), losses, 1)[0]
            return max(0.0, trend)
        else:
            return np.mean(losses)
    
    def rank_samples(self, sample_ids: List[str], ascending: bool = True) -> List[str]:
        """Rank samples by difficulty."""
        difficulties = [(sample_id, self.get_difficulty_score(sample_id)) 
                       for sample_id in sample_ids]
        
        difficulties.sort(key=lambda x: x[1], reverse=not ascending)
        return [sample_id for sample_id, _ in difficulties]


class CurriculumScheduler:
    """Scheduler that manages curriculum progression."""
    
    def __init__(self, strategy: BaseCurriculumStrategy, 
                 data_ranker: Optional[DataDifficultyRanker] = None):
        self.strategy = strategy
        self.data_ranker = data_ranker
        self.epoch_history = []
        
    def step(self, epoch: int, model_performance: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Step the curriculum scheduler."""
        self.strategy.update_epoch(epoch)
        curriculum_params = self.strategy.get_curriculum_params(epoch)
        
        # Update uncertainty curriculum if applicable
        if isinstance(self.strategy, UncertaintyCurriculum) and model_performance:
            if 'uncertainty' in model_performance:
                self.strategy.update_uncertainty(model_performance['uncertainty'])
        
        # Store history
        self.epoch_history.append({
            'epoch': epoch,
            'params': curriculum_params,
            'performance': model_performance
        })
        
        return curriculum_params
    
    def get_training_samples(self, all_sample_ids: List[str], 
                           curriculum_params: Dict[str, Any]) -> List[str]:
        """Get training samples based on curriculum parameters."""
        if self.data_ranker is None:
            return all_sample_ids
        
        # Determine how many samples to use based on progress
        progress = curriculum_params.get('progress', 1.0)
        num_samples = max(1, int(len(all_sample_ids) * (0.1 + 0.9 * progress)))
        
        # Rank samples by difficulty
        ranked_samples = self.data_ranker.rank_samples(all_sample_ids, ascending=True)
        
        # Return appropriate subset
        return ranked_samples[:num_samples]
    
    def get_curriculum_summary(self) -> Dict[str, Any]:
        """Get summary of curriculum progression."""
        if not self.epoch_history:
            return {}
        
        recent_history = self.epoch_history[-10:]  # Last 10 epochs
        
        summary = {
            'strategy_name': self.strategy.name,
            'total_epochs': len(self.epoch_history),
            'current_progress': self.strategy.get_progress(),
            'recent_params': recent_history[-1]['params'] if recent_history else {},
        }
        
        # Add performance trends if available
        if all('performance' in h and h['performance'] for h in recent_history):
            performances = [h['performance'] for h in recent_history]
            
            # Calculate trends for different metrics
            for metric in performances[0].keys():
                values = [p[metric] for p in performances if metric in p]
                if len(values) > 1:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    summary[f'{metric}_trend'] = trend
        
        return summary


class CurriculumLearningRegistry:
    """Registry for curriculum learning strategies."""
    
    def __init__(self):
        self._strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default curriculum strategies."""
        self.register('sequence_length', SequenceLengthCurriculum)
        self.register('complexity', ComplexityCurriculum)
        self.register('uncertainty', UncertaintyCurriculum)
        self.register('multimodal', MultiModalCurriculum)
    
    def register(self, name: str, strategy_class: type):
        """Register a curriculum strategy."""
        self._strategies[name] = strategy_class
    
    def create(self, name: str, **kwargs) -> BaseCurriculumStrategy:
        """Create a curriculum strategy instance."""
        if name not in self._strategies:
            raise ValueError(f"Strategy '{name}' not found. Available: {list(self._strategies.keys())}")
        
        return self._strategies[name](**kwargs)
    
    def list_strategies(self) -> List[str]:
        """List available strategies."""
        return list(self._strategies.keys())


# Global registry instance
curriculum_registry = CurriculumLearningRegistry()


def create_curriculum_strategy(name: str, **kwargs) -> BaseCurriculumStrategy:
    """Create curriculum strategy from global registry."""
    return curriculum_registry.create(name, **kwargs)


def create_adaptive_curriculum(total_epochs: int, config: Dict[str, Any]) -> CurriculumScheduler:
    """Create an adaptive curriculum based on configuration."""
    
    # Determine which strategies to use
    strategies = []
    
    if config.get('use_sequence_curriculum', True):
        seq_strategy = create_curriculum_strategy(
            'sequence_length',
            total_epochs=total_epochs,
            min_seq_len=config.get('min_seq_len', 24),
            max_seq_len=config.get('max_seq_len', 96),
            growth_strategy=config.get('seq_growth_strategy', 'linear')
        )
        strategies.append(seq_strategy)
    
    if config.get('use_complexity_curriculum', True):
        complexity_strategy = create_curriculum_strategy(
            'complexity',
            total_epochs=total_epochs,
            min_experts=config.get('min_experts', 1),
            max_experts=config.get('max_experts', 4)
        )
        strategies.append(complexity_strategy)
    
    if config.get('use_uncertainty_curriculum', False):
        uncertainty_strategy = create_curriculum_strategy(
            'uncertainty',
            total_epochs=total_epochs,
            uncertainty_threshold=config.get('uncertainty_threshold', 0.5)
        )
        strategies.append(uncertainty_strategy)
    
    # Create combined strategy if multiple strategies
    if len(strategies) > 1:
        weights = config.get('curriculum_weights', None)
        combined_strategy = create_curriculum_strategy(
            'multimodal',
            total_epochs=total_epochs,
            strategies=strategies,
            weights=weights
        )
    else:
        combined_strategy = strategies[0] if strategies else None
    
    # Create data ranker if requested
    data_ranker = None
    if config.get('use_data_ranking', False):
        data_ranker = DataDifficultyRanker(
            ranking_strategy=config.get('ranking_strategy', 'loss_based')
        )
    
    return CurriculumScheduler(combined_strategy, data_ranker)