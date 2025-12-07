from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch

class CurriculumLearning(ABC):
    """
    Abstract Base Class for Curriculum Learning strategies in Time Series Forecasting.
    """
    def __init__(self, config: Any, device: torch.device):
        self.config = config
        self.device = device
        self.current_epoch = 0
        self.total_epochs = getattr(config, 'train_epochs', 1)

    def update_epoch(self, epoch: int):
        """Called at the start of each epoch to update internal state."""
        self.current_epoch = epoch

    @abstractmethod
    def get_mask(self, pred_len: int) -> Optional[torch.Tensor]:
        """
        Returns a mask of shape [pred_len] (or broadcastable) to apply to logical loss.
        1.0 means include, 0.0 means exclude. 
        Returns None if no masking should be applied.
        """
        pass

    def on_batch_start(self, batch_x, batch_y):
        """Optional hook for batch-level curriculum updates."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics for logging."""
        return {}
